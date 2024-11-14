#!/usr/bin/env python3
import glob
import heapq
import datetime
import tqdm
import numpy as np
import databento as db
import databento_dbn as dbn
import databento_classes
import pandas as pd

def parse_folder(data_path = "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/"):
    for file in sorted(glob.glob(data_path+'*.zst')):
        for result in process_mbos(file):
            yield result

def parse_file(file = "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/glbx-mdp3-20240904.mbo.dbn.zst"):
    data = db.DBNStore.from_file(file)

    market = databento_classes.Market()

    instrument_map = db.common.symbology.InstrumentMap()
    instrument_map.insert_metadata(data.metadata)
    instrument_codes = dict()

    start = data.metadata.start

    cutoff = 1726916400
    for mbo in data:
        market.apply(mbo)

        if mbo.instrument_id in instrument_codes:
            instrument = instrument_codes[mbo.instrument_id]
        else:
            ticker = instrument_map.resolve(mbo.instrument_id, datetime.datetime.fromtimestamp(mbo.ts_event*1e-9).date())
            if ticker:
                instrument_codes[mbo.instrument_id] = ticker
                instrument = ticker
            else:
                instrument = None

        is_sept_expiry = instrument in ('MESU4','ESU4')
        is_dec_expiry = instrument in ('MESZ4','ESZ4')

        if (mbo.ts_event * 1e-9 < cutoff and is_sept_expiry) or (mbo.ts_event * 1e-9 > cutoff and is_dec_expiry):
            if data.metadata.start <= mbo.ts_event <= data.metadata.end:
                yield (mbo,market,instrument[:-2],(mbo.ts_event-start)*1e-9)
        """
        print([str(o) for l in market.get_books_by_pub(mbo.instrument_id)[1].bids.values() for o in l.orders])
        # If it's the last update in an event, print the state of the aggregated book
        if mbo.flags & db.RecordFlags.F_LAST:
            symbol = (
                instrument_map.resolve(mbo.instrument_id, mbo.pretty_ts_recv.date())
                or ""
            )
            print(f"{symbol} Aggregated BBO | {mbo.pretty_ts_recv}")
            best_bid, best_offer = market.aggregated_bbo(mbo.instrument_id)
            print(f"    {best_offer}")
            print(f"    {best_bid}")
        """

def get_events(df):
    trade_order_id = df.order_id[df.action=='T'].iloc[0] if 'T' in df.action.unique() else None
    df = df[(df.action=='T') | (df.order_id != trade_order_id)]
    filled_orders = set(df[(df.action=='F') & (df.order_id != trade_order_id)].order_id.values)
    trade_qty = df[df.action=='T']['size'].sum()
    fill_qty = df[df.order_id.apply(filled_orders.__contains__) & (df.action=='F')]['size'].sum()
    cancel_qty = df[df.order_id.apply(filled_orders.__contains__) & (df.action=='C')]['size'].sum()
    try:
        assert trade_qty == fill_qty >= cancel_qty, df
    except Exception as e:
        print(e)
    trade_event = df[df.order_id.apply(filled_orders.__contains__) | (df.order_id==trade_order_id)]
    other_events = df[(df.action!='T') & (~df.order_id.apply(filled_orders.__contains__)) & (df.order_id!=trade_order_id)]
    yield trade_event
    if (other_events.action=='C').all() and len(other_events.side.unique())==1:
        yield other_events
    else:
        for i in range(other_events.shape[0]):
            yield other_events.iloc[i:i+1]

def split_events(df,condition):
    #(~((df.action==df.action.shift(1)) & (df.side==df.side.shift(1))))
    indices = list(df[condition].index)
    for prev_trade_index,trade_index in zip([-1]+indices,indices+[np.inf]):
        new_df = df[(prev_trade_index <= df.index) & (df.index < trade_index)]
        if new_df.shape[0] > 0:
            assert (new_df.action=='T').sum() <= 1, new_df
            yield new_df

#def process_day(data,interval = None):
def process_day(data,interval = pd.Interval(36000,37200)):
    events = None
    prev_ts_event = None
    for mbo,market,instrument,seconds in data:
        if interval is None or seconds in interval:
            event = dict(instrument_id=mbo.instrument_id, ts_event=mbo.ts_event, action=mbo.action, size=mbo.size, side=mbo.side, order_id=mbo.order_id, sequence=mbo.sequence, price=mbo.price)
            if mbo.action=='T' or mbo.ts_event != prev_ts_event:
                if events is not None:
                    assert len(events)>0, events
                if events is None:
                    events = []
                if len(events)>0:
                    df = pd.DataFrame(events)
                    for cascade in get_events(df):
                        for subevent in split_events(cascade, condition = cascade.action=='T'):
                            yield (subevent,market,instrument,seconds)
                    events.clear()
            events.append(event)
            prev_ts_event = mbo.ts_event

def process_mbos(filename):
    for event,market,instrument,seconds in process_day(parse_file(filename)):
        assert event.ts_event.unique().shape[0]==1, event
        assert event.shape[0]==1 or (len(event.action.unique())==1 and len(event.side.unique())==1) or event.action.iloc[0]=='T', event
        #assert event[event.action=='M'].shape[0] <= 1, event
        yield (event,market,instrument,seconds)

def interleave(generators):
    queue = []
    def get_next(g):
        try:
            result = next(g)
            if result is not None:
                event, market, instrument, seconds = result
                heapq.heappush(queue, (seconds, instrument, hash(str(event)), result, g))
        except StopIteration:
           pass 
    for g in generators:
        get_next(g)
    while queue:
        seconds,instrument,_,(event,market,instrument,seconds),g = heapq.heappop(queue)
        yield event, market, instrument, seconds
        get_next(g)

def write_csv(generator, output_file):
    bid,ask=None,None
    f.write('seconds|instrument|instrument_id|ts_event|action|size|side|order_id|sequence|price|cancelled_orders|modified_order|modified_order_newsize|price|bq|bp|aq|ap\n')
    for result in generator:
        event, market, instrument, seconds = result
        try:
            bid,ask = market.aggregated_bbo(event.instrument_id.iloc[0])
            #data = [instrument, mbo.ts_event, mbo.ts_recv, seconds, mbo.order_id, mbo.action, mbo.side, mbo.size, mbo.price, bid.size if bid else np.nan, bid.price/dbn.FIXED_PRICE_SCALE if bid else np.nan, ask.size if ask else np.nan, ask.price/dbn.FIXED_PRICE_SCALE if ask else np.nan]
            modified_order = event[event.action=='M']
            modified_order_ids = list(map(int,modified_order.order_id.values))
            modified_order_newsize = list(map(int,modified_order['size'].values))
            market_details = [event.price.iloc[0]/dbn.FIXED_PRICE_SCALE, bid.size if bid else np.nan, bid.price/dbn.FIXED_PRICE_SCALE if bid else np.nan, ask.size if ask else np.nan, ask.price/dbn.FIXED_PRICE_SCALE if ask else np.nan]
            assert event.shape[1]==8, event
            event_details = event.iloc[0].copy()
            if (event['action']=='C').all() and len(event['side'].unique())==1:
                event_details.at['size'] = event['size'].sum()
            elif event.iloc[0]['action']!='T':
                assert event.shape[0] == 1, event
            print(seconds,instrument,*event.iloc[0].values, list(map(int,event[event.action=='C'].order_id.unique())),modified_order_ids,modified_order_newsize,*market_details,sep='|',file=f)
        except Exception as e:
            raise e#print('Exception: ', repr(e), event, bid, ask)

if __name__ == "__main__":
    folders = ["../data/databento/mes/ftp.databento.com/E8XGYL35/GLBX-20241008-PP9KBLT3CX/", "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/"]
    filenames = sorted(set([file.split('/')[-1] for folder in folders for file in glob.glob(f'{folder}/*.zst')]))
    print(filenames)
    for filename in tqdm.tqdm(filenames):
        print('Start',filename)
        mbos = interleave([process_mbos(folder+filename) for folder in folders])
        #output_file = '../output/databento_collated_fullday/'+filename.split('.')[0]+'.csv'
        output_file = '../output/databento_collated/'+filename.split('.')[0]+'.csv'
        datastores = {db.DBNStore.from_file(folder+filename) for folder in folders}
        bounds = {(data.metadata.start,data.metadata.end) for data in datastores}
        assert len(bounds) == 1, (filename,bounds)
        bounds = list(bounds)[0]
        with open(output_file,'w') as f:
            f.write(f'{bounds[0]},{bounds[1]}\n')
            write_csv(tqdm.tqdm(mbos), f)
        print('Done',filename)
