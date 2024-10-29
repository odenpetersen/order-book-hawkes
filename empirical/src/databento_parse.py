#!/usr/bin/env python3
import glob
import heapq
import datetime
import tqdm
import numpy as np
import databento as db
import databento_dbn as dbn
import databento_classes

def parse_folder(data_path = "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/"):
    for file in sorted(glob.glob(data_path+'*.zst')):
        for result in parse_file(file):
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

def interleave(generators):
    mbos_queue = []
    for g in generators:
        try:
            result = next(g)
            if result:
                mbo, market, instrument, seconds = result
                heapq.heappush(mbos_queue, (mbo.ts_recv, mbo.ts_event, str(mbo), mbo, market, instrument, seconds, g))
        except StopIteration:
            continue
    while mbos_queue:
        ts, ts_event, _, mbo, market, instrument, seconds, g = heapq.heappop(mbos_queue)
        yield mbo, market, instrument, seconds
        try:
            result = next(g)
            if result:
                mbo, market, instrument, seconds = result
                heapq.heappush(mbos_queue, (mbo.ts_recv, mbo.ts_event, str(mbo), mbo, market, instrument, seconds, g))
        except StopIteration:
            continue

def write_csv(generator, output_file):
    f.write('instrument,ts_event,ts_recv,seconds_since_start,order_id,action,side,size,price,bq,bp,aq,ap')
    for result in generator:
        mbo, market, instrument, seconds = result
        try:
            bid,ask = market.aggregated_bbo(mbo.instrument_id)
            data = [instrument, mbo.ts_event, mbo.ts_recv, seconds, mbo.order_id, mbo.action, mbo.side, mbo.size, mbo.price, bid.size if bid else np.nan, bid.price/dbn.FIXED_PRICE_SCALE if bid else np.nan, ask.size if ask else np.nan, ask.price/dbn.FIXED_PRICE_SCALE if ask else np.nan]
            f.write(','.join(map(str,data))+'\n')
        except Exception as e:
            print('Exception: ', repr(e), mbo, bid, ask)

if __name__ == "__main__":
    folders = ["../data/databento/mes/ftp.databento.com/E8XGYL35/GLBX-20241008-PP9KBLT3CX/", "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/"]
    filenames = sorted(set([file.split('/')[-1] for folder in folders for file in glob.glob(f'{folder}/*.zst')]))
    print(filenames)
    for filename in tqdm.tqdm(filenames):
        print('Start',filename)
        mbos = interleave([parse_file(folder+filename) for folder in folders])
        output_file = '../output/databento/'+filename.split('.')[0]+'.csv'
        datastores = {db.DBNStore.from_file(folder+filename) for folder in folders}
        bounds = {(data.metadata.start,data.metadata.end) for data in datastores}
        assert len(bounds) == 1, (filename,bounds)
        bounds = list(bounds)[0]
        with open(output_file,'w') as f:
            f.write(f'{bounds[0]},{bounds[1]}\n')
            write_csv(tqdm.tqdm(mbos), f)
        print('Done',filename)
