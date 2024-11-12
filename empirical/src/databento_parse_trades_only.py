#!/usr/bin/env python3
import databento as db
import glob
import tqdm
from databento_parse import interleave
import databento_classes
import databento_dbn as dbn
import datetime

def parse_file(filename):
    data = db.DBNStore.from_file(filename)

    market = databento_classes.Market()

    instrument_map = db.common.symbology.InstrumentMap()
    instrument_map.insert_metadata(data.metadata)

    cutoff = 1726916400
    for mbo in data:
        market.apply(mbo)

        timestamp = mbo.ts_event * 1e-9

        instrument = instrument_map.resolve(mbo.instrument_id, datetime.datetime.fromtimestamp(mbo.ts_event*1e-9).date())

        is_sept_expiry = instrument in ('MESU4','ESU4')
        is_dec_expiry = instrument in ('MESZ4','ESZ4')

        if (mbo.ts_event * 1e-9 < cutoff and is_sept_expiry) or (mbo.ts_event * 1e-9 > cutoff and is_dec_expiry):
            if mbo.action == 'T':
                yield (mbo,market,instrument[:-2])

def write_csv(generator, output_file):
    with open(output_file,'w') as f:
        f.write('instrument,timestamp,action,side,size,price,ts_delta,bq,bp,aq,ap\n')
        for mbo,market,instrument in generator:
            try:
                bid,ask = market.aggregated_bbo(mbo.instrument_id)
                data = [instrument, mbo.ts_event*1e-9, mbo.order_id, mbo.action, mbo.side, mbo.size, mbo.price, bid.size if bid else np.nan, bid.price/dbn.FIXED_PRICE_SCALE if bid else np.nan, ask.size if ask else np.nan, ask.price/dbn.FIXED_PRICE_SCALE if ask else np.nan]
                f.write(','.join(map(str,data))+'\n')
            except Exception as e:
                print('Exception: ', repr(e), mbo, bid, ask)

if __name__ == '__main__':
    folders = ["../data/databento/mes/ftp.databento.com/E8XGYL35/GLBX-20241008-PP9KBLT3CX/", "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/"]
    filenames = set([file.split('/')[-1] for folder in folders for file in glob.glob(f'{folder}/*.zst')])
    print(filenames)
    for filename in tqdm.tqdm(filenames):
        print('Start',filename)
        mbos = interleave([parse_file(folder+filename) for folder in folders])
        write_csv(tqdm.tqdm(mbos), '../output/databento_trades/'+filename.split('.')[0]+'.csv')
        print('Done',filename)
