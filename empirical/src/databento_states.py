#!/usr/bin/env python3
import databento_parse_new
import databento_dbn as dbn
import glob
import tqdm
import pandas as pd
import numpy as np

def depth_profile(instrument_id,market):
    book, = market.get_books_by_pub(instrument_id).values()

    bids = [(lambda t : (p/dbn.FIXED_PRICE_SCALE,t.size))(book.get_bid_level_by_px(p)) for p in book.bids]
    asks = [(lambda t : (p/dbn.FIXED_PRICE_SCALE,-t.size))(book.get_ask_level_by_px(p)) for p in book.offers]

    bids,asks = (pd.DataFrame(side,columns=['p','q']).set_index('p')['q'] for side in (bids,asks))

    bids,asks = (side[~side.index.duplicated(keep='first')] for side in (bids,asks))

    return bids.add(asks,fill_value=0)

folders = ["../data/databento/mes/ftp.databento.com/E8XGYL35/GLBX-20241008-PP9KBLT3CX/", "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/"]
filenames = sorted(set([file.split('/')[-1] for folder in folders for file in glob.glob(f'{folder}/*.zst')]))
print(filenames)
for filename in tqdm.tqdm(filenames):
    output_file = '../output/databento_collated_depth/'+filename.split('.')[0]+'.csv'
    print('Start',filename)
    mbos = databento_parse_new.interleave([databento_parse_new.process_mbos(folder+filename) for folder in folders])

    header_printed = False
    with open(output_file,'w') as f:
        prev_depth = dict(ES=None,MES=None)
        marks = dict(ES=None,MES=None)
        for event,market,instrument,seconds in mbos:
            instrument_id, = event['instrument_id'].unique()
            action = event.iloc[0]['action']
            depth = depth_profile(instrument_id,market)

            bid = depth[depth>0].index.max()
            ask = depth[depth<0].index.min()
            bid2 = depth.index.to_series().where(lambda x : x<bid).max()
            ask2 = depth.index.to_series().where(lambda x : x>ask).min()

            marks[instrument] = [event.iloc[0]['price']/dbn.FIXED_PRICE_SCALE-((bid+ask)/2), event.iloc[0]['size'], ask-bid, depth[bid], -depth[ask], depth[bid]/(depth[bid]-depth[ask]), ask2-bid2, depth[bid2], -depth[ask2], depth[[bid,bid2]].sum()/(depth[[bid,bid2]].sum()-depth[[ask,ask2]].sum()), abs(depth).sum()]
            marks[instrument] = [float(x) for x in marks[instrument]]

            if not any((x is None for x in prev_depth.values())) and not any((x is None for x in marks.values())):
                if not header_printed:
                    print(prev_depth['ES'].to_dict(),file=f)
                    print(prev_depth['MES'].to_dict(),file=f)
                    print('seconds|instrument|action|side|marks|change',file=f)
                    header_printed = True
                print(seconds, instrument, action, event.iloc[0]['side'], marks['ES']+marks['MES'], depth.sub(prev_depth[instrument],fill_value=0).replace(0,np.nan).dropna().to_dict(), sep='|', file=f)
            prev_depth[instrument] = depth
