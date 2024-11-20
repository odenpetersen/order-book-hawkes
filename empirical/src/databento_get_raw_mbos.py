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
import itertools

def parse_file(file = "../data/databento/es/ftp.databento.com/E8XGYL35/GLBX-20241008-8PTR93CRA9/glbx-mdp3-20240904.mbo.dbn.zst"):
    data = db.DBNStore.from_file(file)

    instrument_map = db.common.symbology.InstrumentMap()
    instrument_map.insert_metadata(data.metadata)
    instrument_codes = dict()

    start = data.metadata.start

    cutoff = 1726916400
    for mbo in data:
        if mbo.instrument_id in instrument_codes:
            instrument = instrument_codes[mbo.instrument_id]
        else:
            ticker = instrument_map.resolve(mbo.instrument_id, datetime.datetime.fromtimestamp(mbo.ts_event*1e-9).date())
            if not ticker:
                ticker = instrument_map.resolve(mbo.instrument_id, datetime.datetime.fromtimestamp(start*1e-9).date())
            if ticker:
                instrument_codes[mbo.instrument_id] = ticker
                instrument = ticker
            else:
                instrument = None

        is_sept_expiry = instrument in ('MESU4','ESU4')
        is_dec_expiry = instrument in ('MESZ4','ESZ4')

        if mbo.action=='R' or (mbo.ts_event * 1e-9 < cutoff and is_sept_expiry) or (mbo.ts_event * 1e-9 > cutoff and is_dec_expiry):
            #if data.metadata.start <= mbo.ts_event <= data.metadata.end:
            yield ((mbo.ts_event-start)*1e-9,instrument[:-2] if instrument is not None else None,mbo)

class Book:
    bid = dict() #:: price -> order_id -> size,time
    ask = dict() #::
    order_lookup = dict() #:: order_id -> (side, price)

    def clear(self):
        self.bid.clear()
        self.ask.clear()
        self.order_lookup.clear()

    def trade(self, side, size, order_id):
        pass
        #match against orders

    def cancel(self, order_id):
        if order_id in self.order_lookup:
            side,price = self.order_lookup[order_id]
            del self.order_lookup[order_id]
            del side[price][order_id]
            if len(side[price])==0:
                del side[price]
            return True
        else:
            return False

    def modify(self, order_id, newprice, newsize, seconds):
        if order_id in self.order_lookup:
            side,price = self.order_lookup[order_id]
            if price==newprice:
                print(price,newsize,order_id)
                side[price][order_id] = (newsize, side[price][order_id][1])
            else:
                del side[price][order_id]
                if newprice not in side:
                    side[newprice] = dict()
                side[newprice][order_id] = (newsize, seconds)
            return True
        else:
            return False

    def add(self, order_id, side, price, size, time):
        side = self.bid if side=='B' else self.ask if side=='A' else None
        if price not in side:
            side[price] = dict()
        side[price][order_id] = (size, time)
        self.order_lookup[order_id] = (side, price)

    def __repr__(self):
        return str(dict(B=self.bid,A=self.ask))

if __name__ == '__main__':
    books = dict(ES=Book(),MES=Book())
    for seconds, instrument, mbo in parse_file():
        price = mbo.price/dbn.FIXED_PRICE_SCALE
        if mbo.action=='R':
            pass
        elif mbo.action=='T':
            books[instrument].trade(mbo.side,mbo.size,mbo.order_id)
        elif mbo.action=='F':
            pass
        elif mbo.action=='A':
            books[instrument].add(mbo.order_id, mbo.side, price, mbo.size, seconds)
        elif mbo.action=='C':
            books[instrument].cancel(mbo.order_id)
        elif mbo.action=='M':
            books[instrument].modify(mbo.order_id, price, mbo.size, seconds)
        else:
            raise Exception()
        if mbo.order_id == 6413756797497:
            print(seconds,instrument,mbo.instrument_id,mbo.action,price,mbo.side,mbo.size,mbo.order_id)
