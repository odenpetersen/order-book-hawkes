#!/usr/bin/env python3
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def read_file(file):
    with open(file,'r') as f:
        header = None
        for line in itertools.islice(f,1,None):
            line = line.split('\n')[0].split('|')
            if header is None:
                header = line
            else:
                yield dict(zip(header,line))

start_time = pd.to_datetime('2024-07-07 20:00:00')
def price_graph(file,where=None,resolution=60):
    prev_bucket = None
    times = []
    bids = []
    asks = []
    for event in read_file(file):
        if where is None or where(event):
            time = float(event['seconds'])
            bucket = int(time/resolution) 
            if bucket != prev_bucket:
                bp,ap=map(float,(event['bp'],event['ap']))
                times.append(start_time+pd.to_timedelta(time,unit='s'))
                bids.append(bp)
                asks.append(ap)
            prev_bucket = bucket
        
    plt.scatter(times,bids)
    plt.scatter(times,asks)

def activity_graph(file,where=None,label=None,resolution=60,cumulative=False):
    prev_bucket = None
    times = []
    counts = []
    for event in read_file(file):
        try:
            if where is not None and where(event):
                time = float(event['seconds'])
                bucket = None if resolution is None else int(time/resolution) 
                if resolution is None or bucket != prev_bucket:
                    times.append(start_time+pd.to_timedelta(time,unit='s'))
                    counts.append(0)
                counts[-1] += 1
                prev_bucket = bucket
        except Exception:
            print(event)
        
    if cumulative:
        plt.scatter(times,np.cumsum(counts),label=label)
    else:
        plt.scatter(times,counts,label=label)

def mean_graph(file,function,where=None,label=None,resolution=60):
    prev_bucket = None
    times = []
    means = []

    total = 0
    count = 0

    for event in read_file(file):
        try:
            if where is not None and where(event):
                time = float(event['seconds'])
                bucket = int(time/resolution) 
                if prev_bucket is not None and bucket != prev_bucket:
                    times.append(start_time+pd.to_timedelta(time,unit='s'))
                    means.append(total/count)
                    total = 0
                    count = 0
                total += function(event)
                count += 1
                prev_bucket = bucket
        except Exception:
            print(event)
        
    plt.scatter(times,means,label=label)

def arrivals_vs_intensity(file,label=None,alpha=0.05,linewidth=3,cumulative=False):
    times = []
    residuals = []
    intensities = []
    with open(file,'r') as f:
        for line in f:
            line = line.split(',')
            if len(line) == 1:
                residual = float(line[0])
            else:
                time, residual, intensity = map(float,line)
                times.append(start_time+pd.to_timedelta(time,unit='s'))
                residuals.append(residual)
                intensities.append(intensity)
    if cumulative:
        plt.plot(times,np.cumsum(residuals),label=label,linewidth=linewidth)
        plt.scatter(times,np.arange(0,len(times)),alpha=alpha,marker='.')
    else:
        plt.plot(times,intensities,label=label,linewidth=linewidth)
        plt.vlines(times,ymin=0,ymax=max(intensities),alpha=alpha)

def residual_cdf_serial_plot(file,label=None,alpha=0.5):
    transformed_residuals = []
    with open(file,'r') as f:
        for line in f:
            line = line.split(',')
            if len(line) == 1:
                residual = float(line[0])
            else:
                time, residual, intensity = map(float,line)
            transformed_residuals.append(1 - np.exp(-residual))
    plt.scatter(transformed_residuals[1:],transformed_residuals[:-1],label=label,alpha=alpha)
    return np.corrcoef(transformed_residuals[1:],transformed_residuals[:-1])[0,1]

train_days = "0708","0709","0710","0711","0712","0715","0716","0717","0718","0719"
test_days = "0722","0723","0724","0725","0726","0729","0730","0731","0801","0802"

def intensity_plots(cumulative=False,folder='residuals_train2wkstest2wks_poisson'):
    train_files, test_files = map(lambda x : [f'../../output/{folder}/{day}.csv' for day in x],(train_days,test_days))
    for file,day in zip(train_files,train_days):
        arrivals_vs_intensity(file,label=day,cumulative=cumulative)
    plt.suptitle('Model Trained on 8th-12th July 2024')
    plt.title(f'Arrivals vs Intensity (In Sample)')
    plt.legend()
    plt.savefig(f'intensities_train{"_cumulative" if cumulative else ""}_{folder}.png')
    plt.clf()

    for file,day in zip(test_files,test_days):
        arrivals_vs_intensity(file,label=day,cumulative=cumulative)
    plt.suptitle('Model Trained on 8th-12th July 2024')
    plt.title(f'Arrivals vs Intensity (Out of Sample)')
    plt.legend()
    plt.savefig(f'intensities_test{"_cumulative" if cumulative else ""}_{folder}.png')
    plt.clf()

def residuals_plots(folder='residuals_train2wkstest2wks_poisson'):
    train_files, test_files = map(lambda x : [f'../../output/{folder}/{day}.csv' for day in x],(train_days,test_days))
    corrs = []
    for file,day in zip(train_files,train_days):
        corrs.append(residual_cdf_serial_plot(file,label=day))
    plt.suptitle('Model Trained on 8th-12th July 2024')
    plt.title(f'Consecutive Transformed Residuals (In-Sample, corr={np.mean(corrs):.2f})')
    plt.legend()
    plt.savefig(f'consecutive_train_{folder}.png')
    plt.clf()

    corrs = []
    for file,day in zip(test_files,test_days):
        corrs.append(residual_cdf_serial_plot(file,label=day))
    plt.suptitle('Model Trained on 8th-12th July 2024')
    plt.title(f'Consecutive Transformed Residuals (Out of Sample, corr={np.mean(corrs):.2f})')
    plt.legend()
    plt.savefig(f'consecutive_test_{folder}.png')
    plt.clf()

def insertion_distribution(filename,price=True):
    es,mes = itertools.islice(open(filename,'r'),0,2)
    es,mes = map(pd.Series,map(eval,(es,mes)))

    df = pd.read_csv(filename,sep='|',skiprows=2,usecols=['instrument','action','side','change'])
    df.change = df.change.apply(eval)

    df['bidprice_before'] = np.nan
    df['askprice_before'] = np.nan
    for i,(instrument,change) in enumerate(zip(df.instrument,df.change)):
        if instrument=='ES':
            bid = es[es>0].index.max()
            ask = es[es<0].index.min()
            es = es.add(pd.Series(change),fill_value=0)
        if instrument=='MES':
            bid = mes[mes>0].index.max()
            ask = mes[mes<0].index.min()
            mes = mes.add(pd.Series(change),fill_value=0)
        df.loc[i,'bidprice_before'] = bid
        df.loc[i,'askprice_before'] = ask

    if price:
        for instrument in ('ES','MES'):
            insertions = df[(df.action=='A') & (df.change.apply(len)==1) & (df.instrument==instrument)]
            insertion_price_minus_opposite_side_of_book = abs(insertions.change.apply(lambda x : list(x)[0]) - insertions.bidprice_before.where(insertions.side=='A',insertions.askprice_before))
            log_pricediff = np.log(insertion_price_minus_opposite_side_of_book)
            plt.hist(pd.Series(log_pricediff).where(np.isfinite).dropna(),label=instrument,alpha=0.8)
    else:
        for instrument in ('ES','MES'):
            insertions = df[(df.action=='A') & (df.change.apply(len)==1) & (df.instrument==instrument)]
            plt.hist(np.log(abs(insertions.change.apply(lambda x : list(x.values())[0]))), label=instrument, alpha=0.8)

if __name__ == '__main__':
    insertion_distribution('../../output/databento_collated_depth/glbx-mdp3-20240708.csv',price=True)
    plt.title('Log of Difference between Insertion Price and Other Side of Book (8th July)')
    plt.legend()
    plt.savefig('insertion_distribution_price.png')
    plt.clf()

    insertion_distribution('../../output/databento_collated_depth/glbx-mdp3-20240708.csv',price=False)
    plt.title('Log of Insertion Size (8th July)')
    plt.legend()
    plt.savefig('insertion_distribution_size.png')
    plt.clf()

    print('residuals')
    for folder in ('residuals','residuals_more_imprecise','residuals_train2wkstest2wks_poisson'): #'residuals_train2wkstest2wks_hawkes', 'residuals_train2wkstest2wks_hawkes_imprecise'
        intensity_plots(cumulative=False,folder=folder)
        intensity_plots(cumulative=True,folder=folder)

        residuals_plots(folder=folder)

    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES' and event['action']=='T',label='ES Trades',cumulative=True,resolution=None)
    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='MES' and event['action']=='T',label='MES Trades',cumulative=True,resolution=None)
    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES' and event['action']=='A',label='ES Insertions',cumulative=True,resolution=None)
    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='MES' and event['action']=='A',label='MES Insertions',cumulative=True,resolution=None)
    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES' and event['action']=='C',label='ES Cancellations',cumulative=True,resolution=None)
    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='MES' and event['action']=='C',label='MES Cancellations',cumulative=True,resolution=None)
    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES' and event['action']=='M',label='ES Modifications',cumulative=True,resolution=None)
    activity_graph('../../output/databento_collated/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='MES' and event['action']=='M',label='MES Modifications',cumulative=True,resolution=None)
    plt.title('Cumulative Total Order Book Events, 8th July 2024')
    plt.legend()
    plt.savefig('counting_process_event_type.png')
    plt.clf()

    activity_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES' and event['action']=='T' and 13.5*60*60<=float(event['seconds'])<=20*60*60,label='ES')
    activity_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='MES' and event['action']=='T' and 13.5*60*60<=float(event['seconds'])<=20*60*60,label='MES')
    plt.title('Trades per Minute (9:30-16:00), 8th July 2024')
    plt.legend()
    plt.savefig('activity_graph_zoomed.png')
    plt.clf()

    price_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES')
    plt.title('ES Intraday Price Chart, 8th July 2024')
    plt.savefig('price_graph.png')
    plt.clf()

    activity_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES' and event['action']=='T',label='ES')
    activity_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='MES' and event['action']=='T',label='MES')
    plt.title('Trades per Minute, 8th July 2024')
    plt.legend()
    plt.savefig('activity_graph.png')
    plt.clf()

    activity_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='ES' and event['action']=='T',label='ES',cumulative=True)
    activity_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv',lambda event : event['instrument']=='MES' and event['action']=='T',label='MES',cumulative=True)
    plt.title('Cumulative Total Trades, 8th July 2024')
    plt.legend()
    plt.savefig('counting_process.png')
    plt.clf()

    mean_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv', lambda event : float(event['size']), lambda event : event['instrument']=='ES' and event['action']=='T' and 13.5*60*60<=float(event['seconds'])<=20*60*60,label='ES')
    mean_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv', lambda event : float(event['size']), lambda event : event['instrument']=='MES' and event['action']=='T' and 13.5*60*60<=float(event['seconds'])<=20*60*60,label='MES')
    plt.title('Average Trade Size by Minute, 8th July 2024')
    plt.legend()
    plt.savefig('trade_size.png')
    plt.clf()

    sign = lambda event : 1 if event['side']=='B' else -1 if event['side']=='A' else 0
    mean_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv', sign, lambda event : event['instrument']=='ES' and event['action']=='T' and 13.5*60*60<=float(event['seconds'])<=20*60*60,label='ES')
    mean_graph('../../output/databento_collated_fullday/glbx-mdp3-20240708.csv', sign, lambda event : event['instrument']=='MES' and event['action']=='T' and 13.5*60*60<=float(event['seconds'])<=20*60*60,label='MES')
    plt.title('Average Trade Sign (+1/-1) by Minute, 8th July 2024')
    plt.legend()
    plt.savefig('trade_sign.png')
    plt.clf()
