#!/usr/bin/env python3
import matplotlib.pyplot as plt
import fit_model
import tqdm
import itertools
import pandas as pd
import numpy as np
signs = np.array([1])*1.0
params = fit_model.initial_params(12,len(signs),4)

import glob
param_files = glob.glob('../output/final_model_8_times_3_times_6_imbalances_and_spread_features/*.npy')
params = sorted([file for file in param_files if '_grad' not in file and '_hess' not in file])
params = list(map(np.load,params))
print([x.shape for x in params])
print([abs(p).max() for p in params])

nu,alpha,beta,a,b,kappa = params

#train_files = sorted(glob.glob('../output/databento_collated_depth/*'))
train_files = ['glbx-mdp3-20240708.csv','glbx-mdp3-20240710.csv','glbx-mdp3-20240712.csv','glbx-mdp3-20240716.csv','glbx-mdp3-20240718.csv']
train_files = ['../output/databento_collated_depth/'+f for f in train_files]
#test_files = ['glbx-mdp3-20240709.csv', 'glbx-mdp3-20240711.csv', 'glbx-mdp3-20240715.csv', 'glbx-mdp3-20240717.csv', 'glbx-mdp3-20240719.csv']
start_time=36000
end_time=37200

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

for train_file in train_files[:1]:
    es,mes = [pd.Series(eval(x)) for x in itertools.islice(open(train_file,'r'),0,2)]
    books = dict(ES=es,MES=mes)
    df = pd.read_csv(train_file,skiprows=2,sep='|',usecols=['seconds','side','instrument','action','change'])
    df.change = df.change.apply(eval)

    lambdas = np.zeros((len(nu),len(nu),len(signs)))
    mark = np.zeros(a.shape[-1])
    prev_time = start_time
    state_captured = False
    data = []
    for time,instrument,side,action,change in tqdm.tqdm(zip(df.seconds,df.instrument,df.side,df.action,df.change)):
        event = 2*(action=='T') + (action=='A')
        event = 2*event + (instrument=='ES')
        event = 2*event + (df.side=='A')

        prev_mark = np.zeros(4)
        if time>36600:
            if not state_captured:
                lambdas *= np.exp(-beta*(36600-time))
                prev_time = 36600

                mark_initial = mark.copy()
                books_initial = dict(ES=books['ES'].copy(), MES=books['MES'].copy())
                lambdas_initial = lambdas.copy()
                state_captured = True
            bid = books[instrument][books[instrument]>0].index.max()
            ask = books[instrument][books[instrument]<0].index.min()
            data.append((time,instrument,bid,ask))
        books[instrument] = books[instrument].add(pd.Series(change),fill_value=0)

        lambdas *= np.exp(-beta*(time-prev_time))
        
        mark = []
        for instrument in ('ES','MES'):
            bid = books[instrument][books[instrument]>0].index.max()
            ask = books[instrument][books[instrument]<0].index.min()
            spread = ask-bid
            imbalance = books[instrument][bid] / (books[instrument][bid]-books[instrument][ask])
            mark.append(spread)
            mark.append(imbalance)
        mark = np.nan_to_num(mark)

        lambdas *= np.exp((b*(mark-prev_mark)).sum(axis=3))
        lambdas[event] += alpha[event] * np.exp(((b[event]+a[event])*mark).sum(axis=3))

    data = pd.DataFrame(data,columns=['time','instrument','bid','ask'])
    es = data[data.instrument=='ES']
    mes = data[data.instrument=='MES']
    plt.scatter(es.time,(es.bid+es.ask)/2,label='Actual Prices')

    for _ in tqdm.trange(10):
        data = []
        prev_time = 36600
        prev_mark = mark_initial.copy()
        books = dict(ES=books_initial['ES'].copy(), MES=books_initial['MES'].copy())
        lambdas = lambdas_initial.copy()
        while True:
            intensity_upper_bound = nu + ((lambdas.clip(0).sum(axis=0)**kappa)*(signs.clip(0))).sum(axis=1)

            while True:
                prev_intensity = nu+(signs*lambdas.sum(axis=0)**kappa).sum(axis=1)
                prev_time = time
                time += np.random.exponential() / intensity_upper_bound.sum(axis=0)
                lambdas *= np.exp(-beta*(time-prev_time))
                intensity = nu+(signs*lambdas.sum(axis=0)**kappa).sum(axis=1)
                if np.random.uniform() < intensity.sum(axis=0) / prev_intensity.sum(axis=0):
                    break

            print(time)
            if time>start_time + 36660:
                break

            event = np.random.choice([*range(len(nu))],p=intensity/intensity.sum())

            side = ('B','A')[event // 2]
            instrument = ('MES','ES')[event // 4]
            event_type = ('C','A','T')[event%3]

            bid = books[instrument][books[instrument]>0].index.max()
            ask = books[instrument][books[instrument]<0].index.min()
            if event_type=='C':
                if side=='B':
                    price = bid-0.25*np.random.geometric(1/(31.6 if instrument=='ES' else 77))
                if side=='A':
                    price = ask+0.25*np.random.geometric(1/(31.6 if instrument=='ES' else 77))
                if instrument=='ES':
                    mu,sigma=-2.658,1.096
                else:
                    mu,sigma=-2.182,1.116
                level_qty = books[instrument][price] if price in books[instrument] else 0
                change = {price : -int(level_qty * sigmoid(np.random.normal()*sigma+mu))}
            if event_type == 'A':
                if instrument=='ES':
                    pricediff = 0.25*np.random.geometric(1/28.1)
                else:
                    pricediff = 0.25*np.random.geometric(1/73.6)
                if side=='A':
                    price = bid+pricediff
                else:
                    price = ask-pricediff
                volume = np.random.geometric(1/(1.71 if instrument=='ES' else 2.61))
                change = {price : volume * (1 if side=='B' else -1)}
            if event_type == 'T':
                volume = np.random.geometric(1/(4.726 if instrument=='ES' else 6.125))
                change = dict()
                if side=='B':
                    p = ask
                    while volume>0:
                        q = min(abs(book[p]),volume)
                        change[p] = q
                        p += 0.25
                        volume -= q
                if side=='B':
                    p = bid
                    while volume>0:
                        q = min(abs(book[p]),volume)
                        change[p] = -q
                        p -= 0.25
                        volume -= q
            books[instrument] = books[instrument].add(pd.Series(change), fill_value=0)

            data.append((time,instrument,bid,ask))
                        
            prev_mark = mark
            mark = []
            for instrument in ('ES','MES'):
                bid = books[instrument][books[instrument]>0].index.max()
                ask = books[instrument][books[instrument]<0].index.min()
                spread = ask-bid
                imbalance = books[instrument][bid] / (books[instrument][bid]-books[instrument][ask])
                mark.append(spread)
                mark.append(imbalance)
            mark = np.nan_to_num(mark)

            lambdas *= np.exp((b*(mark-prev_mark)).sum(axis=3))
            lambdas[event] += alpha[event] * np.exp(((b[event]+a[event])*mark).sum(axis=2))

        data = pd.DataFrame(data,columns=['time','instrument','bid','ask'])
        es = data[data.instrument=='ES']
        mes = data[data.instrument=='MES']
        plt.scatter(es.time,(es.bid+es.ask)/2)
    plt.xlabel('Seconds')
    plt.title('Simulated Price Series July 8th 10:10am-10:20am')
    plt.savefig('plots/simulated_prices.png')
