#!/usr/bin/env python3
import fit_model
import numpy as np
signs = np.array([1])*1.0
params = fit_model.initial_params(12,len(signs),4)

import glob
param_files = glob.glob('../output/final_model_8_times_3_times_6_imbalances_and_spread_features/*.npy')
params = sorted([file for file in param_files if '_grad' not in file and '_hess' not in file])
params = list(map(np.load,params))
print([x.shape for x in params])
print([abs(p).max() for p in params])


#train_files = sorted(glob.glob('../output/databento_collated_depth/*'))
train_files = ['glbx-mdp3-20240708.csv','glbx-mdp3-20240710.csv','glbx-mdp3-20240712.csv','glbx-mdp3-20240716.csv','glbx-mdp3-20240718.csv']
train_files = ['../output/databento_collated_depth/'+f for f in train_files]
#test_files = ['glbx-mdp3-20240709.csv', 'glbx-mdp3-20240711.csv', 'glbx-mdp3-20240715.csv', 'glbx-mdp3-20240717.csv', 'glbx-mdp3-20240719.csv']
start_time=36000
end_time=37200

grad_ema,hess_ema = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
while True:
    for train_file in np.random.choice(train_files,len(train_files),replace=False):
        times,events,marks = fit_model.get_data(train_file)
        grad,hess = fit_model.get_derivatives(times, events, marks, start_time, end_time, signs, *params)
        grad,hess = [np.nan_to_num(g) for g in grad], [np.nan_to_num(h) for h in hess]
        grad_ema,hess_ema = [0.1*g1+0.9*g2 for g1,g2 in zip(grad,grad_ema)], [0.1*h1+0.9*h2 for h1,h2 in zip(hess,hess_ema)]
        params=[p+0.1*np.where(h==0,g,g/abs(h)).clip(-1,1) for p,g,h in zip(params,grad_ema,hess_ema)]
        print([abs(g).max() for g in grad])
        print([abs(p).max() for p in params])
        for i,(p,g,h) in enumerate(zip(params,grad,hess)):
            np.save(f'../output/final_model_8_times_3_times_6_imbalances_and_spread_features/{i}.npy',p)
            np.save(f'../output/final_model_8_times_3_times_6_imbalances_and_spread_features/{i}_grad.npy',g)
            np.save(f'../output/final_model_8_times_3_times_6_imbalances_and_spread_features/{i}_hess.npy',h)
