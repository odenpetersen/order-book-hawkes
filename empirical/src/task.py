#!/usr/bin/env python3
import os
import glob
import fit_model
import sys
import warnings

if __name__ == '__main__':
    folder_prefix = f'../output/params/{sys.argv[1]}/'

    if not os.path.exists(folder_prefix):
        os.makedirs(folder_prefix)

    train_files = sorted(glob.glob('../output/databento_collated_depth/*'))[:1]
    print(f'{train_files=}')
    warnings.filterwarnings('ignore')
    signs,params = fit_model.fit_model(train_files, 8, 2, 36000,37200)

    for i,data in enumerate([signs]+list(params)):
        np.savetxt(folder_prefix+f'{i}.txt',data)
