#!/usr/bin/env python3
import numpy as np
import pandas as pd
import fit_model

if __name__ == '__main__':
    df = pd.read_csv('../output/databento_collated/glbx-mdp3-20240724.csv',skiprows=1,sep='|')

    times = df.seconds.values
    events = (df.side=='A') + 2*(df.instrument=='ES')
    marks = df[['bq','aq']].values

    num_event_types = 4
    num_components = 10
    num_mark_variables = marks.shape[1]

    signs = np.ones(num_components)

    signs,params = fit_model.fit_model(times,events,marks,36000,37200)

    print(signs,params)
