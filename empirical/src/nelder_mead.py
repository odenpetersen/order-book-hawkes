import scipy as sp
import numpy as np
import jax.flatten_util
import fit_model
import glob

train_files = sorted(glob.glob('../output/databento_collated_depth/*'))
times,events,marks = fit_model.get_data(train_files[0])

signs = np.array([1,1,1,1,1,1])

num_event_types = max(events)+1
num_mark_variables = marks.shape[1]
params = fit_model.initial_params(num_event_types,len(signs),num_mark_variables)

flat, unflatten = jax.flatten_util.ravel_pytree(params)

start_time = 36000
results = sp.optimize.minimize(lambda x : -fit_model.loglikelihood(times[:10000],events[:10000],marks[:10000],start_time,times[10000],signs,*unflatten(x)), flat)
