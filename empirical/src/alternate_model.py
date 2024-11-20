import numpy as np
import scipy as sp
import jax.flatten_util

def initial_params(num_components, num_marks, num_event_types):
    return np.random.exponential(size=num_event_types)*30, np.random.exponential(size=(num_components,num_marks,num_marks,num_event_types,num_event_types)),\
            np.random.exponential(size=num_components), np.random.exponential(size=num_components)

#nu :: event 
#m :: component x mark_dim x mark_dim x event x event
#beta :: component
#kappa :: component
def loglikelihood(start_time, end_time, times, events, marks, nu, m, beta, kappa):
    print('call')
    num_components = kappa.shape[0]
    num_marks = m.shape[1]
    num_event_types = nu.shape[0]
    lambdas = np.zeros((num_components, num_marks, num_marks, num_event_types, num_event_types))
    prev_time = start_time
    prev_mark = None
    ll = 0
    for time,event,mark in zip(times,events,marks):
        ll -= (time-prev_time)*nu.sum()
        ll -= ((lambdas.sum(axis=(1,2,3))**kappa[:,np.newaxis]).sum(axis=1) * (1 - np.exp(-beta*kappa*(time-prev_time)))/(beta*kappa)).sum(axis=0)

        lambdas *= np.exp(-beta*(time-prev_time))[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]
        component_lambdas = lambdas[:,:,:,:,event].sum(axis=(1,2,3))
        ll += np.log(nu[event] + (np.sign(component_lambdas)*np.abs(component_lambdas)**kappa).sum(axis=0))

        if prev_mark is not None:
            lambdas *= (mark/prev_mark)[np.newaxis,np.newaxis,:,np.newaxis,np.newaxis]
        lambdas += m * (mark[:,np.newaxis] * mark[np.newaxis,:])[np.newaxis,:,:,np.newaxis,np.newaxis]

    return ll

def nelder_mead(start_time,end_time,times,events,marks, num_components):
    num_event_types = max(events)+1
    num_marks = marks.shape[1]
    params = initial_params(num_components,num_marks,num_event_types)

    flat, unflatten = jax.flatten_util.ravel_pytree(params)
    results = sp.optimize.minimize(lambda x : -loglikelihood(start_time, end_time, times,events,marks,*unflatten(x)), flat)

    return results
