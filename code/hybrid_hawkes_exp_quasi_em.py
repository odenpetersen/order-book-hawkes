import numpy as np
import scipy as sp
import scipy.optimize
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm

def unpack_params(params, num_event_types):
    nus = params[:num_event_types]
    alphas = params[num_event_types:num_event_types*(num_event_types+1)].reshape(num_event_types,num_event_types)
    betas = params[num_event_types*(num_event_types+1):].reshape(num_event_types,num_event_types)
    return nus,alphas,betas

def em_step(times, events, states, time_start, time_end, nus,alphas,betas, num_event_types):
    times -= time_start
    time_end -= time_start
    time_start = 0

    intensity = np.stack([nus,np.zeros_like(alphas)])
    alphas_augmented = np.stack([np.zeros_like(nus),alphas])
    betas_augmented = np.stack([np.zeros_like(nus),betas])

    logalpha_coefs = np.zeros_like(alphas_augmented)
    expsum_times = np.zeros_like(alphas_augmented)
    expweights_times = np.zeros_like(alphas_augmented)
    beta_coefs = np.zeros_like(alphas_augmented)
    stability_coefs = np.zeros_like(alphas_augmented)

    t_prev = time_start
    for i,(t,e) in enumerate(zip(times,events)):
        decay = np.exp(-betas_augmented * (t-t_prev))
        intensity *= decay

        logalpha_coefs += intensity/intensity.sum()
        beta_coefs =
        stability_coefs[e+1,:] += 1#Compensator. Approximates time_end as infinity
        expected_background
        
        #Impact
        intensity[:,e] += alphas_augmented[e+1,:]


        t_prev = t


    nus =
    alphas =
    betas =

    return nus,alphas,betas

def estimate_hawkes_parameters(times, events, states, time_start, time_end, num_steps = 50):
    num_event_types = len(np.unique(events))
    nus =
    alphas =
    betas =
    for _ in tqdm.trange(num_steps):
        nus,alphas,betas = em_step(times,events,states,time_start,time_end,nus,alphas,betas)
    return nus,alphas,betas

"""
- Correctness
- Stochastic EM algorithm


- Look across days
- Hidden events
- See if i can get same time period and tickers as the original paper
- Thesis draft. Results, overview & citations

- Hidden events (MA component)
- 



"""
