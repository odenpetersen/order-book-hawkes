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

def stochastic_em(times, events, states, time_start, time_end, num_event_types, epochs=5,momentum=0.8, lr=1e-5, nus=None,alphas=None,betas=None):
    if nus is None:
        nus = np.ones(num_event_types)
    if alphas is None:
        alphas = np.ones((num_event_types,num_event_types))
    if betas is None:
        betas = np.ones((num_event_types,num_event_types))


    nus_grad = np.zeros_like(nus)
    alphas_grad = np.zeros_like(alphas)
    betas_grad = np.zeros_like(betas)

    for _ in range(epochs):
        print(nus)
        print(alphas)
        print(betas)
        intensities = np.zeros_like(alphas)
        prev_t = time_start

        for t,e,s in tqdm.tqdm(zip(times,events,states)):
            assert intensities.shape==alphas.shape==betas.shape
            assert intensities.shape[0]==intensities.shape[1]==nus.shape[0], (intensities.shape,nus.shape)

            nus_grad *= momentum
            alphas_grad *= momentum
            betas_grad *= momentum

            decay = np.exp(-betas*(t-prev_t))

            #Compensator
            nus_grad += (1-momentum) * nus*(t-prev_t)
            alphas_grad += (1-momentum) * (-1)*intensities*(1 - decay)/betas
            betas_grad += (1-momentum) * (-1)*intensities*(decay * (1 + 1/betas**2) - 1/betas**2)

            #Event logintensity
            total_intensity = nus[e] + (decay*intensities)[:,e].sum()
            exo_proba = nus[e]/total_intensity
            endo_probas = (decay*intensities)[:,e]/total_intensity

            nus_grad += (1-momentum) * exo_proba/nus[e]
            alphas_grad += (1-momentum) * endo_probas/alphas[:,e]
            betas_grad += (1-momentum) * endo_probas*(prev_t-t)

            intensities = decay*intensities + alphas[e,:]

            prev_t = t

            assert np.isnan(nus_grad).sum()==0, (nus_grad, nus,alphas,betas)
            assert np.isnan(alphas_grad).sum()==0, (alphas_grad, nus,alphas,betas)
            assert np.isnan(betas_grad).sum()==0, (betas_grad, nus,alphas,betas)

            nus += lr*nus_grad
            alphas += lr*alphas_grad
            betas += lr*betas_grad
            nus = np.clip(nus,np.finfo(float).eps,None)
            alphas = np.clip(alphas,np.finfo(float).eps,None)
            betas = np.clip(betas,np.finfo(float).eps,None)

    #Account for period after last observed event
    #todo

    return nus,alphas,betas

def branching_matrix(times, events, states, time_start, time_end, params, num_event_types):
    nus,alphas,betas = unpack_params(params, num_event_types)

    exo_probas = np.zeros(len(times))
    endo_probas = np.zeros(len(times))

    endo_intensity = np.zeros_like(alphas)
    t_prev = time_start
    for i,(t,e) in enumerate(zip(times,events)):
        #Decay
        endo_intensity *= np.exp(-betas * (t-t_prev))

        total_intensity = endo_intensity[:].sum() + nus[e]
        assert np.array(total_intensity).shape == ()
        exo_probas[i] = nus[e] / total_intensity
        endo_probas[i] = endo_intensity[:].sum() / total_intensity
        assert np.allclose(endo_probas[i].sum() + exo_probas[i], 1)

        #Impact
        endo_intensity += alphas[e]

        t_prev = t
    return exo_probas, endo_probas

def conditional_ll(times, events, states, time_start, time_end, exo_probas, endo_probas, params, num_event_types):
    nus,alphas,betas = unpack_params(params, num_event_types)

    event_ll = 0
    compensator = 0

    endo_intensity = jnp.zeros_like(alphas)
    t_prev = time_start
    for i,(t,e) in enumerate(zip(times,events)):
        compensator += (nus * (t-t_prev)).sum()
        compensator += (-(endo_intensity/betas) * (jnp.exp(-betas*t) - jnp.exp(-betas*t_prev))).sum()

        #Decay
        endo_intensity *= jnp.exp(-betas * (t-t_prev))

        event_ll += exo_probas[i] * jnp.log(nus[e])
        if endo_probas[i] > 0:
            event_ll += endo_probas[i] * jnp.log(endo_intensity[:,e].sum())

        #Impact
        endo_intensity += alphas[e]

        t_prev = t

        #print(compensator,event_ll,nus[e],endo_intensity[e],exo_probas[i],endo_probas[i],endo_intensity)
    
    return event_ll - compensator

def estimate_hawkes_parameters(times, events, states, time_start, time_end, num_steps = 50):
    num_event_types = len(np.unique(events))
    params = np.ones(num_event_types*(2*num_event_types+1))
    for _ in tqdm.trange(num_steps):
        #1. Expectation
        exo_probas, endo_probas = branching_matrix(times, events, states, time_start, time_end, params, num_event_types)

        #2. Minimisation
        minimand = lambda p : conditional_ll(times,events,states,time_start,time_end,exo_probas,endo_probas,p, num_event_types)
        grad = lambda p : jax.grad(conditional_ll, argnums=7)(times,events,states,time_start,time_end,exo_probas,endo_probas,p, num_event_types)
        print(minimand(params),unpack_params(params,num_event_types))
        params = sp.optimize.minimize(minimand, params, jac=grad, method=sgd, options=dict(maxiter=10)).x

    return unpack_params(params, num_event_types)


#https://gist.github.com/jcmgray/e0ab3458a252114beecb1f4b631e19ab
def sgd(
    fun,
    x0,
    jac,
    args=(),
    learning_rate=0.001,
    mass=0.9,
    startiter=0,
    maxiter=1000,
    callback=None,
    **kwargs
):
    """``scipy.optimize.minimize`` compatible implementation of stochastic
    gradient descent with momentum.
    Adapted from ``autograd/misc/optimizers.py``.
    """
    x = x0
    velocity = np.zeros_like(x)

    for i in range(startiter, startiter + maxiter):
        g = jac(x)

        if callback and callback(x):
            break

        velocity = mass * velocity - (1.0 - mass) * g
        x = x + learning_rate * velocity

    i += 1
    return sp.optimize.OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)


"""
- Correctness
- Stochastic EM algorithm


- Look across days
- Hidden events
- See if i can get same time period and tickers as the original paper
- Thesis draft. Results, overview & citations

- Hidden events (MA component)
"""
