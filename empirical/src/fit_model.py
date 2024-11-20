import numpy as np
import glob
import pandas as pd

#times: |T|
#events: |T|
#marks: |T| x num_marks

#signs : num_components
#nu    : |E|
#alpha : |E| x |E| x num_components
#beta  : num_components
#a     : |E| x |E| x num_components x num_marks
#b     : |E| x |E| x num_components x num_marks
#kappa : num_components
def get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa):
    #|T| x |E| x |E| x num_components
    lambda_4d = np.empty((len(times),len(nu),len(nu),len(signs)))
    lambda_4d[:] = np.nan
    lambda_4d_right = np.empty((len(times),len(nu),len(nu),len(signs)))
    lambda_4d_right[:] = np.nan

    #|T| x |E| x num_components
    dlambda_dbeta_3d = np.empty((len(times),len(nu),len(signs)))
    dlambda_dbeta_3d[:] = np.nan
    dlambda_dbeta_3d_right = np.empty((len(times),len(nu),len(signs)))
    dlambda_dbeta_3d_right[:] = np.nan
    d2lambda_dbeta2_3d = np.empty((len(times),len(nu),len(signs)))
    d2lambda_dbeta2_3d[:] = np.nan
    d2lambda_dbeta2_3d_right = np.empty((len(times),len(nu),len(signs)))
    d2lambda_dbeta2_3d_right[:] = np.nan

    #|T| x |E| x |E| x num_components x num_marks
    dlambda_da_5d = np.empty((len(times),len(nu),len(nu),len(signs),marks.shape[1]))
    dlambda_da_5d[:] = np.nan
    dlambda_da_5d_right = np.empty((len(times),len(nu),len(nu),len(signs),marks.shape[1]))
    dlambda_da_5d_right[:] = np.nan
    d2lambda_da2_5d = np.empty((len(times),len(nu),len(nu),len(signs),marks.shape[1]))
    d2lambda_da2_5d[:] = np.nan
    d2lambda_da2_5d_right = np.empty((len(times),len(nu),len(nu),len(signs),marks.shape[1]))
    d2lambda_da2_5d_right[:] = np.nan

    #|T|
    timediffs = np.diff(times,append=end_time)

    #num_components
    gamma = beta*kappa
    #|T| x num_components
    gamma_0 = (1-np.exp(-gamma*timediffs[:,np.newaxis]))/gamma
    gamma_1 = (1-np.exp(-gamma*timediffs[:,np.newaxis])*(gamma*timediffs[:,np.newaxis]+1))/(gamma**2)
    gamma_2 = (2-np.exp(-gamma*timediffs[:,np.newaxis])*((gamma*timediffs[:,np.newaxis] + 2)*gamma*timediffs[:,np.newaxis] + 2))/(gamma**3)

    #|T| x num_marks
    markdiffs = np.diff(marks,prepend=0,axis=0)

    lambdas = np.zeros((len(nu),len(nu),len(signs)))
    dlambda_dbeta = np.zeros((len(nu),len(signs)))
    d2lambda_dbeta2 = np.zeros((len(nu),len(signs)))
    dlambda_da = np.zeros((len(nu),len(nu),len(signs),marks.shape[1]))
    d2lambda_da2 = np.zeros((len(nu),len(nu),len(signs),marks.shape[1]))
    for i,(timediff,e,markdiff,mark) in enumerate(zip(timediffs,events,markdiffs,marks)):
        #Left limits
        lambda_4d[i] = lambdas
        dlambda_dbeta_3d[i] = dlambda_dbeta 
        d2lambda_dbeta2_3d[i] = d2lambda_dbeta2 
        dlambda_da_5d[i] = dlambda_da 
        d2lambda_da2_5d[i] = d2lambda_da2 

        #Update in reaction to event
        lambdas *= np.exp((b*markdiff).sum(axis=3))
        lambdas[e] += alpha[e] * np.exp(((b[e]+a[e])*mark).sum(axis=2))
        dlambda_da *= np.exp((b*markdiff).sum(axis=3))[:,:,:,np.newaxis]
        dlambda_da[e] += mark[np.newaxis,np.newaxis,:] * (alpha[e] * np.exp(((b[e]+a[e])*mark).sum(axis=2)))[:,:,np.newaxis]
        d2lambda_da2 *= np.exp((b*markdiff).sum(axis=3))[:,:,:,np.newaxis]
        d2lambda_da2[e] += mark**2 * (alpha[e] * np.exp(((b[e]+a[e])*mark).sum(axis=2)))[:,:,np.newaxis]

        #Right limits
        lambda_4d_right[i] = lambdas
        dlambda_dbeta_3d_right[i] = dlambda_dbeta 
        d2lambda_dbeta2_3d_right[i] = d2lambda_dbeta2 
        dlambda_da_5d_right[i] = dlambda_da 
        d2lambda_da2_5d_right[i] = d2lambda_da2 

        #Progress time
        d2lambda_dbeta2 = np.exp(-beta*timediff) * (d2lambda_dbeta2 - 2*timediff*dlambda_dbeta + timediff**2*lambdas.sum(axis=0))
        dlambda_dbeta = np.exp(-beta*timediff) * (dlambda_dbeta - timediff*lambdas.sum(axis=0)) 
        lambdas *= np.exp(-beta*timediff)
        dlambda_da *= np.exp(-beta*timediff)[np.newaxis,np.newaxis,:,np.newaxis]
        d2lambda_da2 *= np.exp(-beta*timediff)[np.newaxis,np.newaxis,:,np.newaxis]

    #|T| x |E| x num_components
    lambda_3d = lambda_4d.sum(axis=1)
    lambda_3d_right = lambda_4d_right.sum(axis=1)

    #|T| x |E| x num_components
    components = signs*lambda_3d**kappa

    #|T| x |E|
    intensities = components.sum(axis=2) + nu

    #|T| x |E|
    P0 = np.where(intensities==0,0,nu / intensities)
    P0 *= np.eye(len(nu))[events]

    #|T| x |E| x num_components
    P = np.where(intensities[:,:,np.newaxis]==0,0,components / intensities[:,:,np.newaxis])
    P *= np.eye(len(nu))[events][:,:,np.newaxis]

    grad_nu = P0.sum(axis=0)/nu - (end_time-start_time)
    hess_nu = -P0.sum(axis=0)/(nu*nu)

    #check: signs, gamma, right/left, powers add to kappa, +/- in front of terms
    grad_alpha = (kappa/alpha)*(lambda_4d*np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:])[1:].sum(axis=0) \
            - (signs*kappa/alpha) * (lambda_4d_right * (np.where(lambda_3d_right==0,0,(lambda_3d_right)**(kappa-1)) * gamma_0[:,np.newaxis,:])[:,np.newaxis,:,:]).sum(axis=0)
    hess_alpha = (kappa*(kappa-1)/(alpha**2)) * (lambda_4d**2 * np.where(lambda_3d==0,0,P/lambda_3d**2)[:,np.newaxis,:,:])[1:].sum(axis=0) \
                - (kappa/alpha)**2 * ((lambda_4d * np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:])**2)[1:].sum(axis=0) \
                - (signs*kappa*(kappa-1)/(alpha*alpha)) * (lambda_4d_right**2 * (np.where(lambda_3d_right==0,0,lambda_3d_right**(kappa-2)) * gamma_0[:,np.newaxis,:])[:,np.newaxis,:,:]).sum(axis=0)

    #num_components
    grad_beta = kappa * np.where(lambda_3d==0,0,dlambda_dbeta_3d * P / lambda_3d)[1:].sum(axis=(0,1)) \
                + signs*kappa * ((lambda_3d_right**kappa).sum(axis=1)*gamma_1).sum(axis=0) \
                - signs*kappa * ((np.where(lambda_3d_right==0,0,lambda_3d_right**(kappa-1)) * dlambda_dbeta_3d_right).sum(axis=1)*gamma_0).sum(axis=0)
    hess_beta = kappa*(kappa-1) * (np.where(lambda_3d==0,0,dlambda_dbeta_3d/lambda_3d)**2 * P)[1:].sum(axis=(0,1)) \
                + kappa * np.where(lambda_3d==0,0,P * d2lambda_dbeta2_3d/lambda_3d)[1:].sum(axis=(0,1)) \
                - kappa**2 * (np.where(lambda_3d==0,0,P * dlambda_dbeta_3d/lambda_3d)**2)[1:].sum(axis=(0,1)) \
                - signs*kappa*(kappa-1) * ((np.where(lambda_3d_right==0,0,lambda_3d_right**(kappa-2)) * dlambda_dbeta_3d_right**2).sum(axis=1)*gamma_0).sum(axis=0) \
                + 2*signs*kappa*(kappa-2) * ((np.where(lambda_3d_right==0,0,lambda_3d_right**(kappa-1)) * dlambda_dbeta_3d_right).sum(axis=1)*gamma_1).sum(axis=0) \
                - signs*kappa*(kappa-2) * ((lambda_3d_right**kappa).sum(axis=1)*gamma_2).sum(axis=0) \
                + signs*kappa * ((np.where(lambda_3d_right==0,0,lambda_3d_right**(kappa-1)) * d2lambda_dbeta2_3d_right).sum(axis=1)*gamma_0).sum(axis=0)

    #|E| x |E| x num_components x num_marks
    grad_a = kappa[np.newaxis,np.newaxis,:,np.newaxis] * (dlambda_da_5d * np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:,np.newaxis])[1:].sum(axis=0) \
            - (signs*kappa)[np.newaxis,np.newaxis,:,np.newaxis] * ((np.where(lambda_3d_right==0,0,lambda_3d_right**(kappa-1))[:,np.newaxis,:,:,np.newaxis] * dlambda_da_5d_right).sum(axis=1) * gamma_0[:,np.newaxis,:,np.newaxis]).sum(axis=0)
    hess_a = (kappa*(kappa-1))[np.newaxis,np.newaxis,:,np.newaxis] * ((dlambda_da_5d)**2 * np.where(lambda_3d==0,0,P/(lambda_3d**2))[:,np.newaxis,:,:,np.newaxis])[1:].sum(axis=0) \
            + kappa[np.newaxis,np.newaxis,:,np.newaxis] * (d2lambda_da2_5d * np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:,np.newaxis])[1:].sum(axis=0) \
            - (kappa**2)[np.newaxis,np.newaxis,:,np.newaxis] * ((dlambda_da_5d * np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:,np.newaxis])**2)[1:].sum(axis=0) \
            - (signs*kappa*(kappa-1))[np.newaxis,np.newaxis,:,np.newaxis] * (dlambda_da_5d_right**2 * (gamma_0[:,np.newaxis,:] * np.where(lambda_3d_right==0,0,lambda_3d_right**(kappa-2)))[:,np.newaxis,:,:,np.newaxis]).sum(axis=0)

    #|E| x |E| x num_components x num_marks
    grad_b = kappa[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d * np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:]).sum(axis=0) \
            - (signs*kappa)[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d_right * (gamma_0[:,np.newaxis,:] * lambda_3d_right**(kappa-1))[:,np.newaxis,:,:])[:,:,:,:,np.newaxis] * marks[:,np.newaxis,np.newaxis,np.newaxis,:]).sum(axis=0)
    hess_b = - kappa[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d**2 * np.where(lambda_3d==0,0,P/lambda_3d**2)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:]**2)[1:].sum(axis=0) \
            + kappa[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d * np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:]**2 ).sum(axis=0) \
            - (kappa**2)[np.newaxis,np.newaxis,:,np.newaxis] * (((lambda_4d * np.where(lambda_3d==0,0,P/lambda_3d)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:])**2).sum(axis=0) \
            - (signs*kappa*(kappa-1))[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d_right**2 * (gamma_0[:,np.newaxis,:] * lambda_3d_right**(kappa-2))[:,np.newaxis,:,:])[:,:,:,:,np.newaxis] * marks[:,np.newaxis,np.newaxis,np.newaxis,:]**2).sum(axis=0)

    #num_components
    grad_kappa = (np.log(lambda_3d)*P)[1:].sum(axis=(0,1)) \
            - signs * (lambda_3d_right**kappa * (np.log(lambda_3d_right)*gamma_0[:,np.newaxis,:] - (beta*gamma_1)[:,np.newaxis,:])).sum(axis=(0,1))
    hess_kappa = (np.log(lambda_3d)**2 * (1-P)*P)[1:].sum(axis=(0,1)) \
            - signs * (lambda_3d_right**kappa * np.log(lambda_3d_right)**2*gamma_0[:,np.newaxis,:]).sum(axis=(0,1)) \
            + signs*beta * (lambda_3d_right**kappa * np.log(lambda_3d_right)*gamma_1[:,np.newaxis,:]).sum(axis=(0,1)) \
            - signs*beta**2 * (lambda_3d_right**kappa * gamma_2[:,np.newaxis,:]).sum(axis=(0,1))

    return (grad_nu,grad_alpha,grad_beta,grad_a,grad_b,grad_kappa), (hess_nu,hess_alpha,hess_beta,hess_a,hess_b,hess_kappa)

def intensities_residuals(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa):
    #|T| x |E| x |E| x num_components
    lambda_4d = np.empty((len(times),len(nu),len(nu),len(signs)))
    lambda_4d[:] = np.nan
    lambda_4d_right = np.empty((len(times),len(nu),len(nu),len(signs)))
    lambda_4d_right[:] = np.nan

    #|T|
    timediffs = np.diff(times,append=end_time)

    #num_components
    gamma = beta*kappa
    #|T| x num_components
    gamma_0 = (1-np.exp(-gamma*timediffs[:,np.newaxis]))/gamma

    #|T| x num_marks
    markdiffs = np.diff(marks,prepend=0,axis=0)

    lambdas = np.zeros((len(nu),len(nu),len(signs)))
    for i,(timediff,e,markdiff,mark) in enumerate(zip(timediffs,events,markdiffs,marks)):
        #Left limits
        lambda_4d[i] = lambdas

        #Update in reaction to event
        lambdas *= np.exp((b*markdiff).sum(axis=3))
        lambdas[e] += alpha[e] * np.exp(((b[e]+a[e])*mark).sum(axis=2))

        #Right limits
        lambda_4d_right[i] = lambdas

        #Progress time
        lambdas *= np.exp(-beta*timediff)

    #|T| x |E| x num_components
    lambda_3d = lambda_4d.sum(axis=1)
    lambda_3d_right = lambda_4d_right.sum(axis=1)

    #|T| x |E|
    intensities = np.choose(events, ((signs*lambda_3d**kappa).sum(axis=2) + nu).T)
    residuals = (signs*lambda_3d_right**kappa * gamma_0[:,np.newaxis,:]).sum(axis=(1,2)) + nu.sum()*timediffs

    return intensities, residuals

def loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa):
    intensities, residuals = intensities_residuals(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
    ll = np.log(intensities).sum() - residuals.sum()
    print('ll',ll)
    return ll

def initial_params(num_event_types, num_components, num_mark_variables):
    return np.random.exponential(size=num_event_types),\
        np.random.exponential(size=(num_event_types,num_event_types,num_components))*1e-9,\
        np.random.exponential(size=num_components)*100,\
        np.zeros((num_event_types,num_event_types,num_components,num_mark_variables),dtype=np.float64),\
        np.zeros((num_event_types,num_event_types,num_components,num_mark_variables),dtype=np.float64),\
        np.ones(num_components)

def get_data(train_file='../output/databento_collated/glbx-mdp3-20240724.csv'):
    df = pd.read_csv(train_file,skiprows=2,sep='|',usecols=['seconds','side','instrument','action','marks'])

    times = df.seconds.values
    events = np.zeros_like(times,dtype=np.int64)
    events = 2*(df.action=='T') + (df.action=='A') #+ 0*(df.action=='C') + 0*(df.action=='M')
    events = 2*events + (df.instrument=='ES')
    events = 2*events + (df.side=='A')
    marks = np.array(df.marks.apply(eval).tolist())[:,[2,5,13,16]]
    marks = pd.DataFrame(marks).fillna(method='ffill').values

    return times,events,marks

"""
def fit_model(times,events,marks,start_time, end_time, signs, params,lr=1e-2):
    step_max = None
    params_old = [p.copy() for p in params]
    while True:
        grad,hess = get_derivatives(times, events, marks, start_time, end_time, signs, *params)
        ll = loglikelihood(times, events, marks, start_time, end_time, signs, *params)

        print(f'{ll=}')
        print(f'{lr=}')
        if not all((np.isfinite(np.linalg.norm(g)).all() and np.isfinite(np.linalg.norm(h)).all() for g,h in zip(grad,hess))):
            if step_max is None:
                lr *= 1e-1
            else:
                lr *= 1e-1/np.sqrt(step_max)
            print(f'backtracking')
            params = [p.copy()*(1-1e-5) for p in params_old]
            continue
        else:
            params_old = [p.copy() for p in params]
            grad_old = [g.copy() for g in grad]
            hess_old = [h.copy() for h in hess]
            lr = min(1,9*lr)

        print('changing params')
        step_max = 0
        hess_negative = True
        for k,(g,h,p) in enumerate(zip(grad,hess,params)):
            step = g/abs(h)
            print(abs(step).max() if step.size>0 else 0, (h<0).all(), (h<0).mean())
            hess_negative = hess_negative and (h<0).all()
            step_max = max(step_max,abs(step).max() if step.size>0 else 0)
            params[k] += lr*step
        print([np.linalg.norm(p) for p in params])

        yield signs,params,grad,hess
        if step_max < tol and hess_negative:
            return signs,params

"""
def fit_model(train_files, num_event_types, num_mark_variables, start_time, end_time, lr=1e-6,tol=5e-2):
    print('Initialising')

    signs = [1]

    params = initial_params(num_event_types,len(signs),num_mark_variables)
    #aic,signs,params = max([(ll,signs,params) for ll,signs,params in [(loglikelihood(times, events, marks, start_time, end_time, signs, *params)-sum(map(lambda x : x.size,params)), signs, params) for _ in ((print(x_),x_)[1] for x_ in range(40)) for signs in [np.array([1,1,1])] for (times,events,marks) in [get_data(train_files[np.random.choice([*range(len(train_files))])])] for params in [initial_params(num_event_types,len(signs),num_mark_variables)]] if np.isfinite(ll)])
    params = list(params)
    times,_,_ = get_data(train_files[0])
    params[0] *= len(times)/(end_time-start_time)/params[0].size
    params[1] *= len(times)/(end_time-start_time)/params[0].size
    params[2] *= 1e6
    print(f'{params=}')

    signs = np.array(sorted(signs))
    print(f'{len(signs)=}')
    #print(f'{aic=}')
    print(f'{[x.shape for x in params]}')

    while True:
        for train_file in sorted(train_files):
            print(train_file)
            times,events,marks = get_data(train_file)
            grad, hess = get_derivatives(times, events, marks, start_time, end_time, signs, *params)
            params = [p+(g/abs(h)).clip(-1,1) for p,g,h in zip(params,grad,hess)]
            print([(abs(g/abs(h))).max() for g,h in zip(grad,hess)])
            print(params)


    params_old = [p.copy() for p in params]
    grad_old = [0 for p in params]
    hess_old = [0 for p in params]

    i = 0
    step_max = None
    while True:
        grad = [0 for _ in params]
        hess = [0 for _ in params]
        ll = 0
        for train_file in train_files:
            times,events,marks = get_data(train_file)

            g, h = get_derivatives(times, events, marks, start_time, end_time, signs, *params)
            for k in range(len(params)):
                grad[k] += g[k]
                hess[k] += h[k]
            ll += loglikelihood(times, events, marks, start_time, end_time, signs, *params)

        print(f'{ll=}')
        print(f'{lr=}')
        if not all((np.isfinite(np.linalg.norm(g)).all() and np.isfinite(np.linalg.norm(h)).all() for g,h in zip(grad,hess))):
            params[1] = abs(params[1])
            if step_max is None:
                lr *= 1e-1
            else:
                lr *= 1e-1/np.sqrt(step_max)
            print(f'backtracking')
            params = [p.copy()*(1-1e-5) for p in params_old]
            continue
        else:
            params_old = [p.copy() for p in params]
            grad_old = [g.copy() for g in grad]
            hess_old = [h.copy() for h in hess]
            lr = min(1,9*lr)

        print('changing params')
        step_max = 0
        hess_negative = True
        for k,(g,h,p) in enumerate(zip(grad,hess,params)):
            print('gradient step only no hessian')
            step = (g/abs(h)).clip(-1,1)
            print(abs(step).max() if step.size>0 else 0, (h<0).all(), (h<0).mean())
            hess_negative = hess_negative and (h<0).all()
            step_max = max(step_max,abs(step).max() if step.size>0 else 0)
            params[k] += lr*step
        print([np.linalg.norm(p) for p in params])

        yield signs,params,grad,hess
        if step_max < tol and hess_negative:
            return signs,params

        i += 1




























def check_gradients(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa, delta=1e-3):
    (grad_nu,grad_alpha,grad_beta,grad_a,grad_b,grad_kappa), (hess_nu,hess_alpha,hess_beta,hess_a,hess_b,hess_kappa) = get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)

    print('nu')
    for idx in np.ndindex(nu.shape):
        step = np.zeros(nu.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu+step, alpha, beta, a, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_nu[idx])

        finite_difference = get_derivatives(times, events, marks, start_time, end_time, signs, nu+step, alpha, beta, a, b, kappa)[0][0][idx] \
                            - get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)[0][0][idx]
        print(finite_difference / delta)
        print(hess_nu[idx])


    print('alpha')
    for idx in np.ndindex(alpha.shape):
        step = np.zeros(alpha.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha+step, beta, a, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_alpha[idx])

        finite_difference = get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha+step, beta, a, b, kappa)[0][1][idx] \
                            - get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)[0][1][idx]
        print(finite_difference / delta)
        print(hess_alpha[idx])

    print('beta')
    for idx in np.ndindex(beta.shape):
        step = np.zeros(beta.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta+step, a, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_beta[idx])

        finite_difference = get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta+step, a, b, kappa)[0][2][idx] \
                            - get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)[0][2][idx]
        print(finite_difference / delta)
        print(hess_beta[idx])

    print('a')
    for idx in np.ndindex(a.shape):
        step = np.zeros(a.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a+step, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_a[idx])

        finite_difference = get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a+step, b, kappa)[0][3][idx] \
                            - get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)[0][3][idx]
        print(finite_difference / delta)
        print(hess_a[idx])

    print('b')
    for idx in np.ndindex(b.shape):
        step = np.zeros(b.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b+step, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_b[idx])

        finite_difference = get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b+step, kappa)[0][4][idx] \
                            - get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)[0][4][idx]
        print(finite_difference / delta)
        print(hess_b[idx])

    print('kappa')
    for idx in np.ndindex(kappa.shape):
        step = np.zeros(kappa.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa+step) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_kappa[idx])

        finite_difference = get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa+step)[0][5][idx] \
                            - get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)[0][5][idx]
        print(finite_difference / delta)
        print(hess_kappa[idx])
