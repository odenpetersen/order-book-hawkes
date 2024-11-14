import numpy as np

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
        dlambda_da[e] += mark * (alpha[e] * np.exp(((b[e]+a[e])*mark).sum(axis=2)))[:,:,np.newaxis]
        d2lambda_da2 *= np.exp((b*markdiff).sum(axis=3))[:,:,:,np.newaxis]
        dlambda_da[e] += mark**2 * (alpha[e] * np.exp(((b[e]+a[e])*mark).sum(axis=2)))[:,:,np.newaxis]

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

    print(lambda_4d[1,0,0,0])
    print(dlambda_da_5d[1,0,0,0,0])

    0/0

    #|T| x |E| x num_components
    lambda_3d = lambda_4d.sum(axis=1)
    lambda_3d_right = lambda_4d_right.sum(axis=1)

    #|T| x |E| x num_components
    components = signs*lambda_3d**kappa

    #|T| x |E|
    intensities = components.sum(axis=2) + nu

    #|T| x |E|
    P0 = nu / intensities
    P0 *= np.eye(len(nu))[events]

    #|T| x |E| x num_components
    P = components / intensities[:,:,np.newaxis]
    P *= np.eye(len(nu))[events][:,:,np.newaxis]

    grad_nu = P0.sum(axis=0)/nu - (end_time-start_time)
    hess_nu = -P0.sum(axis=0)/(nu*nu)

    #check: signs, gamma, right/left, powers add to kappa, +/- in front of terms
    grad_alpha = (kappa/alpha)*(lambda_4d*(P/lambda_3d)[:,np.newaxis,:,:])[1:].sum(axis=0) \
            - (signs*kappa/alpha) * (lambda_4d_right * (((lambda_3d_right)**(kappa-1)) * gamma_0[:,np.newaxis,:])[:,np.newaxis,:,:]).sum(axis=0)
    hess_alpha = (kappa*(kappa-1)/(alpha**2)) * (lambda_4d**2 * (P/lambda_3d**2)[:,np.newaxis,:,:])[1:].sum(axis=0) \
                - (kappa/alpha)**2 * ((lambda_4d * (P/lambda_3d)[:,np.newaxis,:,:])**2)[1:].sum(axis=0) \
                - (signs*kappa*(kappa-1)/(alpha*alpha)) * (lambda_4d_right**2 * (lambda_3d_right**(kappa-2) * gamma_0[:,np.newaxis,:])[:,np.newaxis,:,:]).sum(axis=0)

    #num_components
    grad_beta = kappa * (dlambda_dbeta_3d * P / lambda_3d)[1:].sum(axis=(0,1)) \
                + signs*kappa * ((lambda_3d_right**kappa).sum(axis=1)*gamma_1).sum(axis=0) \
                - signs*kappa * ((lambda_3d_right**(kappa-1) * dlambda_dbeta_3d_right).sum(axis=1)*gamma_0).sum(axis=0)
    hess_beta = kappa*(kappa-1) * ((dlambda_dbeta_3d/lambda_3d)**2 * P)[1:].sum(axis=0) \
                + kappa * (P * d2lambda_dbeta2_3d/lambda_3d)[1:].sum(axis=0) \
                - kappa**2 * ((P * dlambda_dbeta_3d/lambda_3d)**2)[1:].sum(axis=0) \
                - signs*kappa*(kappa-1) * ((lambda_3d_right**(kappa-2) * dlambda_dbeta_3d_right**2).sum(axis=1)*gamma_0).sum(axis=0) \
                + signs*2*kappa*(kappa-2) * ((lambda_3d_right**(kappa-1) * dlambda_dbeta_3d_right).sum(axis=1)*gamma_1).sum(axis=0) \
                - signs*kappa*(kappa-2) * ((lambda_3d_right**kappa).sum(axis=1)*gamma_2).sum(axis=0) \
                + signs*kappa * ((lambda_3d_right**(kappa-1) * d2lambda_dbeta2_3d_right).sum(axis=1)*gamma_0).sum(axis=0)

    #|E| x |E| x num_components x num_marks
    print(f'{dlambda_da_5d.shape}')
    print(f'{dlambda_da_5d_right.shape}')
    print(f'{P.shape}')
    print(f'{lambda_3d.shape}')
    print(f'{lambda_3d_right.shape}')
    print(f'{gamma_0.shape}')
    grad_a = kappa[np.newaxis,np.newaxis,:,np.newaxis] * (dlambda_da_5d * (P/lambda_3d)[:,np.newaxis,:,:,np.newaxis])[1:].sum(axis=0) \
            - (signs*kappa)[np.newaxis,np.newaxis,:,np.newaxis] * (((lambda_3d_right**(kappa-1))[:,np.newaxis,:,:,np.newaxis] * dlambda_da_5d_right).sum(axis=1) * gamma_0[:,np.newaxis,:,np.newaxis]).sum(axis=0)
    hess_a = (kappa*(kappa-1))[np.newaxis,np.newaxis,:,np.newaxis] * ((dlambda_da_5d)**2 * (P/(lambda_3d**2))[:,np.newaxis,:,:,np.newaxis])[1:].sum(axis=0) \
            + kappa[np.newaxis,np.newaxis,:,np.newaxis] * (d2lambda_da2_5d * (P/lambda_3d)[:,np.newaxis,:,:,np.newaxis])[1:].sum(axis=0) \
            - (kappa**2)[np.newaxis,np.newaxis,:,np.newaxis] * ((dlambda_da_5d * (P/lambda_3d)[:,np.newaxis,:,:,np.newaxis])**2)[1:].sum(axis=0) \
            - (signs*kappa*(kappa-1))[np.newaxis,np.newaxis,:,np.newaxis] * (dlambda_da_5d_right**2 * (gamma_0[:,np.newaxis,:] * lambda_3d_right**(kappa-2))[:,np.newaxis,:,:,np.newaxis]).sum(axis=0)

    #|E| x |E| x num_components x num_marks
    grad_b = kappa[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d * (P/lambda_3d)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:]).sum(axis=0) \
            - (signs*kappa)[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d_right * (gamma_0[:,np.newaxis,:] * lambda_3d_right**(kappa-1))[:,np.newaxis,:,:])[:,:,:,:,np.newaxis] * marks[:,np.newaxis,np.newaxis,np.newaxis,:]).sum(axis=0)
    hess_b = - kappa[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d**2 * (P/lambda_3d**2)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:]**2)[1:].sum(axis=0) \
            + kappa[np.newaxis,np.newaxis,:,np.newaxis] * ((lambda_4d * (P/lambda_3d)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:]**2 ).sum(axis=0) \
            - (kappa**2)[np.newaxis,np.newaxis,:,np.newaxis] * (((lambda_4d * (P/lambda_3d)[:,np.newaxis,:,:])[1:,:,:,:,np.newaxis] * marks[:-1,np.newaxis,np.newaxis,np.newaxis,:])**2).sum(axis=0) \
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
    return np.log(intensities).sum() - residuals.sum()

def check_gradients(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa, delta=1e-3):
    (grad_nu,grad_alpha,grad_beta,grad_a,grad_b,grad_kappa), (hess_nu,hess_alpha,hess_beta,hess_a,hess_b,hess_kappa) = get_derivatives(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)

    """
    print('nu')
    for idx in np.ndindex(nu.shape):
        step = np.zeros(nu.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu+step, alpha, beta, a, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_nu[idx])

    print('alpha')
    for idx in np.ndindex(alpha.shape):
        step = np.zeros(alpha.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha+step, beta, a, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_alpha[idx])

    print('beta')
    for idx in np.ndindex(beta.shape):
        step = np.zeros(beta.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta+step, a, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_beta[idx])

    print('a')
    for idx in np.ndindex(a.shape):
        step = np.zeros(a.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a+step, b, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_a[idx])
    """

    print('b')
    for idx in np.ndindex(b.shape):
        step = np.zeros(b.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b+step, kappa) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_b[idx])

    print('kappa')
    for idx in np.ndindex(kappa.shape):
        step = np.zeros(kappa.shape)
        step[idx] = delta
        finite_difference = loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa+step) \
                            - loglikelihood(times, events, marks, start_time, end_time, signs, nu, alpha, beta, a, b, kappa)
        print(finite_difference / delta)
        print(grad_kappa[idx])
