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
def simulate(signs,nu,alpha,beta,a,b,kappa, start_time, end_time):
    lambdas = np.zeros((len(nu),len(nu),len(signs)))

    mark = np.random.normal(size=a.shape[-1])
    time = start_time
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
        if time>end_time:
            return
        
        event = np.random.choice([*range(len(nu))],p=intensity/intensity.sum())

        prev_mark = mark
        mark = np.random.normal(size=a.shape[-1])

        yield time,event,mark

        lambdas *= np.exp((b*(mark-prev_mark)).sum(axis=3))
        lambdas[event] += alpha[event] * np.exp(((b[event]+a[event])*mark).sum(axis=2))
