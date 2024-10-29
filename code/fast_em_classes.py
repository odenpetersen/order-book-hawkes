import numpy as np
import scipy as sp
import scipy.stats
import itertools
import jax
import jax.numpy as jnp
import jax.flatten_util
import tqdm
class Kernel():
    def __init__(self):
        pass
    def update_intensity(self, time=None, event=None, state=None, intensity_state=None):
        raise Exception('Kernel is an abstract class. Please use a subclass instead.')
    def simulate(self):
        raise Exception('Kernel is an abstract class. Please use a subclass instead.')
    def intensity_upper_bound(self):
        raise Exception('Kernel is an abstract class. Please use a subclass instead.')
    def get_params(self):
        raise Exception('Kernel is an abstract class. Please use a subclass instead.')
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        raise Exception('Not implemented.')
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        raise Exception('Kernel is an abstract class. Please use a subclass instead.')
    def em(self, start_time, end_time, times, events=None, states=None, weights=None, max_iter=None):
        if max_iter is None:
            while not self.em_step(start_time, end_time, times, events, states, weights):
                continue
            return True
        else:
            for _ in range(max_iter):
                if self.em_step(start_time, end_time, times, events, states, weights):
                    return True
            return False
        
class CompositeKernel(Kernel):
    def __init__(self,components):
        super().__init__()
        self.components = components
    def __repr__(self):
        return f'CompositeKernel({self.components})'
    def get_params(self):
        return [component.get_params() for component in self.components]
    def get_component_intensities(self, start_time, end_time, times, events=None, states=None):
        return np.vstack([component.get_intensities(start_time,end_time,times,events,states) for component in self.components])
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        return self.get_component_intensities(start_time,end_time,times,events,states).sum(axis=0)
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        if weights is None:
            weights = np.ones_like(times)
        #Determine weights with component intensities
        #print([component.get_intensities(start_time, end_time, times, events, states).shape for component in self.components])
        weights_matrix = np.vstack([component.get_intensities(start_time, end_time, times, events, states) for component in self.components])
        weights_matrix /= weights_matrix.sum(axis=0)
        print(f'{weights_matrix.shape=}')
        #Weighted EM step on components
        converged = all([component.em_step(start_time, end_time, times, events, states, component_weights * weights)\
                             for component, component_weights in zip(self.components, weights_matrix)])
        print('components',weights_matrix.mean(axis=1))
        print(self)
        return converged
            
class ConstantKernel(Kernel):
    def __init__(self, nu):
        super().__init__()
        self.nu = np.atleast_1d(nu)
    def get_params(self):
        return self.nu
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        if events is None:
            events = np.zeros_like(times,dtype=int)
        return self.nu[events].squeeze()
    def __repr__(self):
        return f'ConstantKernel({self.nu})'
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        if events is None:
            events = np.zeros_like(times,dtype=int)
        nu = ((1 if weights is None else weights.reshape(-1,1))*np.eye(self.nu.shape[0])[events]).sum(axis=0)/(end_time-start_time).squeeze()
        
        converged = np.isclose(nu,self.nu).all()
        self.nu = nu
        print('nu',nu)
        return converged
            
class MononomialExogenousKernel(Kernel):
    def __init__(self, nu, degree, negative=False):
        super().__init__()
        self.nu = np.atleast_1d(nu)
        self.degree = degree
        self.negative = negative
    def get_params(self):
        return self.nu
    def normalise_times(self, start_time, end_time, times):
        return (times-start_time)/(end_time-start_time)
    def poly_values(self, start_time, end_time, times):
        return self.normalise_times(start_time, end_time, times)**self.degree
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        poly_values = self.poly_values(start_time, end_time, times)
        if self.negative:
            poly_values = poly_values
        if events is not None:
            return self.nu[events] * poly_values
        else:
            return self.nu * poly_values
    def __repr__(self):
        return f'ConstantKernel({self.nu})'
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        if events is None:
            events = np.zeros_like(times,dtype=int)
        compensator = (end_time-start_time) * self.degree/(self.degree+1)
        if self.negative:
            compensator = (end_time - start_time) - compensator
        nu = ((1 if weights is None else weights.reshape(-1,1))*np.eye(self.nu.shape[0])[events]).sum(axis=0) / compensator
        
        converged = np.isclose(nu,self.nu).all()
        self.nu = nu
        #print('nu',nu)
        return converged

class PolynomialExogenousKernel(CompositeKernel):
    def __init__(self, nus, degree):
        nus = np.atleast_1d(nus) / (2*degree+1)
        super().__init__([ConstantKernel(nus)] + [MononomialExogenousKernel(nus,n+1,negative) for n in range(1,degree+1) for negative in (True,False)])
    def __repr__(self):
        return f'PolynomialExogenousKernel({[component.nu for component in self.components]})'
            
class SmoothingKernel(Kernel):
    def __init__(self,nu,bandwidth=None):
        self.nu = np.atleast_1d(nu)
        self.bandwidth = bandwidth
        self.kde = None
    def __repr__(self):
        if self.bandwidth is None:
            return f'SmoothingKernel({self.nu})'
        return f'SmoothingKernel({self.nu},{self.bandwidth})'
    def get_params(self):
        return [self.nu,self.bandwidth]
    def set_kde(self, start_time, end_time, points, weights=None):
        self.kde = sp.stats.gaussian_kde(points, bw_method=self.bandwidth, weights=weights)
    def get_normalised_density(self, start_time, end_time, points, weights=None):
        if self.kde is None:
            self.set_kde(start_time, end_time, points, weights)
        return (end_time - start_time) * self.kde(points) / self.kde.integrate_box_1d(start_time, end_time)
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        if self.kde is None:
            self.set_kde(start_time, end_time, times)
        if len(self.nu)>1:
            assert events is not None
            intensities = np.full_like(times, np.nan)
            for e,nu in enumerate(self.nu):
                points = times[events==e]
                intensities[events==e] = nu * self.get_normalised_density(start_time, end_time, times)
            return intensities
        else:
            return self.nu * self.get_normalised_density(start_time, end_time, times)
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        self.set_kde(start_time, end_time, times, weights)
        if events is None:
            events = np.zeros_like(times,dtype=int)
        #nu = ((1 if weights is None else weights.reshape(-1,1))*np.eye(self.nu.shape[0])[events]).sum(axis=0)/(end_time-start_time)
        #densities = (end_time - start_time) * self.kde(times).reshape(-1,1) / self.kde.integrate_box_1d(start_time,end_time)
        weights = (1 if weights is None else weights.reshape(-1,1))
        events_one_hot_encoding = np.eye(self.nu.shape[0])[events]
        #nu = (weights * events_one_hot_encoding * densities).sum(axis=0) / (end_time-start_time)
        nu = (weights * events_one_hot_encoding).sum(axis=0) / (end_time-start_time)
        
        converged = np.isclose(nu,self.nu).all()
        self.nu = nu
        #print('nu',nu)
        return converged

class ReflectedSmoothingKernel(SmoothingKernel):
    def __init__(self, nu, bandwidth=None, num_reflected=None, frac_reflected=None):
        super().__init__(nu,bandwidth)
        assert num_reflected is None or frac_reflected is None
        self.num_reflected = num_reflected
        self.frac_reflected = frac_reflected
    def __repr__(self):
        if self.bandwidth is None:
            return f'ReflectedSmoothingKernel({self.nu},{self.num_reflected if self.num_reflected is not None else self.frac_reflected})'
        return f'ReflectedSmoothingKernel({self.nu},{self.bandwidth},{self.num_reflected if self.num_reflected is not None else self.frac_reflected})'
    def set_kde(self, start_time, end_time, points, weights=None):
        num_reflected = self.num_reflected if self.num_reflected is not None else int(len(points)*self.frac_reflected)
        points = np.hstack([2*start_time-points[:num_reflected][::-1],points,2*end_time-points[-num_reflected:][::-1]])
        if weights is not None:
            weights = np.hstack([weights[:num_reflected][::-1],weights,weights[-num_reflected:][::-1]])
        super().set_kde(start_time, end_time, points, weights)

class ExponentialHawkesKernel(Kernel):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = np.atleast_3d(alpha)
        self.beta = np.atleast_3d(beta)
    def get_params(self):
        return [self.alpha,self.beta]
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        """
        if events is None:
            events = itertools.repeat(0)
        if states is None:
            states = itertools.repeat(0)
            
        intensities = np.full_like(times,np.nan)
        
        intensity = np.zeros_like(self.alpha)
        
        prev_time = start_time
        for i,(t,e,s) in enumerate(zip(times,events,states)):
            intensity *= np.exp(-self.beta * (t - prev_time))
            intensities[i] = intensity[:,e,:].sum()
            intensity[e,:,s] += self.alpha[e,:,s]
            prev_time = t
        return intensities
        """
        print('getting intensities')
        intensities= exponential_hawkes_get_intensities(self.alpha, self.beta, start_time, end_time, times, events, states)
        print('got intensities')
        return intensities
    def __repr__(self):
        alpha,beta = (str(x.squeeze()).replace('\n','') for x in (self.alpha,self.beta))
        return f'ExponentialHawkesKernel({alpha},{beta})'
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        print('em step')
        self.alpha, self.beta, converged = exponential_hawkes_em_step(self.alpha, self.beta, start_time, end_time, times, events, states, weights)
        print('done em step')
        return converged
#@jax.jit
def exponential_hawkes_get_intensities(self_alpha, self_beta, start_time, end_time, times, events, states):
    if events is None:
        events = itertools.repeat(0)
    if states is None:
        states = itertools.repeat(0)
        
    intensities = jnp.full_like(times,jnp.nan)
    
    intensity = jnp.zeros_like(self_alpha)
    
    prev_time = start_time
    for i,(t,e,s) in tqdm.tqdm(enumerate(zip(times,events,states)),total=len(times)):
        intensity = intensity * jnp.exp(-self_beta * (t - prev_time))
        intensities = intensities.at[i].set(intensity[:,e,:].sum())
        intensity = intensity.at[e,:,s].set(intensity[e,:,s] + self_alpha[e,:,s])
        prev_time = t
    return intensities
#@jax.jit
def exponential_hawkes_em_step(self_alpha, self_beta, start_time, end_time, times, events, states, weights):
    if events is None:
        events = itertools.repeat(0)
    if states is None:
        states = itertools.repeat(0)
    if weights is None:
        weights = itertools.repeat(1)
    
    #intensities = jnp.full_like(times,jnp.nan)
    
    log_alpha_coef = jnp.zeros_like(self_alpha)
    decaying_sum_t_i = jnp.zeros_like(self_alpha)
    beta_coef = jnp.zeros_like(self_alpha)
    alpha_on_beta_coef = jnp.zeros_like(self_alpha)
    intensity = jnp.zeros_like(self_alpha)
    
    prev_time = start_time
    for j,(t,e,s,w) in tqdm.tqdm(enumerate(zip(times,events,states,weights)),total=len(times)):
        decay = jnp.exp(-self_beta*(t-prev_time))
        decaying_sum_t_i *= decay
        intensity *= decay
        total_intensity = intensity[:,e,:].sum()
        #intensities[j] = total_intensity
        
        multiplier = jnp.where(jnp.isclose(total_intensity,0), 0, w / total_intensity)
        log_alpha_coef = log_alpha_coef.at[:,e,s].set(log_alpha_coef[:,e,s] + multiplier * intensity[:,e,s])
        beta_coef = beta_coef.at[:,e,s].set(beta_coef[:,e,s] + multiplier * (decaying_sum_t_i - intensity * t)[:,e,s])
        alpha_on_beta_coef = alpha_on_beta_coef.at[e,:,s].set(alpha_on_beta_coef[e,:,s] + 1) #Quasi EM approximation
        
        intensity = intensity.at[e,:,s].set(intensity[e,:,s] + self_alpha[e,:,s])
        decaying_sum_t_i = decaying_sum_t_i.at[e,:,s].set(decaying_sum_t_i[e,:,s] + t*self_alpha[e,:,s])
        prev_time = t
        
    #print(log_alpha_coef,alpha_on_beta_coef,beta_coef,multiplier,total_intensity,intensity,intensity[:,0].sum())
    #alpha = np.nan_to_num(np.where(beta_coef<0,-log_alpha_coef**2/(alpha_on_beta_coef*beta_coef),0))
    #beta = np.nan_to_num(np.where(beta_coef<0,-log_alpha_coef/beta_coef,0))
    #if (alpha_on_beta_coef==0).any():
    #    raise Exception(f'No observations of (event_type,state) == {set([(e,s) for e in np.arange(alpha.shape[0]) for s in np.arange(alpha.shape[2]) if (alpha_on_beta_coef[e,:,s]==0).any()])}')
    unused_component = (beta_coef==0) & (log_alpha_coef==0)
    alpha = jnp.where(unused_component, 0, -log_alpha_coef**2 / (alpha_on_beta_coef * beta_coef))
    beta = jnp.where(unused_component, 1, -log_alpha_coef/beta_coef)
    #print(alpha)
    #print(beta)
    if jnp.isnan(alpha).any() or jnp.isnan(beta).any():
        alpha[jnp.isnan(alpha)] = self_alpha[jnp.isnan(alpha)]
        beta[jnp.isnan(beta)] = self_beta[jnp.isnan(beta)]
        #print(alpha);print(beta)
        return False
    #print('alpha and beta',alpha,beta,self.alpha,self.beta)
    converged = jnp.isclose([*alpha.reshape(-1),*beta.reshape(-1)],[*self_alpha.reshape(-1),*self_beta.reshape(-1)]).all()
    self_alpha, self.beta = alpha, beta
    
    return self_alpha, self_beta, converged


"""
    @jax.jit
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        if events is None:
            events = itertools.repeat(0)
        if states is None:
            states = itertools.repeat(0)
        if weights is None:
            weights = itertools.repeat(1)
        
        #intensities = jnp.full_like(times,jnp.nan)
        
        log_alpha_coef = jnp.zeros_like(self.alpha)
        decaying_sum_t_i = jnp.zeros_like(self.alpha)
        beta_coef = jnp.zeros_like(self.alpha)
        alpha_on_beta_coef = jnp.zeros_like(self.alpha)
        intensity = jnp.zeros_like(self.alpha)
        
        prev_time = start_time
        for j,(t,e,s,w) in enumerate(zip(times,events,states,weights)):
            decay = jnp.exp(-self.beta*(t-prev_time))
            decaying_sum_t_i *= decay
            intensity *= decay
            total_intensity = intensity[:,e,:].sum()
            #intensities[j] = total_intensity
            
            multiplier = 0 if jnp.isclose(total_intensity,0) else (w / total_intensity)
            log_alpha_coef[:,e,s] += multiplier * intensity[:,e,s]
            beta_coef[:,e,s] += multiplier * (decaying_sum_t_i - intensity * t)[:,e,s]
            alpha_on_beta_coef[e,:,s] += 1 #Quasi EM approximation
            
            intensity[e,:,s] += self.alpha[e,:,s]
            decaying_sum_t_i[e,:,s] += t*self.alpha[e,:,s]
            prev_time = t
            
        #print(log_alpha_coef,alpha_on_beta_coef,beta_coef,multiplier,total_intensity,intensity,intensity[:,0].sum())
        #alpha = np.nan_to_num(np.where(beta_coef<0,-log_alpha_coef**2/(alpha_on_beta_coef*beta_coef),0))
        #beta = np.nan_to_num(np.where(beta_coef<0,-log_alpha_coef/beta_coef,0))
        if (alpha_on_beta_coef==0).any():
            raise Exception(f'No observations of (event_type,state) == {set([(e,s) for e in np.arange(alpha.shape[0]) for s in np.arange(alpha.shape[2]) if (alpha_on_beta_coef[e,:,s]==0).any()])}')
        unused_component = (beta_coef==0) & (log_alpha_coef==0)
        alpha = jnp.where(unused_component, 0, -log_alpha_coef**2 / (alpha_on_beta_coef * beta_coef))
        beta = jnp.where(unused_component, 1, -log_alpha_coef/beta_coef)
        #print(alpha)
        #print(beta)
        if jnp.isnan(alpha).any() or jnp.isnan(beta).any():
            alpha[jnp.isnan(alpha)] = self.alpha[jnp.isnan(alpha)]
            beta[jnp.isnan(beta)] = self.beta[jnp.isnan(beta)]
            print(alpha);print(beta)
            return False
        #print('alpha and beta',alpha,beta,self.alpha,self.beta)
        converged = jnp.isclose([*alpha.reshape(-1),*beta.reshape(-1)],[*self.alpha.reshape(-1),*self.beta.reshape(-1)]).all()
        self.alpha, self.beta = alpha, beta
        
        return converged
"""

class TruncatedGaussianHawkesKernel(Kernel):
    def __init__(self, alpha, mu, sigma):
        raise Exception('TruncatedGaussianHawkesKernel is untested')
        super().__init__()
        self.alpha = np.atleast_3d(alpha)
        self.mu = np.atleast_3d(mu)
        self.sigma = np.atleast_3d(sigma)
    def kernel(self,x,e1,e2,state):
        return self.alpha[e1,e2,state]*np.exp(-((x-self.mu[e1,e2,state])/self.sigma[e1,e2,state])**2/2)
    def get_params(self):
        return [self.alpha,self.mu,self.sigma]
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        if events is None:
            events = itertools.repeat(0)
        if states is None:
            states = itertools.repeat(0)
            
        intensities = np.full_like(times,np.nan)
        
        trigger_times = []
        trigger_events = dict()
        trigger_states = dict()
        for i,(t,e,s) in tqdm.tqdm(enumerate(zip(times,events,states))):
            intensity_components = [self.kernel(t-trigger_time,trigger_events[trigger_time],e,trigger_states[trigger_time]) for trigger_time in trigger_times]
            intensities[i] = sum(intensity_components)

            trigger_times = [trigger_time for trigger_time,intensity in zip(trigger_times,intensity_components) if not np.all(np.isclose(intensity,0))]
            trigger_events = {t:trigger_events[t] for t in trigger_times}
            trigger_states = {t:trigger_states[t] for t in trigger_times}

            trigger_times.append(t)
            trigger_events[t] = e
            trigger_states[t] = s
        return intensities
    def __repr__(self):
        return f'TruncatedGaussianHawkesKernel({",".join(map(str,map(np.squeeze,self.get_params())))})'
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        if events is None:
            events = itertools.repeat(0)
        if states is None:
            states = itertools.repeat(0)
        if weights is None:
            weights = itertools.repeat(1)
        
        trigger_times = []
        trigger_events = dict()
        trigger_states = dict()
        weighted_sum_probas = np.zeros_like(self.alpha)
        weighted_sum_weighted_duration = np.zeros_like(self.alpha)
        weighted_sum_weighted_duration_squared = np.zeros_like(self.alpha)
        weighted_num_events = np.zeros((self.alpha.shape[0],self.alpha.shape[2]))
        for i,(t,e,s,w) in enumerate(zip(times,events,states,weights)):
            intensity_components = [{e_:self.kernel(t-trigger_time,trigger_events[trigger_time],e_,trigger_states[trigger_time]) for e_ in range(self.alpha.shape[0])} for trigger_time in trigger_times]
            intensity = sum([x[e] for x in intensity_components])
            probas = [x[e]/intensity for x in intensity_components]

            weighted_num_events[e][s] += w
            for trigger_event in range(self.alpha.shape[0]):
                for trigger_state in range(self.alpha.shape[2]):
                    weighted_sum_probas[trigger_event,e,trigger_state] += w*sum([b for b,trigger_time in zip(probas,trigger_times) if trigger_events[trigger_time]==trigger_event and trigger_states[trigger_time]==trigger_state])
                    weighted_sum_weighted_duration[trigger_event,e,trigger_state] += w*sum([b*(t-trigger_time) for b,trigger_time in zip(probas,trigger_times) if trigger_events[trigger_time]==trigger_event and trigger_states[trigger_time]==trigger_state])
                    weighted_sum_weighted_duration_squared[trigger_event,e,trigger_state] += w*sum([b*(t-trigger_time)**2 for b,trigger_time in zip(probas,trigger_times) if trigger_events[trigger_time]==trigger_event and trigger_states[trigger_time]==trigger_state])

            trigger_times = [trigger_time for n,trigger_time in enumerate(trigger_times) if not all([np.all(np.isclose(intensity_components[n][e_],0)) for e_ in intensity_components[n]])]
            trigger_events = {t:trigger_events[t] for t in trigger_times}
            trigger_states = {t:trigger_states[t] for t in trigger_times}

            trigger_times.append(t)
            trigger_events[t] = e
            trigger_states[t] = s

        @jax.jit
        def loss_function(alpha,mu,sigma):
            ll = weighted_sum_probas*jnp.log(alpha)
            ll -= weighted_sum_weighted_duration_squared / (2*sigma**2)
            ll += (mu/sigma**2) * weighted_sum_weighted_duration
            ll -= (mu/sigma)**2 * weighted_sum_probas / 2
            compensator = jnp.sqrt(2*jnp.pi)*weighted_num_events * (alpha*sigma*jax.scipy.stats.norm.cdf(mu/sigma)).sum(axis=1)
            return ll.sum() - compensator.sum()

        params,unflatten = jax.flatten_util.ravel_pytree([self.alpha,self.mu,self.sigma])
        f = lambda x : -loss_function(*unflatten(x))
        result = sp.optimize.minimize(fun=f,jac=jax.grad(f),x0=params)
        if not result.success:
            print(result)
            return False
        alpha,mu,sigma = unflatten(result.x)
        alpha = np.clip(alpha,0,np.inf)
        sigma = np.clip(np.abs(sigma),np.finfo(float).eps,np.inf)

        converged = np.isclose([*alpha.reshape(-1),*mu.reshape(-1),*sigma.reshape(-1)],[*self.alpha.reshape(-1),*self.mu.reshape(-1),*self.sigma.reshape(-1)]).all()
        self.alpha, self.mu = alpha, mu, sigma
        
        return converged

class StateDependentKernelSelection(Kernel):
    #I'm pretty sure this is NOT the same as in the paper
    #I think their kernels depend on the state at time of the triggering event, not the resultant event
    def __init__(self, kernels):
        super().__init__()
        self.kernels = kernels
    def get_params(self):
        return [kernel.get_params() for kernel in self.kernels]
    def __repr__(self):
        return f'StateDependentKernelSelection({self.kernels})'
    def get_intensities(self, start_time, end_time, times, events=None, states=None):
        assert states is not None
        intensities_matrix = np.vstack([kernel.get_intensities(start_time, end_time, times, events, states) for kernel in self.kernels])
        return intensities_matrix[states]
    def em_step(self, start_time, end_time, times, events=None, states=None, weights=None):
        assert states is not None
        for s,kernel in enumerate(self.kernels):
            kernel.em_step(weights = np.where(states==s,1,0))
