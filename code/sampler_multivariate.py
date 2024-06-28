import numpy as np
"""
E = number of event types
S = number of states
C = number of components in exponential kernel

base_rate : E x S
alpha : E x E x S x C
beta : E x E x S x C
transition : E x S x S
"""
def generate_events(base_rate,alpha,beta,transition,endogenous_intensity=None,maxtime = 500):
    num_event_types, _, num_states, num_components = alpha.shape
    event_types = np.arange(num_event_types)
    states = np.arange(num_states)
    if endogenous_intensity is None:
        endogenous_intensity = np.zeros_like(alpha)

    assert base_rate.shape == (num_event_types, num_states), (base_rate.shape, (num_event_types,num_states))
    assert alpha.shape == (num_event_types, num_event_types, num_states, num_components)
    assert beta.shape == (num_event_types, num_event_types, num_states, num_components)
    assert transition.shape == (num_event_types, num_states, num_states)
    assert endogenous_intensity.shape == (num_event_types, num_event_types, num_states, num_components)

    state = 0
    events = []
    T = None
    while T is None or T < maxtime:
        if T is None:
            T = 0
        else:
            T = events[-1]['time']

        while True:
            U = np.random.exponential(1/(base_rate[:,state].sum() + endogenous_intensity[:,:,state].sum()))
            decay_factor = np.exp(-beta[:,:,[state],:]*U)
            T = T + U
            endogenous_intensity *= decay_factor
            if T >= maxtime:
                return events
            if np.random.uniform() < (base_rate[:,state].sum() + endogenous_intensity[:,:,state].sum()) / (base_rate[:,state].sum() + (endogenous_intensity/decay_factor)[:,:,state].sum()):
                break

        total_intensity = base_rate[:,state] + endogenous_intensity[:,:,[state],:].sum(axis=(0,2,3))
        event_type = np.random.choice(event_types, p=total_intensity/total_intensity.sum())
        new_state = np.random.choice(states, p=transition[event_type,state])
        event = dict(time=T, type=event_type, base_rate=base_rate, endogenous_intensity=endogenous_intensity, total_intensity=total_intensity, old_state=state, new_state=new_state)
        state = new_state
        events.append(event)

        endogenous_intensity[event_type,:,new_state] += alpha[event_type,:,new_state]
