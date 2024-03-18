import numpy as np
def generate_events(base_rate = 1,alpha=0.1,beta=0.2,maxtime = 500):
    endogenous_intensity = 0
    events = []
    T = None
    while T is None or T < maxtime:
        if T is None:
            T = 0
        else:
            T = events[-1][0]

        while True:
            U = np.random.exponential(1/(base_rate + endogenous_intensity))
            decay_factor = np.exp(-beta*U)
            T = T + U
            endogenous_intensity *= decay_factor
            if np.random.uniform() < (base_rate + endogenous_intensity*decay_factor) / (base_rate + endogenous_intensity):
                break

        event = (T,base_rate+endogenous_intensity)
        events.append(event)
        endogenous_intensity += alpha

    return events[:-1]


