/*
 * nu(t) + ( sum_{t_} alpha[state] * exp(continuous_marks@c[state]) exp(-beta[state] * (t-t_)) )[state_prev] **2
 */

Event {
	float time;
	float weight;
	std::vector<int> discrete_marks;
	std::vector<float> continuous_marks;
}

Realisation {
	float start_time;
	float end_time;
	std::vector<event> get_events() { } // Should be a generator instead.
}

Kernel {
	update_intensity(event e) {
	}

	simulate(reversed=false) {
	}

	intensity_upper_bound() {
	}

	get_params() {
	}

	get_intensities(std::vector<event> events) {
	}

	em_step(realisation, weights) {
	}

	em(std::vector<realisation> realisations, int max_iter = -1) { //Not virtual function, shouldnt need to be overwritten
		if (max_iter == -1) {
			em_step(realisation)
			while (!self.em_step(realisation))
		}
		//Todo later: parallelise across realisations
		/*
		if max_iter is None:
		    while not self.em_step(start_time, end_time, times, events, states, weights):
			continue
		    return True
		else:
		    for _ in range(max_iter):
			if self.em_step(start_time, end_time, times, events, states, weights):
			    return True
		    return False
		*/
	}
}

CompositeKernel {
	em_step(process process, std::vector<event> events) {
		for (e : events) {
			;
		}
	}
	simulate(reversed=false) {
	}
	intensity_upper_bound() {
	}
}

StateHawkesKernel {
	//momentum
	simulate(reversed=false) {
	}
	intensity_upper_bound() {
	}
}

QuadraticStateHawkesKernel {
	simulate(reversed=false) {
	}
	intensity_upper_bound() {
	}
}

RegressiveStateHawkesKernel {

	simulate(reversed=false) {
	}
	intensity_upper_bound() {
	}
}

RegressiveQuadraticStateHawkesKernel {

	simulate(reversed=false) {
	}
	intensity_upper_bound() {
	}
}


int main() {
	//momentum
	//constraints
}
