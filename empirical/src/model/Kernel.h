#include <utility>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <string>
#include <exception>

#include <Eigen/Dense>

#include "Types.h"
#include "constants.h"

std::random_device rd;
std::mt19937 generator(rd());


/*
 * Basic:
 * Linear Spline Background
 * State-dependent regressive hawkes + 'reverse state dependent'
 * Quadratic and beyond
 *
 * Sophisticated:
 * Dependence on number of orders in the book
 * Full order book simulation
 * */

class Kernel {
	public:
		Kernel(int num_event_types, REAL start_time, REAL end_time) : num_event_types(num_event_types), start_time(start_time), end_time(end_time) {
			reset();
		}

		virtual std::pair<Eigen::VectorXd,Eigen::VectorXd> get_hessian_and_gradient() = 0;

		virtual Eigen::VectorXd get_em_step() {
			auto [hess,grad] = get_hessian_and_gradient();
			return get_em_step(hess,grad);
		}

		virtual Eigen::VectorXd get_em_step(Eigen::VectorXd hess, Eigen::VectorXd grad) {
			return (hess.array()==0.0).select(0.0,grad.cwiseQuotient(hess.cwiseAbs()));
		}

		virtual Eigen::VectorXd get_params() = 0;

		virtual void set_params(Eigen::VectorXd new_params) = 0;

		virtual int get_param_count() = 0;

		//Return a time and event type label
		std::pair<REAL,int> simulate() {
			return {0.0,0};
			REAL total_timediff = 0;

			while (true) {
				REAL intensity_upper_bound = get_intensity_upper_bound();
				std::exponential_distribution<REAL> timediff_distribution(intensity_upper_bound);
				REAL timediff = timediff_distribution(generator);

				progress_time(timediff);
				total_timediff += timediff;

				Eigen::VectorXd intensities = get_intensities();

				REAL total_intensity = get_intensity();

				std::uniform_real_distribution<REAL> unif_distribution(0.0,1.0);
				REAL unif = unif_distribution(generator);
				if (unif > total_intensity / intensity_upper_bound) {
					std::discrete_distribution<int> event_type_distribution(intensities.begin(), intensities.end());
					int event_type = event_type_distribution(generator);

					progress_time(-total_timediff);

					return {current_time + timediff, event_type};
				}
			}
		}

		REAL get_intensity() {
			Eigen::VectorXd intensities = get_intensities();
			return std::accumulate(intensities.begin(), intensities.end(), 0);
		}

		virtual Eigen::VectorXd get_intensities() = 0;

		virtual REAL progress_time(REAL timediff) = 0;

		virtual std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) = 0;
		//Progress time (get residual)
		//attribute event (weighted)
		//trigger intensities

		void parameter_step(Eigen::VectorXd diff) {
			set_params(get_params() + diff);
		}

		virtual REAL get_intensity_upper_bound() = 0;

		virtual void reset() {
			current_time = start_time;
		};

		virtual void reset_event_type(int event_type) = 0;

		virtual std::string as_string() = 0;

		virtual void perturb(REAL amount) {};

		REAL start_time, end_time, current_time;
		int num_event_types;

		//KS test for streaming data https://personal.denison.edu/~lalla/papers/ks-stream.pdf

};

class PoissonKernel : public Kernel {
	public:
		PoissonKernel(int num_event_types, REAL start_time, REAL end_time,bool allow_negative=false) : Kernel(num_event_types, start_time, end_time), allow_negative(allow_negative) {
			nu = Eigen::VectorXd::Random(num_event_types).array() + 1.0;
			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types,0.0);
		}

		std::pair<Eigen::VectorXd,Eigen::VectorXd> get_hessian_and_gradient() override {
			Eigen::VectorXd gradient = weighted_event_counts.cwiseQuotient(nu);
			gradient.array() -= end_time-start_time;

			Eigen::VectorXd hessian = -(weighted_event_counts.array().log()-2*nu.cwiseAbs().array().log()).exp();
			/*
			std::cout << as_string() << std::endl;
			std::cout << "weighted event counts " << weighted_event_counts << std::endl;
			std::cout << "nu " << nu << std::endl;
			std::cout << "step " << -gradient.cwiseQuotient(hessian) << std::endl;
			*/

			return {hessian, gradient};
		}

		int get_param_count() {
			return num_event_types;
		}

		Eigen::VectorXd get_params() {
			return nu;
		}

		void set_params(Eigen::VectorXd new_params) {
			nu = new_params;
			if (!allow_negative) {
				nu = (nu.array()>=0.0).select(nu,Eigen::VectorXd::Random(num_event_types).array()+1.0);
			}
		}

		Eigen::VectorXd get_intensities() {
			return nu;
		}

		REAL progress_time(REAL timediff) override {
			current_time += timediff;
			if (!(timediff>=0)) {
				std::cout << "timediff " << timediff << std::endl;
			}
			return nu.sum()*timediff;
		}

		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			REAL timediff = observation.time - current_time;
			if (timediff<0) {
				std::cout << "Warning: current_time=" << current_time << " but observed " << observation << std::endl;
			}
			weighted_event_counts.array()[observation.event_type] += weight;
			REAL residual = progress_time(timediff);
			return {residual, nu(observation.event_type)};
		}

		REAL get_intensity_upper_bound() {
			return get_intensity();
		}

		void reset() override {
			current_time = start_time;
			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types, 0.0);
		};

		void reset_event_type(int event_type) {
			std::uniform_real_distribution<REAL> unif_distribution(0,1);
			nu(event_type) = unif_distribution(generator);
		}

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "PoissonKernel(" << nu.format(format) << ")";
			return ss.str();
		}

		Eigen::VectorXd weighted_event_counts;
	private:
		Eigen::VectorXd nu;
		bool allow_negative;
};

class LinearSpline : public Kernel {
	public:
		LinearSpline(int num_event_types, REAL start_time, REAL end_time) : Kernel(num_event_types, start_time, end_time) {
			coef = 1e-2*(Eigen::VectorXd::Random(num_event_types).array() + 1.0)/(end_time-start_time);

			std::uniform_real_distribution<REAL> unif_distribution(start_time,end_time);
			knot = unif_distribution(generator);

			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types,0.0);
		}

		std::pair<Eigen::VectorXd,Eigen::VectorXd> get_hessian_and_gradient() override {
			Eigen::VectorXd hessian(get_param_count());
			Eigen::VectorXd gradient(get_param_count());

			Eigen::VectorXd gradient_coef = weighted_event_counts.cwiseQuotient(coef).array() - 0.5 * (end_time - knot)*(end_time - knot);
			REAL gradient_knot = - weighted_max_reciprocal_sum_after_knot + (end_time - knot) * coef.sum();

			Eigen::VectorXd hess_coef = - weighted_event_counts.cwiseQuotient(coef.cwiseProduct(coef));
			REAL hess_knot = weighted_maxsquared_reciprocal_sum_after_knot - coef.sum();

			for (int i = 0; i < coef.size(); i++) {
				if (coef(i) == 0.0) {
					gradient_coef(i) = 0.0;
					hess_coef(i) = 0.0;
				}
			}

			gradient.segment(0, num_event_types) = gradient_coef;
			gradient.array()[num_event_types] = gradient_knot;

			hessian.segment(0, num_event_types) = hess_coef;
			hessian.array()[num_event_types] = hess_knot;

			//std::cout << as_string() << " weighted_event_counts " << weighted_event_counts << " time " << current_time << " intensity " << get_intensities() << std::endl;

			return {hessian, gradient};
		}

		Eigen::VectorXd get_params() {
			Eigen::VectorXd params(num_event_types + 1);
			params.block(0,0,num_event_types,1) = coef;
			params(num_event_types) = knot;
			return params;
		}

		int get_param_count() {
			return num_event_types+1;
		}

		void set_params(Eigen::VectorXd new_params) {
			coef = new_params.block(0,0,num_event_types,1);
			knot = new_params(num_event_types);
			if (knot > end_time) {
				coef.setZero();
				std::uniform_real_distribution<REAL> unif_distribution(start_time,end_time);
				knot = unif_distribution(generator);
			}
		}

		Eigen::VectorXd get_intensities() {
			return coef * std::max(current_time - knot, 0.0);
		}

		REAL progress_time(REAL timediff) override {
			REAL residual = [](std::pair<REAL,REAL> x){return 0.5*(x.second*x.second-x.first*x.first);}({std::max(coef.sum() * (current_time-knot), 0.0), std::max(coef.sum() * (current_time+timediff-knot), 0.0)});
			current_time += timediff;
			return residual;
		}

		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			REAL timediff = observation.time - current_time;
			REAL residual = progress_time(timediff);
			weighted_event_counts.array()[observation.event_type] += weight;
			//std::cout << current_time << "," << knot << std::endl;
			if (current_time > knot) {
				//std::cout << "current_time>knot " << this << std::endl;
				REAL max = std::max(current_time - knot,0.0);
				weighted_max_reciprocal_sum_after_knot += weight/max;
				weighted_maxsquared_reciprocal_sum_after_knot += weight/(max*max);
			}
			return {residual, get_intensities()(observation.event_type)};
		}

		REAL get_intensity_upper_bound() {
			return std::max(coef.sum() * (end_time - knot), 0.0);
			//This is quite inefficient for simulation purposes. Maybe write a custom simulation method for this subclass.
		}

		void reset() override {
			current_time = start_time;
			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types, 0.0);
			weighted_max_reciprocal_sum_after_knot = 0;
			weighted_maxsquared_reciprocal_sum_after_knot = 0;
		};

		void reset_event_type(int event_type) {
			std::uniform_real_distribution<REAL> unif_distribution(0,1);
			coef(event_type) = unif_distribution(generator);
		}

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "LinearSpline(" << knot << ";" << coef.format(format) << ")";
			return ss.str();
		}

		/*
		virtual Eigen::VectorXd get_em_step() {
			Eigen::VectorXd step = Eigen::VectorXd::Constant(get_param_count(),0.0);

			Eigen::VectorXd optimal_coef = 2*weighted_event_counts/(end_time - knot);
			step.segment(0,num_event_types) = (optimal_coef - coef);

			REAL knot_hess = weighted_maxsquared_reciprocal_sum_after_knot-optimal_coef.sum();
			REAL knot_grad = -weighted_max_reciprocal_sum_after_knot - (knot-end_time)*optimal_coef.sum();
			REAL knot_step = knot_hess==0 ? 0 : - knot_grad/knot_hess;
			//std::cout << "knot_hess, knot_grad, knot_step = " << knot_hess << "," << knot_grad << "," << knot_step << std::endl;
			step.array()[num_event_types] = knot_step;
			
			return step;
		}
		*/

		/*
		void perturb(REAL amount) override {
			std::uniform_real_distribution<REAL> unif_distribution(-1.0,1.0);
			REAL noise = amount*unif_distribution(generator);
			coef.array() += noise;
		}
		*/
	private:
		Eigen::VectorXd coef;
		REAL knot;
		Eigen::VectorXd weighted_event_counts;
		REAL weighted_max_reciprocal_sum_after_knot;
		REAL weighted_maxsquared_reciprocal_sum_after_knot;
};

class CompositeKernel : public Kernel {
	public:
		CompositeKernel(int num_event_types, REAL start_time, REAL end_time) : Kernel(num_event_types, start_time, end_time) {}

		~CompositeKernel() {
			for (Kernel* kernel : kernels) {
				delete kernel;
			}
		}

		std::pair<Eigen::VectorXd,Eigen::VectorXd> get_hessian_and_gradient() override {
			std::vector<Eigen::VectorXd> hessians(kernels.size());
			std::vector<Eigen::VectorXd> gradients(kernels.size());

			int total_params = get_param_count();
			Eigen::VectorXd hessian = Eigen::VectorXd::Constant(total_params,0.0/0.0);
			Eigen::VectorXd gradient = Eigen::VectorXd::Constant(total_params,0.0/0.0);

			int current_index = 0;
			for (Kernel* kernel : kernels) {
				int num_kernel_params = kernel->get_param_count();
				std::pair<Eigen::VectorXd, Eigen::VectorXd> hessian_and_gradient = kernel->get_hessian_and_gradient();
				hessian.block(current_index,0,num_kernel_params,1) = hessian_and_gradient.first;
				gradient.block(current_index,0,num_kernel_params,1) = hessian_and_gradient.second;
				current_index += num_kernel_params;
			}
			
			return {hessian, gradient};
		}

		virtual Eigen::VectorXd get_em_step() {
			Eigen::VectorXd step = Eigen::VectorXd::Constant(get_param_count(),0.0);

			int current_index = 0;
			for (Kernel* kernel : kernels) {
				step.block(current_index, 0, kernel->get_param_count(), 1) = kernel->get_em_step();
				current_index += kernel->get_param_count();
			}

			return step;
		}

		int get_param_count() {
			int param_count = 0;
			for (Kernel* kernel : kernels) {
				param_count += kernel->get_param_count();
			}
			return param_count;
		}


		Eigen::VectorXd get_params() {
			Eigen::VectorXd params(get_param_count());

			int current_index = 0;
			for (Kernel* kernel : kernels) {
				params.block(current_index, 0, kernel->get_param_count(), 1) = kernel->get_params();
				current_index += kernel->get_param_count();
			}

			return params;
		}

		void set_params(Eigen::VectorXd new_params) {
			int current_index = 0;
			for (Kernel* kernel : kernels) {
				kernel->set_params(new_params.block(current_index, 0, kernel->get_param_count(), 1));
				current_index += kernel->get_param_count();
			}
		}

		Eigen::VectorXd get_intensities() {
			Eigen::VectorXd intensities = Eigen::VectorXd::Constant(num_event_types, 0);
			for (Kernel* kernel : kernels) {
				intensities += kernel->get_intensities();
			}
			return intensities;
		}

		REAL progress_time(REAL timediff) override {
			REAL residual = 0;
			for (Kernel* kernel : kernels) {
				residual += kernel->progress_time(timediff);
			}
			current_time += timediff;
			return residual;
		}

		virtual std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			REAL residual = 0;
			REAL intensity = get_intensities()(observation.event_type);
			for (Kernel* kernel : kernels) {;
				REAL component_intensity = kernel->get_intensities()(observation.event_type) ;
				REAL kernel_weight = component_intensity / intensity;
				//std::cout << kernel->as_string().substr(0,10) << " weight " << kernel_weight << " intensity " << component_intensity << std::endl;
				std::pair<REAL,REAL> pair = kernel->update(observation, kernel_weight);
				residual += pair.first;
			}
			current_time = observation.time;
			if (residual<0) {
				std::cout << "residual and intensity: " << residual << ", " << intensity << std::endl;
				std::cout << as_string() << std::endl;
				reset_event_type(observation.event_type);
			}
			return {residual, intensity};
		}

		REAL get_intensity_upper_bound() {
			REAL intensity_upper_bound = 0;
			for (Kernel* kernel : kernels) {
				intensity_upper_bound += kernel->get_intensity_upper_bound();
			}
			return intensity_upper_bound;
		}

		void reset() override {
			current_time = start_time;
			for (Kernel* kernel : kernels) {
				kernel->reset();
			}
		};

		void reset_event_type(int event_type) {
			for (Kernel* kernel : kernels) {
				kernel->reset_event_type(event_type);
			}
		}

		void add_kernel(Kernel* kernel) {
			kernels.push_back(kernel);
		}

		std::string as_string() {
			std::stringstream ss;
			ss << "CompositeKernel(";
			for (Kernel* kernel : kernels) {
				ss << "\n|\t";
				ss << kernel->as_string();
			}
			ss << "\n)";
			return ss.str();
		}

		void perturb(REAL amount) {
			for (Kernel* kernel : kernels) {
				kernel->perturb(amount);
			}
		}

		std::vector<Kernel*> kernels{};

};

/*
class InstantaneousExcitationKernel : public Kernel {
	public:
		InstantaneousExcitationKernel(int num_event_types, REAL start_time, REAL end_time) : Kernel(num_event_types,start_time,end_time) {
			alpha = Eigen::MatrixXd::Random(num_event_types,num_event_types).array() + 2.0;
			reset();
		}

		void reset() {
			lambda = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
		}

		std::pair<Eigen::VectorXd,Eigen::VectorXd> get_hessian_and_gradient() override {
		
		}

		int get_param_count() override {
			return alpha.size();
		}

		Eigen::VectorXd get_intensities() override {
			return lambda.colwise().sum();
		}

		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			progress_time(observation.time - current_time);
			if (weight == 0) {
				{0,0};
			}
			lambda += 
		}

		REAL progress_time(REAL timediff) override {
			if (timediff>0) {
				lambda.setZero();
			}
			current_time += timediff;
			return 0;
		}

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "InstantaneousExcitationkernel(" << state_cause << "," << state_effect << "," << power << "," << negative <<  ")";
			return ss.str();
		}

		Eigen::MatrixXd alpha,lambda;
}
*/

class InstantaneousExcitationKernel: public Kernel {
	public:
		InstantaneousExcitationKernel(int num_event_types, REAL start_time, REAL end_time, int num_state_variables, bool state_cause=false, bool state_effect=false, bool power=false, bool negative=false) : Kernel(num_event_types, start_time, end_time), num_state_variables(num_state_variables), state_cause(state_cause), state_effect(state_effect), power(power), negative(negative) {
			alpha = 1e-4*(Eigen::MatrixXd::Random(num_event_types, num_event_types).array() + 2.0);
			if (negative) {
				alpha *= 1e-2;
			}
			a.clear();
			a.resize(num_event_types);
			b.clear();
			b.resize(num_event_types);
			for (int i = 0; i < num_event_types; i++) {
				a[i] = {};
				a[i].resize(num_event_types);
				b[i] = {};
				b[i].resize(num_event_types);
				for (int j = 0; j < num_event_types; j++) {
					a[i][j] = 1e-6*Eigen::VectorXd::Random(num_state_variables,1) * (state_cause ? 1 : 0);
					b[i][j] = 1e-6*Eigen::VectorXd::Random(num_state_variables,1) * (state_effect ? 1 : 0);
				}
			}
			k = 1;
			reset();


		}

		std::pair<Eigen::VectorXd,Eigen::VectorXd> get_hessian_and_gradient() override {
			Eigen::VectorXd hessian(get_param_count());
			Eigen::VectorXd gradient(get_param_count());

			gradient.segment(0,alpha.size()) = grad_alpha.reshaped();
			hessian.segment(0,alpha.size()) = hess_alpha.reshaped();

			int index = alpha.size();
			if (power) {
				gradient.array()[alpha.size()] = grad_k;
				hessian.array()[alpha.size()] = hess_k;
				index++;
			}
			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					if (state_cause) {
						gradient.segment(index,num_state_variables) = grad_a[i][j];
						hessian.segment(index,num_state_variables) = hess_a[i][j];
						index += num_state_variables;
					}
					if (state_effect) {
						gradient.segment(index,num_state_variables) = grad_b[i][j];
						hessian.segment(index,num_state_variables) = hess_b[i][j];
						index += num_state_variables;
					}
				}
			}

			if (index != get_param_count()) {
				std::cout << "index" << index << std::endl;
				std::cout << "get_param_count()" << get_param_count() << std::endl;
				assert(false);
			}

			return {hessian, gradient};
		}

		int get_param_count() {
			return alpha.size()+(power ? 1 : 0)+(state_cause ? num_event_types*num_event_types*num_state_variables : 0)+(state_effect ? num_event_types*num_event_types*num_state_variables : 0);
		}

		Eigen::VectorXd get_params() {
			Eigen::VectorXd params(get_param_count());

			params.segment(0,alpha.size()) = alpha.reshaped();

			int index = alpha.size();
			if (power) {
				params.array()[alpha.size()] = k;
				index++;
			}
			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					if (state_cause) {
						params.segment(index,num_state_variables) = a[i][j];
						index += num_state_variables;
					}
					if (state_effect) {
						params.segment(index,num_state_variables) = b[i][j];
						index += num_state_variables;
					}
				}
			}

			assert (index == get_param_count());

			return params;
		}

		void set_params(Eigen::VectorXd new_params) {
			alpha = Eigen::MatrixXd::Map(new_params.segment(0,alpha.size()).data(),alpha.rows(),alpha.cols());

			int index = alpha.size();
			if (power) {
				k = new_params.array()[alpha.size()];
				index++;
			}
			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					if (state_cause) {
						a[i][j] = new_params.segment(index,num_state_variables);
						index += num_state_variables;
					}
					if (state_effect) {
						b[i][j] = new_params.segment(index,num_state_variables);
						index += num_state_variables;
					}
				}
			}

			assert (index == get_param_count());
		}

		Eigen::VectorXd get_intensities() {
			Eigen::VectorXd intensities = lambda_matrix.colwise().sum().eval();
			intensities = intensities.unaryExpr([this](double x) {return std::pow(x,this->k);});
			intensities = (negative ? -1.0 : 1.0)*intensities;
			return intensities;
		}

		REAL get_intensity_upper_bound() {
			return get_intensities().cwiseMax(0).sum();
		}

		void reset() {
			current_time = start_time;
			lambda_matrix = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			grad_alpha = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			hess_alpha = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			grad_k = 0;
			hess_k = 0;

			marks = Eigen::VectorXd::Constant(num_state_variables,0.0);

			for (std::vector<std::vector<Eigen::VectorXd>> *vec : {&grad_a, &hess_a, &grad_b, &hess_b, &dlambda_da, &dsquaredlambda_dasquared}) {
				vec->clear();
				vec->resize(num_event_types);
				for (int i = 0; i < num_event_types; i++) {
					(*vec)[i] = {};
					(*vec)[i].resize(num_event_types);
					for (int j = 0; j < num_event_types; j++) {
						(*vec)[i][j] = Eigen::VectorXd::Constant(num_state_variables,0);
					}
				}
			}
		};

		void reset_event_type(int event_type) {
			alpha.block(0,num_event_types,event_type,1) = Eigen::VectorXd::Random(num_event_types).array() + 1.0;
		}


		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			weight = weight * (negative ? -1 : 1);

			REAL timediff = observation.time - current_time;
			REAL residual = progress_time(timediff);

			Eigen::VectorXd marks_change = observation.marks - marks;
			marks = observation.marks;

			REAL total_lambda = lambda_matrix.colwise().sum()(observation.event_type);
			if (total_lambda != 0) {
				grad_k += lambda_matrix.colwise().sum().array().log()(observation.event_type)*weight;

				grad_alpha.block(0,observation.event_type,num_event_types,1) += weight*k*lambda_matrix.cwiseQuotient(alpha).block(0,observation.event_type,num_event_types,1) / total_lambda;

				hess_alpha.block(0,observation.event_type,num_event_types,1) += weight*k*lambda_matrix.cwiseQuotient(alpha.cwiseProduct(alpha)).block(0,observation.event_type,num_event_types,1) / total_lambda;
				hess_alpha.block(0,observation.event_type,num_event_types,1) -= weight*k*(lambda_matrix.cwiseQuotient(alpha).block(0,observation.event_type,num_event_types,1) / total_lambda).unaryExpr([](double x){return x*x;});

				for (int i = 0; i < num_event_types; i++) {
					Eigen::VectorXd dlambda_db = marks*lambda_matrix(i,observation.event_type);
					Eigen::VectorXd dsquaredlambda_dbsquared = marks.cwiseProduct(marks)*lambda_matrix(i,observation.event_type);
					grad_b[i][observation.event_type] += weight*k*dlambda_db / total_lambda;
					hess_b[i][observation.event_type] += weight*k*(dsquaredlambda_dbsquared / total_lambda - (dlambda_db / total_lambda).unaryExpr([](double x){return x*x;}));
				}
			}

			if (total_lambda != 0) {
				for (int i = 0; i < num_event_types; i++) {
					grad_a[i][observation.event_type] += weight*k*dlambda_da[i][observation.event_type] / total_lambda;
					hess_a[i][observation.event_type] += weight*k*(dsquaredlambda_dasquared[i][observation.event_type] / total_lambda - (dlambda_da[i][observation.event_type] / total_lambda).unaryExpr([](double x){return x*x;}));
				}
			}
			for (int j = 0; j < num_event_types; j++) {
				dlambda_da[observation.event_type][j] = dlambda_da[observation.event_type][j] * std::exp(marks_change.dot(b[observation.event_type][j]));
				dlambda_da[observation.event_type][j] += alpha(observation.event_type,j) * std::exp(marks.dot(b[observation.event_type][j]+a[observation.event_type][j])) * marks;
				dsquaredlambda_dasquared[observation.event_type][j] = dsquaredlambda_dasquared[observation.event_type][j] * std::exp(marks_change.dot(b[observation.event_type][j]));
				dsquaredlambda_dasquared[observation.event_type][j] += alpha(observation.event_type,j) * std::exp(marks.dot(b[observation.event_type][j]+a[observation.event_type][j])) * marks.cwiseProduct(marks);
			}

			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					lambda_matrix(i,j) *= std::exp(marks_change.dot(b[i][j]));
				}
			}
			for (int j = 0; j < num_event_types; j++) {
				lambda_matrix(observation.event_type,j) += alpha(observation.event_type,j) * std::exp(marks.dot(b[observation.event_type][j]+a[observation.event_type][j]));
			}


			REAL intensity = std::pow(lambda_matrix.colwise().sum()(observation.event_type),k);
			return {residual, intensity};
		}


		REAL progress_time(REAL timediff) override {
			REAL residual = 0;

			current_time += timediff;

			if (timediff>0) {
				lambda_matrix.setZero();
			}

			return residual;
		}

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "InstantaneousExcitationKernel(" << state_cause << "," << state_effect << "," << power << "," << negative <<  ")";
			return ss.str();
		}
	private:
		Eigen::MatrixXd alpha,lambda_matrix,grad_alpha,hess_alpha;
		std::vector<std::vector<Eigen::VectorXd>> a {}, grad_a {}, hess_a{}, b {}, grad_b {}, hess_b {}, dlambda_da {}, dsquaredlambda_dasquared {};
		Eigen::VectorXd marks;
		bool state_cause, state_effect, power, negative;
		int num_state_variables;
		REAL k = 1;
		REAL grad_k, hess_k;
};

class ContinuousStateHawkesKernel : public Kernel {
	public:
		ContinuousStateHawkesKernel(int num_event_types, REAL start_time, REAL end_time, int num_state_variables, bool state_cause=false, bool state_effect=false, bool power=false, bool negative=false) : Kernel(num_event_types, start_time, end_time), num_state_variables(num_state_variables), state_cause(state_cause), state_effect(state_effect), power(power), negative(negative) {
			alpha = 1e-4*(Eigen::MatrixXd::Random(num_event_types, num_event_types).array() + 2.0);
			beta = 2*alpha.cwiseMax(0).sum();
			if (negative) {
				alpha *= 1e-2;
			}
			a.clear();
			a.resize(num_event_types);
			b.clear();
			b.resize(num_event_types);
			for (int i = 0; i < num_event_types; i++) {
				a[i] = {};
				a[i].resize(num_event_types);
				b[i] = {};
				b[i].resize(num_event_types);
				for (int j = 0; j < num_event_types; j++) {
					a[i][j] = 1e-6*Eigen::VectorXd::Random(num_state_variables,1) * (state_cause ? 1 : 0);
					b[i][j] = 1e-6*Eigen::VectorXd::Random(num_state_variables,1) * (state_effect ? 1 : 0);
				}
			}
			k = 1;
			reset();


		}

		std::pair<Eigen::VectorXd,Eigen::VectorXd> get_hessian_and_gradient() override {
			Eigen::VectorXd hessian(get_param_count());
			Eigen::VectorXd gradient(get_param_count());

			/*
			std::cout << "alpha " << alpha.norm() << std::endl;
			std::cout << "grad_alpha " << grad_alpha.norm() << std::endl;
			std::cout << "beta " << beta << std::endl;
			std::cout << "grad_beta " << grad_beta << std::endl;
			std::cout << "k " << k << std::endl;
			std::cout << "grad_k " << grad_k << std::endl;
			std::cout << "a[0][0] " << a[0][0].norm() << std::endl;
			std::cout << "grad_a[0][0] " << grad_a[0][0].norm() << std::endl;
			std::cout << "b[0][0] " << b[0][0].norm() << std::endl;
			std::cout << "grad_b[0][0] " << grad_b[0][0].norm() << std::endl;
			*/

			for (REAL x : {alpha.norm(), grad_alpha.norm(), beta, grad_beta}) {
				assert(std::isfinite(x));
			}

			gradient.segment(0,alpha.size()) = grad_alpha.reshaped();
			hessian.segment(0,alpha.size()) = hess_alpha.reshaped();

			gradient.array()[alpha.size()] = grad_beta;
			hessian.array()[alpha.size()] = hess_beta;

			int index = alpha.size() + 1;
			if (power) {
				gradient.array()[alpha.size()+1] = grad_k;
				hessian.array()[alpha.size()+1] = hess_k;
				index++;
			}
			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					if (state_cause) {
						gradient.segment(index,num_state_variables) = grad_a[i][j];
						hessian.segment(index,num_state_variables) = hess_a[i][j];
						index += num_state_variables;
					}
					if (state_effect) {
						gradient.segment(index,num_state_variables) = grad_b[i][j];
						hessian.segment(index,num_state_variables) = hess_b[i][j];
						index += num_state_variables;
					}
				}
			}

			assert (index == get_param_count());

			return {hessian, gradient};
		}

		int get_param_count() {
			return alpha.size()+1+(power ? 1 : 0)+(state_cause ? num_event_types*num_event_types*num_state_variables : 0)+(state_effect ? num_event_types*num_event_types*num_state_variables : 0);
		}

		Eigen::VectorXd get_params() {
			Eigen::VectorXd params(get_param_count());

			params.segment(0,alpha.size()) = alpha.reshaped();

			params.array()[alpha.size()] = beta;

			int index = alpha.size() + 1;
			if (power) {
				params.array()[alpha.size()+1] = k;
				index++;
			}
			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					if (state_cause) {
						params.segment(index,num_state_variables) = a[i][j];
						index += num_state_variables;
					}
					if (state_effect) {
						params.segment(index,num_state_variables) = b[i][j];
						index += num_state_variables;
					}
				}
			}

			assert (index == get_param_count());

			return params;
		}

		void set_params(Eigen::VectorXd new_params) {
			alpha = Eigen::MatrixXd::Map(new_params.segment(0,alpha.size()).data(),alpha.rows(),alpha.cols());

			beta = new_params.array()[alpha.size()];

			int index = alpha.size() + 1;
			if (power) {
				k = new_params.array()[alpha.size()+1];
				index++;
			}
			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					if (state_cause) {
						a[i][j] = new_params.segment(index,num_state_variables);
						index += num_state_variables;
					}
					if (state_effect) {
						b[i][j] = new_params.segment(index,num_state_variables);
						index += num_state_variables;
					}
				}
			}

			assert (index == get_param_count());
		}

		Eigen::VectorXd get_intensities() {
			Eigen::VectorXd intensities = lambda_matrix.colwise().sum().eval();
			intensities = intensities.unaryExpr([this](double x) {return std::pow(x,this->k);});
			intensities = (negative ? -1.0 : 1.0)*intensities;
			return intensities;
		}

		REAL get_intensity_upper_bound() {
			return get_intensities().cwiseMax(0).sum();
		}

		void reset() {
			current_time = start_time;
			lambda_matrix = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			dlambda_matrix_dbeta = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			dsquaredlambda_matrix_dbetasquared = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			grad_alpha = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			hess_alpha = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			grad_k = 0;
			hess_k = 0;
			grad_beta = 0;
			hess_beta = 0;

			marks = Eigen::VectorXd::Constant(num_state_variables,0.0);

			for (std::vector<std::vector<Eigen::VectorXd>> *vec : {&grad_a, &hess_a, &grad_b, &hess_b, &dlambda_da, &dsquaredlambda_dasquared}) {
				vec->clear();
				vec->resize(num_event_types);
				for (int i = 0; i < num_event_types; i++) {
					(*vec)[i] = {};
					(*vec)[i].resize(num_event_types);
					for (int j = 0; j < num_event_types; j++) {
						(*vec)[i][j] = Eigen::VectorXd::Constant(num_state_variables,0);
					}
				}
			}
		};

		void reset_event_type(int event_type) {
			alpha.block(0,num_event_types,event_type,1) = Eigen::VectorXd::Random(num_event_types).array() + 1.0;
		}

		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			weight = weight * (negative ? -1 : 1);

			REAL timediff = observation.time - current_time;
			REAL residual = progress_time(timediff);

			Eigen::VectorXd marks_change = observation.marks - marks;
			marks = observation.marks;

			REAL total_lambda = lambda_matrix.colwise().sum()(observation.event_type);
			if (total_lambda != 0) {
				grad_k += lambda_matrix.colwise().sum().array().log()(observation.event_type)*weight;

				grad_beta += (k*dlambda_matrix_dbeta.colwise().sum()(observation.event_type)/total_lambda)*weight;

				//hess_beta += k*(dsquaredlambda_matrix_dbetasquared.colwise().sum().cwiseQuotient(lambda_matrix.colwise().sum()) - (dlambda_matrix_dbeta.colwise().sum()).cwiseQuotient(lambda_matrix.colwise().sum()).unaryExpr([](double x){return x*x;}))(observation.event_type)*weight;
				hess_beta += k*(dsquaredlambda_matrix_dbetasquared.colwise().sum()(observation.event_type)/total_lambda - std::pow(dlambda_matrix_dbeta.colwise().sum()(observation.event_type)/total_lambda,2))*weight;

				grad_alpha.block(0,observation.event_type,num_event_types,1) += weight*k*lambda_matrix.cwiseQuotient(alpha).block(0,observation.event_type,num_event_types,1) / total_lambda;

				hess_alpha.block(0,observation.event_type,num_event_types,1) += weight*k*lambda_matrix.cwiseQuotient(alpha.cwiseProduct(alpha)).block(0,observation.event_type,num_event_types,1) / total_lambda;
				hess_alpha.block(0,observation.event_type,num_event_types,1) -= weight*k*(lambda_matrix.cwiseQuotient(alpha).block(0,observation.event_type,num_event_types,1) / total_lambda).unaryExpr([](double x){return x*x;});

				for (int i = 0; i < num_event_types; i++) {
					Eigen::VectorXd dlambda_db = marks*lambda_matrix(i,observation.event_type);
					Eigen::VectorXd dsquaredlambda_dbsquared = marks.cwiseProduct(marks)*lambda_matrix(i,observation.event_type);
					grad_b[i][observation.event_type] += weight*k*dlambda_db / total_lambda;
					hess_b[i][observation.event_type] += weight*k*(dsquaredlambda_dbsquared / total_lambda - (dlambda_db / total_lambda).unaryExpr([](double x){return x*x;}));
				}
			}

			if (total_lambda != 0) {
				for (int i = 0; i < num_event_types; i++) {
					grad_a[i][observation.event_type] += weight*k*dlambda_da[i][observation.event_type] / total_lambda;
					hess_a[i][observation.event_type] += weight*k*(dsquaredlambda_dasquared[i][observation.event_type] / total_lambda - (dlambda_da[i][observation.event_type] / total_lambda).unaryExpr([](double x){return x*x;}));
				}
			}
			for (int j = 0; j < num_event_types; j++) {
				dlambda_da[observation.event_type][j] = dlambda_da[observation.event_type][j] * std::exp(marks_change.dot(b[observation.event_type][j]));
				dlambda_da[observation.event_type][j] += alpha(observation.event_type,j) * std::exp(marks.dot(b[observation.event_type][j]+a[observation.event_type][j])) * marks;
				dsquaredlambda_dasquared[observation.event_type][j] = dsquaredlambda_dasquared[observation.event_type][j] * std::exp(marks_change.dot(b[observation.event_type][j]));
				dsquaredlambda_dasquared[observation.event_type][j] += alpha(observation.event_type,j) * std::exp(marks.dot(b[observation.event_type][j]+a[observation.event_type][j])) * marks.cwiseProduct(marks);
			}

			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					lambda_matrix(i,j) *= std::exp(marks_change.dot(b[i][j]));
				}
			}
			for (int j = 0; j < num_event_types; j++) {
				lambda_matrix(observation.event_type,j) += alpha(observation.event_type,j) * std::exp(marks.dot(b[observation.event_type][j]+a[observation.event_type][j]));
			}


			REAL intensity = std::pow(lambda_matrix.colwise().sum()(observation.event_type),k);
			return {residual, intensity};
		}


		REAL progress_time(REAL timediff) override {

			REAL integral_exp_minus_beta_k_t = (1-std::exp(-beta*k*timediff))/(beta*k);
			REAL integral_t_exp_minus_beta_k_t = (std::exp(-beta*k*timediff) * (-beta*k*timediff -1) + 1) / (beta*beta*k*k);
			REAL integral_tsquared_exp_minus_beta_k_t = (std::exp(-beta*k*timediff)*(-beta*k*timediff*(beta*k*timediff+2)-2) + 2) / (beta*beta*beta * k*k*k);

			grad_k -= lambda_matrix.colwise().sum().unaryExpr([this,integral_exp_minus_beta_k_t,integral_t_exp_minus_beta_k_t](double x) {
						return std::pow(x,this->k) * ((x==0 ? 0 : std::log(x)) * integral_exp_minus_beta_k_t - this->beta * integral_t_exp_minus_beta_k_t);
					}).sum();
			hess_k -= lambda_matrix.colwise().sum().unaryExpr([this,integral_exp_minus_beta_k_t,integral_t_exp_minus_beta_k_t,integral_tsquared_exp_minus_beta_k_t](double x){
						return std::pow(x,this->k) * ((x==0 ? 0 : std::log(x)*std::log(x))*integral_exp_minus_beta_k_t - 2*(x==0 ? 0 : std::log(x))*this->beta*integral_t_exp_minus_beta_k_t + beta*beta*integral_tsquared_exp_minus_beta_k_t);
					}).sum();

			{
				Eigen::MatrixXd change_grad_beta = lambda_matrix.colwise().sum();
				change_grad_beta = change_grad_beta.unaryExpr([this](double x) { return x==0 ? 0 : std::pow(x,this->k-1); });
				change_grad_beta = change_grad_beta.cwiseProduct(dlambda_matrix_dbeta.colwise().sum()*integral_exp_minus_beta_k_t - lambda_matrix.colwise().sum()*integral_t_exp_minus_beta_k_t);
				grad_beta -= k*change_grad_beta.sum();
			}

			{
				Eigen::MatrixXd change_hess_beta =	lambda_matrix.colwise().sum();
				change_hess_beta =	change_hess_beta.unaryExpr([this](double x){
								return x==0 ? 0 : std::pow(x,this->k-2);
							});
				change_hess_beta = change_hess_beta.cwiseProduct(dlambda_matrix_dbeta.colwise().sum().unaryExpr([](double x){ return x*x; }));
				hess_beta -= k*(k-1)*change_hess_beta.sum();
				change_hess_beta =	lambda_matrix.colwise().sum();
				change_hess_beta = change_hess_beta.unaryExpr([this](double x){
								return x==0 ? 0 : std::pow(x,this->k-1);
							});
				change_hess_beta = change_hess_beta.cwiseProduct(dsquaredlambda_matrix_dbetasquared.colwise().sum()*integral_exp_minus_beta_k_t - 2*dlambda_matrix_dbeta.colwise().sum()*integral_t_exp_minus_beta_k_t + lambda_matrix.colwise().sum()*integral_tsquared_exp_minus_beta_k_t);
				hess_beta -=	k*change_hess_beta.sum();
			}

			{
				Eigen::MatrixXd change_grad_alpha = lambda_matrix;
				change_grad_alpha = change_grad_alpha.cwiseQuotient(alpha).eval();
				Eigen::VectorXd vector = lambda_matrix.colwise().sum().eval();
				vector = vector.unaryExpr([this](double x){ return x==0 ? 0 : std::pow(x,this->k-1); });
				change_grad_alpha.array().rowwise() *= vector.transpose().array();
				change_grad_alpha *= k*integral_exp_minus_beta_k_t;
				grad_alpha -= change_grad_alpha;
			}
			{

				Eigen::MatrixXd change_hess_alpha = lambda_matrix.cwiseQuotient(alpha).unaryExpr([](double x){ return x*x; });
				Eigen::VectorXd vector = lambda_matrix.colwise().sum();
				vector = vector.unaryExpr([this](double x) { return x==0 ? 0 : std::pow(x,this->k-2); });
				change_hess_alpha.array().rowwise() *= vector.transpose().array();
				hess_alpha -= k*(k-1)*change_hess_alpha*integral_exp_minus_beta_k_t;
			}
			{
				Eigen::MatrixXd change_hess_alpha = lambda_matrix.cwiseQuotient(alpha.cwiseProduct(alpha));
				Eigen::VectorXd vector = lambda_matrix.colwise().sum().eval();
				vector = vector.unaryExpr([this](double x) { return x==0 ? 0 : std::pow(x,this->k-1); });
				change_hess_alpha.array().rowwise() *= vector.transpose().array();
				hess_alpha -= k*integral_exp_minus_beta_k_t * change_hess_alpha;
			}

			dsquaredlambda_matrix_dbetasquared += timediff*timediff*lambda_matrix - 2*timediff*dlambda_matrix_dbeta;
			dsquaredlambda_matrix_dbetasquared *= std::exp(-beta*timediff);
			dlambda_matrix_dbeta += -timediff*lambda_matrix;
			dlambda_matrix_dbeta *= std::exp(-beta*timediff);

			for (int i = 0; i < num_event_types; i++) {
				for (int j = 0; j < num_event_types; j++) {
					REAL total_lambda = lambda_matrix.colwise().sum()(j);
					if (total_lambda != 0) {
						grad_a[i][j] -= k * std::pow(total_lambda,k-1) * dlambda_da[i][j] * integral_exp_minus_beta_k_t;
						hess_a[i][j] -= k * (k-1) * std::pow(total_lambda,k-2) * dlambda_da[i][j].cwiseProduct(dlambda_da[i][j]) * integral_exp_minus_beta_k_t;
						hess_a[i][j] -= k * std::pow(total_lambda,k-1) * dsquaredlambda_dasquared[i][j] * integral_exp_minus_beta_k_t;

						Eigen::VectorXd dlambda_db = marks*lambda_matrix(i,j);
						Eigen::VectorXd dsquaredlambda_dbsquared = marks.cwiseProduct(marks)*lambda_matrix(i,j);
						grad_b[i][j] -= k * std::pow(total_lambda,k-1) * marks * lambda_matrix(i,j) * integral_exp_minus_beta_k_t;
						hess_b[i][j] -= k * (k-1) * std::pow(total_lambda,k-2) * dlambda_db.cwiseProduct(dlambda_db) * integral_exp_minus_beta_k_t;
						hess_b[i][j] -= k * std::pow(total_lambda,k-1) * dsquaredlambda_dbsquared * integral_exp_minus_beta_k_t;
					}

					dlambda_da[i][j] *= std::exp(-beta*timediff);
					dsquaredlambda_dasquared[i][j] *= std::exp(-beta*timediff);
				}
			}

			REAL residual = lambda_matrix.colwise().sum().unaryExpr([this](double x){return std::pow(x,this->k);}).sum() * integral_exp_minus_beta_k_t;

			current_time += timediff;

			lambda_matrix *= std::exp(-beta*timediff);

			return residual;
		}

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "ContinuousStateHawkesKernel(" << state_cause << "," << state_effect << "," << k << "," << negative << "," << alpha.norm() << "," << beta << ")";
			return ss.str();
		}
	private:
		Eigen::MatrixXd alpha,lambda_matrix,grad_alpha,hess_alpha,dlambda_matrix_dbeta,dsquaredlambda_matrix_dbetasquared;
		std::vector<std::vector<Eigen::VectorXd>> a {}, grad_a {}, hess_a{}, b {}, grad_b {}, hess_b {}, dlambda_da {}, dsquaredlambda_dasquared {};
		Eigen::VectorXd marks;
		bool state_cause, state_effect, power, negative;
		int num_state_variables;
		REAL k = 1;
		REAL beta;
		REAL grad_k, hess_k, grad_beta, hess_beta;
};

/*
class ExpHawkesKernel : Kernel {
	public:
		ExpHawkesKernel(int num_event_types) : Kernel(num_event_types) {
			alpha = Eigen::MatrixXd::Random(num_event_types, num_event_types) + 2.0;
			beta = 2*alpha;
		}

		std::pair<Eigen::MatrixXd,Eigen::VectorXd> get_hessian_and_gradient() {
		}

		Eigen::VectorXd get_params() {
		}

		void set_params(Eigen::VectorXd new_params) {
		}

		Eigen::VectorXd get_intensities() {
			return intensity_matrix.rowwise().sum();
		}

		void reset() {
			current_time = start_time;
			intensity_matrix = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			time_ema_matrix = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			alpha_gradient = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
			beta_gradient = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
		};

		REAL update(Event observation, REAL weight=1.0) {
			if (weight != 0) {
				REAL timediff = observation.time - current_time;
				weighted_event_counts.array()[observation.event_type] += weight;
				progress_time(timediff);

				intensity_matrix.array() *= (-beta*timediff).array().exp();
				intensity_matrix += alpha;

				time_ema_matrix.array() *= (-beta*timediff).array().exp();
				time_ema_matrix += alpha*observation.time;
			}

			return residual;
		}

		REAL get_intensity_upper_bound() {
			return std::max(0,get_intensity());
		}
	private:
		Eigen::MatrixXd alpha, beta, intensity_matrix, time_ema_matrix, alpha_gradient, beta_gradient, alpha_hessian_diag, beta_hessian_diag, hessian_crossterm;
}
*/


/*
 * State dependent kernel
 * Switching kernel
 */

/*
class ExpHawkesKernel {
	public:
		long double nu;
		long double alpha;
		long double beta;

		ExpHawkesKernel(float nu, float alpha, float beta) : nu(nu), alpha(alpha), beta(beta) {
			reset();
		};

		bool first;
		long double start_time;
		long double end_time;

		long double prev_time;
		long double endo_intensity = 0;

		long double total_intensity = 0;

		long double estimated_num_endo = 0;
		long double estimated_num_exo = 0;
		int num_events = 0;
		long double decaying_sum_ti = 0;
		long double beta_denominator = 0;

		void reset() {
			first = true;
			endo_intensity = 0;
			total_intensity = 0;

			estimated_num_endo = 0;
			estimated_num_exo = 0;
			num_events = 0;
			decaying_sum_ti = 0;
			beta_denominator = 0;
		}

		void update(long double time) {
			if (first) {
				start_time = time;
				prev_time = time;
				first = false;
			} else {
				long double decay = beta < 0 ? 1 : std::exp(-beta * (time - prev_time));
				endo_intensity *= decay;
				decaying_sum_ti *= decay;
			}
			endo_intensity += alpha;

			total_intensity = nu + endo_intensity;

			estimated_num_exo += nu / total_intensity;

			estimated_num_endo += endo_intensity / total_intensity;

			num_events++;

			decaying_sum_ti += alpha * time;

			beta_denominator += (time * endo_intensity - decaying_sum_ti) / (nu + endo_intensity);
		}

		bool em_step() {
			long double nu_old = nu;
			long double alpha_old = alpha;
			long double beta_old = beta;

			nu = estimated_num_exo / (end_time - start_time);
			beta = std::abs(estimated_num_endo / beta_denominator);
			alpha = std::abs(beta * estimated_num_endo / num_events);

			std::cerr << nu << ", " << alpha << ", " << beta << std::endl;

			reset();

			return !(std::abs(nu-nu_old) < 1e-4 & std::abs(alpha-alpha_old) < 1e-4 & std::abs(beta-beta_old) < 1e-4);
		}
};
*/

// Power hawkes
// State dependent
// Regime switching
// 
