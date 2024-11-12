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

		virtual std::pair<Eigen::MatrixXd,Eigen::VectorXd> get_hessian_and_gradient() = 0;

		virtual Eigen::VectorXd get_em_step() {
			auto [hess,grad] = get_hessian_and_gradient();
			return get_em_step(hess,grad);
		}

		virtual Eigen::VectorXd get_em_step(Eigen::MatrixXd hess, Eigen::MatrixXd grad) {
			return -hess.colPivHouseholderQr().solve(grad);
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

		virtual void progress_time(REAL timediff) {
			current_time += timediff;
		}

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

		virtual std::string as_string() = 0;

		virtual void perturb(REAL amount) {};

		REAL start_time, end_time, current_time;
		int num_event_types;

		//KS test for streaming data https://personal.denison.edu/~lalla/papers/ks-stream.pdf

};

class PoissonKernel : public Kernel {
	public:
		PoissonKernel(int num_event_types, REAL start_time, REAL end_time) : Kernel(num_event_types, start_time, end_time) {
			nu = Eigen::VectorXd::Random(num_event_types).array() + 1.0;
			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types,0.0);
		}

		std::pair<Eigen::MatrixXd,Eigen::VectorXd> get_hessian_and_gradient() {
			Eigen::VectorXd gradient = weighted_event_counts.cwiseProduct(nu.cwiseInverse());
			std::cout << "weighted event counts " << weighted_event_counts << std::endl;
			std::cout << "weighted event counts over duration " << weighted_event_counts/(end_time-start_time) << std::endl;
			gradient.array() -= end_time-start_time;

			Eigen::MatrixXd hessian = (-(weighted_event_counts.array().log()-2*nu.array().log()).exp()).matrix().asDiagonal();

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
		}

		Eigen::VectorXd get_intensities() {
			return nu;
		}

		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			REAL timediff = observation.time - current_time;
			if (timediff<0) {
				std::cout << "Warning: current_time=" << current_time << " but observed " << observation << std::endl;
			}
			weighted_event_counts.array()[observation.event_type] += weight;
			progress_time(timediff);
			return {nu.sum()*timediff, nu(observation.event_type)};
		}

		REAL get_intensity_upper_bound() {
			return get_intensity();
		}

		void reset() override {
			current_time = start_time;
			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types, 0.0);
		};

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "PoissonKernel(" << nu.format(format) << ")";
			return ss.str();
		}

	private:
		Eigen::VectorXd nu;
		Eigen::VectorXd weighted_event_counts;
};


class LinearSpline : public Kernel {
	public:
		LinearSpline(int num_event_types, REAL start_time, REAL end_time) : Kernel(num_event_types, start_time, end_time) {
			coef = Eigen::VectorXd::Random(num_event_types).array() + 1.0;

			std::uniform_real_distribution<REAL> unif_distribution(start_time,end_time);
			knot = unif_distribution(generator);

			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types,0.0);
		}

		std::pair<Eigen::MatrixXd,Eigen::VectorXd> get_hessian_and_gradient() {
			Eigen::VectorXd gradient_coef = weighted_event_counts.cwiseProduct(coef.cwiseInverse()).array() - 0.5 * (end_time - knot)*(end_time - knot);
			REAL gradient_knot = - weighted_max_reciprocal_sum_after_knot + (end_time - knot) * coef.sum();
			Eigen::VectorXd hessian_coef_coef = - weighted_event_counts.cwiseProduct(coef.cwiseProduct(coef).cwiseInverse());
			Eigen::VectorXd hessian_coef_knot = Eigen::VectorXd::Constant(num_event_types,end_time - knot);
			REAL hessian_knot_knot = weighted_maxsquared_reciprocal_sum_after_knot - coef.sum();

			Eigen::MatrixXd hessian(num_event_types + 1, num_event_types + 1);
			hessian.block(0,0,num_event_types,num_event_types) = hessian_coef_coef.asDiagonal();
			hessian.block(num_event_types,0,1,num_event_types) = hessian_coef_knot.transpose();
			hessian.block(0,num_event_types,num_event_types,1) = hessian_coef_knot;
			hessian(num_event_types, num_event_types) = hessian_knot_knot;

			Eigen::VectorXd gradient(num_event_types + 1);
			gradient.block(0,0,num_event_types,1) = gradient_coef;
			gradient(num_event_types) = gradient_knot;

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
				knot = start_time;
			}
		}

		Eigen::VectorXd get_intensities() {
			return coef * std::max(current_time - knot, 0.0l);
		}

		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			REAL timediff = observation.time - current_time;
			REAL residual = [](std::pair<REAL,REAL> x){return 0.5*(x.second*x.second-x.first*x.first);}({std::max(coef.sum() * (current_time-knot), 0.0l), std::max(coef.sum() * (observation.time-knot), 0.0l)});
			progress_time(timediff);
			weighted_event_counts.array()[observation.event_type] += weight;
			//std::cout << current_time << "," << knot << std::endl;
			if (current_time > knot) {
				//std::cout << "current_time>knot " << this << std::endl;
				REAL max = std::max(current_time - knot,0.0l);
				weighted_max_reciprocal_sum_after_knot += 1/max;
				weighted_maxsquared_reciprocal_sum_after_knot += 1/(max*max);
			}
			return {residual, get_intensities()(observation.event_type)};
		}

		REAL get_intensity_upper_bound() {
			return std::max(coef.sum() * (end_time - knot), 0.0l);
			//This is quite inefficient for simulation purposes. Maybe write a custom simulation method for this subclass.
		}

		void reset() override {
			current_time = start_time;
			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types, 0.0);
			weighted_max_reciprocal_sum_after_knot = 0;
			weighted_maxsquared_reciprocal_sum_after_knot = 0;
		};

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "LinearSpline(" << knot << ";" << coef.format(format) << ")";
			return ss.str();
		}

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

		std::pair<Eigen::MatrixXd,Eigen::VectorXd> get_hessian_and_gradient() {
			std::vector<Eigen::MatrixXd> hessians(kernels.size());
			std::vector<Eigen::VectorXd> gradients(kernels.size());

			int total_params = get_param_count();
			Eigen::MatrixXd hessian(total_params, total_params);
			Eigen::VectorXd gradient(total_params);

			int current_index = 0;
			for (Kernel* kernel : kernels) {
				int num_kernel_params = kernel->get_param_count();
				std::pair<Eigen::MatrixXd, Eigen::VectorXd> hessian_and_gradient = kernel->get_hessian_and_gradient();
				hessian.block(current_index,current_index,num_kernel_params,num_kernel_params) = hessian_and_gradient.first;
				gradient.block(current_index,0,num_kernel_params,1) = hessian_and_gradient.second;
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

		void progress_time(REAL timediff) {
			for (Kernel* kernel : kernels) {
				kernel->progress_time(timediff);
			}
		}

		virtual std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			REAL residual = 0;
			REAL intensity = 0;
			for (Kernel* kernel : kernels) {
				REAL component_intensity = kernel->get_intensities()(observation.event_type) ;
				REAL kernel_weight = component_intensity / get_intensities()(observation.event_type);
				std::pair<REAL,REAL> pair = kernel->update(observation, kernel_weight);
				if (pair.first<0) {
					std::cout << kernel->as_string().substr(0,10) << " | residual and intensity: " << pair.first << ", " << pair.second << std::endl;
				}
				residual += pair.first;
				intensity += pair.second;
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

class ContinuousStateHawkesKernel : public Kernel {
	public:
		ContinuousStateHawkesKernel(int num_event_types, REAL start_time, REAL end_time, int num_state_variables, bool state_cause=0, bool state_effect=0) : Kernel(num_event_types, start_time, end_time), num_state_variables(num_state_variables), state_cause(state_cause), state_effect(state_effect) {
			alpha = Eigen::MatrixXd::Random(num_event_types, num_event_types).array() + 2.0;
			beta = 2*alpha;
			if (state_cause) {
				a = Eigen::MatrixXd::Random(num_event_types,num_event_types);
			}
			if (state_effect) {
				b = Eigen::MatrixXd::Random(num_event_types,num_event_types);
			}

			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types,0.0);
		}

		std::pair<Eigen::MatrixXd,Eigen::VectorXd> get_hessian_and_gradient() {
			throw std::logic_error("Not implemented");
			Eigen::VectorXd gradient{};
			Eigen::MatrixXd hessian{};

			return {hessian, gradient};
		}

		int get_param_count() {
			return alpha.size()+beta.size();
		}

		Eigen::VectorXd get_params() {
			params = Eigen::VectorXd(get_param_count());
			params.segment(0,alpha.size()) = alpha.reshaped();
			params.segment(alpha.size(),beta.size()) = beta.reshaped();
			return params;
		}

		void set_params(Eigen::VectorXd new_params) {
			alpha = new_params.segment(0,num_event_types*num_event_types).reshaped(num_event_types,num_event_types);
			beta = new_params.segment(num_event_types*num_event_types,num_event_types*num_event_types).reshaped(num_event_types,num_event_types);
		}

		Eigen::VectorXd get_intensities() {
			return endo_intensity.colwise().sum();
		}

		REAL get_intensity_upper_bound() {
			return endo_intensity.cwiseMax(0).sum();
		}

		std::pair<REAL,REAL> update(Event observation, REAL weight=1.0) override {
			REAL timediff = observation.time - current_time;
			progress_time(timediff);
			weighted_event_counts.array()[observation.event_type] += weight;

		}

		void reset() {
			current_time = start_time;
			weighted_event_counts = Eigen::VectorXd::Constant(num_event_types, 0.0);
			endo_intensity = Eigen::MatrixXd::Constant(num_event_types, num_event_types, 0.0);
		};

		void progress_time(REAL timediff) {
			current_time += timediff;
			
			endo_intensity += alpha;
			Eigen::MatrixXd decay = (-beta*timediff).array().exp();
			endo_intensity.array() *= decay.array();
		}

		std::string as_string() {
			std::stringstream ss;
			Eigen::IOFormat format = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, " ", ",\t");
			ss << "ExpHawkesKernel(" << nu.format(format) << ")";
			return ss.str();
		}
	private:
		Eigen::MatrixXd alpha,beta,endo_intensity,time_ema;
		std::vector<Eigen::MatrixXd> a {}, b {};
		Eigen::VectorXd weighted_event_counts;
		bool state_cause, state_effect;
		int num_state_variables;
}

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
