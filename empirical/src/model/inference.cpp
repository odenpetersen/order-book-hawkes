#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <fstream>

#include <Eigen/Dense>

// Example usage
#include "Parse.h"
#include "Kernel.h"
#include "constants.h"

int main() {
	CompositeKernel kernel(NUM_EVENT_TYPES,START_TIME,END_TIME);
	kernel.add_kernel(new PoissonKernel(NUM_EVENT_TYPES,START_TIME,END_TIME));
	for (int i = 0; i < 5; i++) {
		kernel.add_kernel(new LinearSpline(NUM_EVENT_TYPES,START_TIME,END_TIME));
	}
	REAL lr = LR_START;
	REAL perturb_amt = NOISE_START;
	Eigen::VectorXd step;
	do {
		std::cout << "lr" << std::log(lr)/std::log(10) << std::endl;

		//Realisation session("../../output/databento/glbx-mdp3-20240913.csv"); // Replace with your CSV file path
		Realisation session("../../output/databento_example.csv"); // Replace with your CSV file path
		kernel.reset();

		int bucket_size = 100000;
		int bucket_counter = 0;
		std::cout << std::setprecision(5);
		for (const auto event : session) {
			kernel.update(*event);
			bucket_counter++;
			if (bucket_counter >= bucket_size) {
				assert (event->time<=END_TIME);
				std::cout << (event->time / seconds_in_day)*100 << "\% done" << std::endl;
				std::cout << *event << std::endl;
				bucket_counter = 0;
			}
		}

		std::cout << std::setprecision(20);

		/*
		std::cout << "Getting hess and grad" << std::endl;
		auto [hess,grad] = kernel.get_hessian_and_gradient();
		std::cout << "Got hess and grad, " << hess.rows() << "x" << hess.cols() << ", " << grad.rows() << std::endl;
		std::cout << hess << std::endl;
		std::cout << grad << std::endl;
		std::cout << kernel.get_params() << std::endl;
		std::cout << "Grad norm: " << grad.norm() << std::endl;
		*/
		std::cout << "Solving" << std::endl;
		step = kernel.get_em_step();
		std::cout << "Solved" << std::endl;
		std::cout << "Step vector size: " << step.size() << " (" << step.rows() << "," << step.cols() << ")" << std::endl;
		std::cout << "Step vector norm: " << step.norm() << std::endl;
		std::cout << "Step Linfty norm: " << step.cwiseAbs().maxCoeff() << std::endl;
		//std::cout << step << std::endl;
		std::cout << "Setting params" << std::endl;
		std::cout << "Parameter vector norm before: " << kernel.get_params().norm() << std::endl;
		kernel.parameter_step(lr*step);
		std::cout << "Parameter vector norm after: " << kernel.get_params().norm() << std::endl;
		//std::cout << kernel.get_params() << std::endl;

		std::cout << std::setprecision(5);
		std::cout << kernel.as_string() << std::endl;
		std::cout << std::setprecision(20);

		std::cout << "Checking parameter shape integrity" << std::endl;
		Eigen::VectorXd old_params = kernel.get_params();
		kernel.set_params(old_params);
		Eigen::VectorXd new_params = kernel.get_params();
		assert(old_params == new_params);
		std::cout << "Checked." << std::endl;

		std::cout << "Perturb by " << perturb_amt << std::endl;
		kernel.perturb(perturb_amt);

		lr = lr*(1-LR_DECAY) + LR_DECAY*LR_END;
		perturb_amt *= NOISE_DECAY;
	} while (step.cwiseAbs().maxCoeff() > STOPPING_CRITERION);

	Realisation session("../../output/databento_example.csv"); // Replace with your CSV file path
	kernel.reset();
	std::cout << "time after reset " << kernel.current_time << ", " << kernel.start_time << std::endl;

	std::ofstream file;
	file.open("../../output/residuals_example.csv");
	file << std::setprecision(20);
	REAL residual, intensity;
	for (const auto event : session) {
		auto [residual,intensity] = kernel.update(*event);
		assert (residual>=0);
		file << residual << "," << intensity << std::endl;
	}
	file.close();

	return 0;
}
