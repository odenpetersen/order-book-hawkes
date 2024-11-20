#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <fstream>
#include <filesystem>

#include <Eigen/Dense>

// Example usage
#include "Parse.h"
#include "Kernel.h"
#include "constants.h"

int main(int argc, char *argv) {
	std::string output_folder = "../../output/";
	output_folder += argv[1];
	
	std::cout << "Output folder: " << output_folder << std::endl;

	assert(std::filesystem::create_directory(output_folder));
	assert(std::filesystem::create_directory(output_folder + "/residuals/"));
	assert(std::filesystem::create_directory(output_folder + "/simulations/"));

	CompositeKernel kernel(NUM_EVENT_TYPES,START_TIME,END_TIME);
	kernel.add_kernel(new PoissonKernel(NUM_EVENT_TYPES,START_TIME,END_TIME));
	/*
	for (int i = 0; i < 40; i++) {
		kernel.add_kernel(new LinearSpline(NUM_EVENT_TYPES,START_TIME,END_TIME));
	}
	*/
	for (int i = 0; i < 10; i++) {
		//kernel.add_kernel(new ContinuousStateHawkesKernel(NUM_EVENT_TYPES,START_TIME,END_TIME,NUM_MARK_VARIABLES,true,true,true,false));
		kernel.add_kernel(new ContinuousStateHawkesKernel(NUM_EVENT_TYPES,START_TIME,END_TIME,NUM_MARK_VARIABLES,true,true,true,i<5));
	}
	//std::string prefix = "../../output/databento_collated_fullday/glbx-mdp3-2024";
	std::string prefix = "../../output/databento_collated/glbx-mdp3-2024";
	std::vector<std::string> train_files = {"0708.csv","0709.csv","0710.csv","0711.csv","0712.csv","0715.csv","0716.csv","0717.csv","0718.csv","0719.csv"};
	std::vector<std::string> test_files = {"0722.csv","0723.csv","0724.csv","0725.csv","0726.csv","0729.csv","0730.csv","0731.csv","0801.csv","0802.csv"};

	REAL lr = LR_START;
	REAL perturb_amt = NOISE_START;
	Eigen::VectorXd step;
	std::cout << std::setprecision(20);
	do {
		std::cout << "kernel " << kernel.as_string() << std::endl;

		std::cout << "lr " << std::log(lr)/std::log(10) << std::endl;

		std::cout << "Perturb by " << perturb_amt << std::endl;
		kernel.perturb(perturb_amt);
		perturb_amt *= NOISE_DECAY;

		Eigen::VectorXd hess = Eigen::VectorXd::Constant(kernel.get_param_count(),0.0);
		Eigen::VectorXd grad = Eigen::VectorXd::Constant(kernel.get_param_count(),0.0);
		//
		//std::cout << std::setprecision(5);
		for (std::string file : train_files) {
			std::cout << prefix+file << std::endl;
			Realisation session(prefix+file); 
			kernel.reset();

			/*
			int num_events_observed = 0;
			int bucket_size = 1000;
			int bucket_counter = 0;
			*/
			std::cout << "Processing Events" << std::endl;
			for (const auto event : session) {
				//num_events_observed++;
				kernel.update(*event);
				/*
				bucket_counter++;
				if (bucket_counter >= bucket_size) {
					assert (event->time<=END_TIME);
					std::cout << (event->time / seconds_in_day)*100 << "\% done (" << event->time << " seconds, " << num_events_observed << " events)" << std::endl;
					std::cout << *event << std::endl;
					bucket_counter = 0;
					for (Kernel* k : kernel.kernels) {
						std::cout << k->as_string() << std::endl;
						auto pair = k->get_hessian_and_gradient();
						//std::cout << "hess " << pair.first.norm() << std::endl;
						//std::cout << "grad " << pair.second.norm() << std::endl;
						std::cout << "grad/hess " << (pair.first.array()==0.0).select(0.0,pair.second.cwiseQuotient(pair.first)).norm() << std::endl;
						//std::cout << "step " << k->get_em_step().norm() << std::endl;
					}
				}
				*/
			}
			kernel.progress_time(END_TIME - kernel.current_time);

			/*
			for (Kernel* k : kernel.kernels) {
				std::cout << k->as_string() << std::endl;
				auto pair = k->get_hessian_and_gradient();
				std::cout << "hess " << pair.first.norm() << std::endl;
				std::cout << "grad " << pair.second.norm() << std::endl;
				std::cout << "grad/hess " << (pair.first.array()==0.0).select(0.0,pair.second.cwiseQuotient(pair.first)).norm() << std::endl;
				//std::cout << "step " << k->get_em_step().norm() << std::endl;
			}
			*/

			std::pair<Eigen::VectorXd,Eigen::VectorXd> pair = kernel.get_hessian_and_gradient();
			hess += pair.first;
			grad += pair.second;

			//std::cout << "step " << grad.cwiseQuotient(hess.cwiseAbs()) << std::endl;
		}

		//std::cout << std::setprecision(20);

		//step = grad.cwiseQuotient(hess.cwiseAbs()).unaryExpr([](double v) { return std::isfinite(v)? v : 0.0; });
		step = grad.cwiseQuotient(hess.cwiseAbs());
		std::cout << "Step Linfty norm: " << step.cwiseAbs().maxCoeff() << std::endl;

		/*
		std::cout << "Getting hess and grad" << std::endl;
		auto [hess,grad] = kernel.get_hessian_and_gradient();
		std::cout << "Got hess and grad, " << hess.rows() << "x" << hess.cols() << ", " << grad.rows() << std::endl;
		std::cout << hess << std::endl;
		std::cout << grad << std::endl;
		std::cout << kernel.get_params() << std::endl;
		std::cout << "Solving" << std::endl;
		step = kernel.get_em_step();
		std::cout << "Solved" << std::endl;
		*/
		//std::cout << step << std::endl;
		std::cout << "Setting params" << std::endl;
		std::cout << "Parameter vector norm before: " << kernel.get_params().norm() << std::endl;
		kernel.parameter_step(lr*step);
		std::cout << "Parameter vector norm after: " << kernel.get_params().norm() << std::endl;
		//std::cout << kernel.get_params() << std::endl;

		//std::cout << std::setprecision(5);
		std::cout << kernel.as_string() << std::endl;
		//std::cout << std::setprecision(20);

		std::cout << "Checking parameter shape integrity" << std::endl;
		Eigen::VectorXd old_params = kernel.get_params();
		kernel.set_params(old_params);
		Eigen::VectorXd new_params = kernel.get_params();
		assert(old_params == new_params);
		std::cout << "Checked." << std::endl;

		lr = lr*(1-LR_DECAY) + LR_DECAY*LR_END;

		std::cout << "precision, threshold = " << step.cwiseAbs().maxCoeff() << "," << ATOL + RTOL*kernel.get_params().cwiseAbs().minCoeff() << std::endl;
	} while (step.cwiseAbs().maxCoeff() > ATOL + RTOL*kernel.get_params().cwiseAbs().minCoeff());

	{
		std::ofstream file(output_folder + "/params.csv",std::ios::out);
		file << kernel.get_params() << std::endl;
		file.close();
	}

	{
		std::ofstream file(output_folder + "/describe.csv",std::ios::out);
		file << kernel.as_string() << std::endl;
		file.close();
	}

	for (std::vector<std::string> files : {train_files, test_files}) {
		for (std::string filename : files) {
			std::cout << "residuals for " << filename << std::endl;
			std::string output_file = output_folder + "/residuals_poweredstatedependenthawkesprocess_5pos5neg/" + filename;
			std::cout << "Writing to " << output_file << std::endl;
			std::ofstream file(output_file,std::ios::out);
			file << std::setprecision(20);

			Realisation session(prefix+filename);

			kernel.reset();
			REAL residual, intensity;
			for (const auto event : session) {
				auto [residual,intensity] = kernel.update(*event);
				if (!(residual>=0)) {
					std::cout << *event << std::endl;
					std::cout << kernel.current_time << std::endl;
					std::cout << "residual " << residual << ", intensity " << intensity << std::endl;
					//std::cout << "kernel params" << kernel.get_params() << std::endl;
					assert(false);
				}
				file << event->time << "," << residual << "," << intensity << std::endl;
			}
			residual = kernel.progress_time(END_TIME - kernel.current_time);
			file << residual << std::endl;
			file.close();
		}
	}

	//Simulate -> fit to simulated data -> record params (for hypothesis testing) and residuals (should be Exp(1) for simulated data)

	return 0;
}
