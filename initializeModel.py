from runner.runner_utilities import extract_observed_data
from runner.run_model import fit_model
from runner.runner_utilities import build_intervention_info
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Guinea Worm Model")
    parser.add_argument("-n", "--num-its", action="store", help="number of iterations", default=1, type=int)
    parser.add_argument("-t", "--timestep", action="store", help="timestep of model", default=15, type=int)
    parser.add_argument("-f", "--fit", action="store", help="run over many parameters", default=False, type=bool)
    parser.add_argument("-p", "--population", action="store", help="population of model", default=800, type=int)
    parser.add_argument("--run-time", action="store", help="number of years to run for", default=20, type=int)
    parser.add_argument("-d", "--desc", action="store", help="folder description", default="", type=str)
    parser.add_argument("-b", "--burn-in", action="store", help="burnin time in years", default=5, type=int)
    parser.add_argument("--r-naught", action="store", help="R0 value", default=2.9, type=float)
    parser.add_argument("--rsw", action="store", help="R sink to worm value", default=0.4, type=float)
    parser.add_argument("--ndnc", action="store", help="ND/NC value [0, 1]", default=0.7, type=float)
    parser.add_argument("--exp-het", action="store", help="exposure heterogeneity parameter", default=0.39, type=float)
    
    arguments = parser.parse_args()

    input_params = []
    desc = arguments.desc
    num_iters = arguments.num_its
    if (desc == ""):
        desc = f"{num_iters}_its"
    # TODO: make more customizable
    if arguments.fit is True:
        r0s = np.arange(1, 4.1, 1).tolist() + [2.9]
        asymmetries = np.arange(1, 2.01, 0.05)
        r0_sink_to_worms = np.arange(0.1, 1, 0.1)
        nc_nhs = 1 / np.array([0.1, 0.50, 0.70, 0.9])
        exposure_heterogeneities = [0.39]
        initial_infecteds = [1]
        total_runs = len(r0_sink_to_worms) * len(r0s) * len(nc_nhs) * len(initial_infecteds) * len(exposure_heterogeneities) * num_iters
        for initial_infected in initial_infecteds:
            for exposure_heterogeneity in exposure_heterogeneities:
                for nc_nh in nc_nhs:
                    for r0 in r0s:
                        for r0_sink_to_worm in r0_sink_to_worms:
                            for run_num in range(0, num_iters):
                                input_params.append({
                                    "max_year": arguments.run_time,
                                    "timestep": arguments.timestep,
                                    "initial_infected": initial_infected,
                                    "r0": r0,
                                    "nc_nh": nc_nh,
                                    "r0_sink_to_worm": r0_sink_to_worm,
                                    "exposure_heterogeneity": exposure_heterogeneity,
                                    "verbose": False,
                                    "run_num": run_num,
                                    "use_intervention": False,
                                    "intervention_infos": [
                                        build_intervention_info(start_time_years=10, end_time_years=arguments.run_time, value=0.8, name="teathering"), 
                                        build_intervention_info(start_time_years=10, end_time_years=arguments.run_time, value=0, name="abate")
                                    ],
                                    "population": arguments.population,
                                    "desc": desc,
                                    "burn_in": arguments.burn_in,
                                    "observed_data": extract_observed_data(region="Cb")
                                })
        fit_model(
            input_params=input_params,
            total_runs=total_runs,
            desc=desc
        )
        
                    
    if not arguments.fit:
        print("Single run implementation TBD")
        # do_single_run(arguments)