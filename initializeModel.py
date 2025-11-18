from guinea_worm.tools import write_data
from runner.runner_utilities import extract_observed_data
from runner.run_model import fit_model_grid, run_model_n_times_parallel
from runner.runner_utilities import build_intervention_info
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_single_runs(model_output):
    fig, axes = plt.subplots(nrows=4)
    axes[0].plot(model_output.loc[model_output["measure"] == "infective_larvae", "year"], model_output.loc[model_output["measure"] == "infective_larvae", "value"])
    axes[0].set_title("infective larvae")
    axes[1].plot(model_output.loc[model_output["measure"] == "emergences_per_infected_host_copepod", "year"], model_output.loc[model_output["measure"] == "emergences_per_infected_host_copepod", "value"])
    axes[1].set_title("emergences_per_infected_host_copepod")
    axes[2].plot(model_output.loc[model_output["measure"] == "emergence_copepod", "year"], model_output.loc[model_output["measure"] == "emergence_copepod", "value"])
    axes[2].set_title("worm emergences")
    axes[3].plot(model_output.loc[model_output["measure"] == "female_worm_prev", "year"], model_output.loc[model_output["measure"] == "female_worm_prev", "value"])
    axes[3].set_title("female worm prev")
    plt.show()

def create_param_set(input_args, r0=2.96, initial_infected=1, exposure_heterogeneity=0.26, a=0.0089, year_over_year_r0_factor = None, desc="", use_intervention=False, abate_eff=0, tether_eff = 0):
    return {
        "max_year": input_args.run_time,
        "timestep": input_args.timestep,
        "initial_infected": initial_infected,
        "r0": round(r0, 3),
        "a": round(a, 4),
        "exposure_heterogeneity": round(exposure_heterogeneity, 3),
        "verbose": False,
        "use_intervention": use_intervention,
        "intervention_infos": [
            build_intervention_info(start_time_years=input_args.fit_time+6, end_time_years=input_args.fit_time+17, value=0.839*tether_eff, name="tethering"), 
            # build_intervention_info(start_time_years=input_args.fit_time+6, end_time_years=input_args.fit_time+7, value=0.78, name="tethering"), 
            # build_intervention_info(start_time_years=input_args.fit_time+7, end_time_years=input_args.fit_time+8, value=0.80, name="tethering"), 
            # build_intervention_info(start_time_years=input_args.fit_time+8, end_time_years=input_args.fit_time+9, value=0.73, name="tethering"), 
            # build_intervention_info(start_time_years=input_args.fit_time+9, end_time_years=input_args.fit_time+10, value=0.76, name="tethering"), 
            # build_intervention_info(start_time_years=input_args.fit_time+10, end_time_years=input_args.fit_time+11, value=0.71, name="tethering"), 
            # build_intervention_info(start_time_years=input_args.fit_time+11, end_time_years=input_args.fit_time+12, value=0.60, name="tethering"), 
            build_intervention_info(
                start_time_years=input_args.fit_time+6, end_time_years=input_args.fit_time+17, value=abate_eff, name="abate",
                months_to_apply=[1, 2, 3, 4, 5, 6, 7]
            )
        ],
        "year_over_year_r0_factor": year_over_year_r0_factor,
        "measure_to_fit": input_args.fit_measure,
        "population": input_args.population,
        "desc": desc,
        "burn_in": input_args.burn_in,
        "fit_time": input_args.fit_time,
        "seasonality_start": input_args.fit_time-4
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Guinea Worm Model")
    parser.add_argument("-n", "--num-its", action="store", help="number of iterations", default=1, type=int)
    parser.add_argument("-t", "--timestep", action="store", help="timestep of model", default=15, type=int)
    parser.add_argument("-f", "--fit", action="store", help="run over many parameters", default=False, type=bool)
    parser.add_argument("-p", "--population", action="store", help="population of model", default=800, type=int)
    parser.add_argument("--run-time", action="store", help="number of years to run for", default=40, type=int)
    parser.add_argument("-d", "--desc", action="store", help="folder description", default="", type=str)
    parser.add_argument("-b", "--burn-in", action="store", help="burnin time in years", default=5, type=int)
    parser.add_argument("--r-naught", action="store", help="R0 value", default=2.9, type=float)
    parser.add_argument("-a", action="store", help="a value", default=0.0089, type=float)
    parser.add_argument("--exp-het", action="store", help="exposure heterogeneity parameter", default=0.26, type=float)
    parser.add_argument("--fit-measure", action="store", help="measure to use to fit", default="single", type=str)
    parser.add_argument("--fit-time", action="store", help="model time to start fit", default=24, type=int)
    parser.add_argument("--fit-cluster", action="store", help="cluster to fit to", default=-1, type=int)
    parser.add_argument("-v", "--verbose", action="store", help="print yearly model results", default=False, type=bool)
    parser.add_argument("--single-file-name", action="store", help="name of single model run file", default="", type=str)
    parser.add_argument("--abate-efficacy", action="store", help="efficacy of abate", default=0, type=float)
    parser.add_argument("--tether-efficacy", action="store", help="efficacy of tethering", default=0, type=float)
    


    arguments = parser.parse_args()
    desc = arguments.desc
    num_iters = arguments.num_its
    if (desc == ""):
        desc = f"{num_iters}_its"

    if (arguments.fit_measure == "single"):
        model_output = run_model_n_times_parallel(
            param_set = create_param_set(input_args=arguments, r0=arguments.r_naught, exposure_heterogeneity=arguments.exp_het, a=arguments.a, desc=desc, use_intervention=True, abate_eff=arguments.abate_efficacy, tether_eff=arguments.tether_efficacy),
            num_iters=arguments.num_its,
            desc=desc,
        )

        write_data(
            f"output_data_{desc}/total_results/",
            model_output,
            f"model_output_initinf_{arguments.single_file_name}.csv"
        )

        # plot_single_runs(model_output)
    else:
        input_params = []
        # TODO: make more customizable
        r0s = np.arange(1, 7.1, 1).tolist() # [2.96]
        a_vals = np.arange(0.001, 0.005, 0.0002).tolist() # [0.0089]
        exposure_heterogeneities = np.arange(0.05, 0.5, 0.05) # [0.26]
        initial_infecteds = [1]
        total_runs = len(a_vals) * len(r0s) * len(initial_infecteds) * len(exposure_heterogeneities) * num_iters
        for initial_infected in initial_infecteds:
            for exposure_heterogeneity in exposure_heterogeneities:
                for a in a_vals:
                #for nc_nh in nc_nhs:
                    for r0 in r0s:
                        #for r0_sink_to_worm in r0_sink_to_worms:
                        input_params.append({
                            "max_year": arguments.run_time,
                            "timestep": arguments.timestep,
                            "initial_infected": initial_infected,
                            "r0": round(r0, 3),
                            "a": round(a, 4),
                            "exposure_heterogeneity": round(exposure_heterogeneity, 3),
                            "verbose": False,
                            "use_intervention": False,
                            "intervention_infos": [
                                # build_intervention_info(start_time_years=10, end_time_years=arguments.run_time, value=0.8, name="tethering"), 
                                # build_intervention_info(start_time_years=10, end_time_years=arguments.run_time, value=0, name="abate")
                            ],
                            "measure_to_fit": arguments.fit_measure,
                            "population": arguments.population,
                            "desc": desc,
                            "burn_in": arguments.burn_in,
                            "fit_time": arguments.fit_time,
                            "seasonality_start": arguments.fit_time-4
                        })
        fit_model_grid(
            input_params=input_params,
            total_runs=total_runs,
            iters_per_param_set=num_iters,
            desc=desc,
            observed_data = extract_observed_data(fit_measure = arguments.fit_measure, start_year=2014, cluster = arguments.fit_cluster)
        )