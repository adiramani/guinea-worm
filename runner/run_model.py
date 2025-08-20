from guinea_worm.model_wrapper import GuineaWormModel
from guinea_worm.tools import write_data
from runner.runner_utilities import calculate_mse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd



def run_model(total_pop, larval_death_rate, host_mortality_rate, worm_death_rate, worm_death_gamma_shape, exposure_heterogeneity, initial_infected, initial_proportion_sink_infected, timestep, use_intervention, intervention_infos, endtime, r0, nc_nh, r0_sink_to_worm, burnin, verbose=False):
    if not use_intervention:
        intervention_infos = []
    gw_model = GuineaWormModel(
        sink_info=[
            {#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6989452/
                "population_name":"copepod",
                "infectivity_rate": initial_proportion_sink_infected,
                "density": 250, # per liter, unused
                "size": 4500,# liters, unused
                "larval_death_rate": larval_death_rate
            }
        ],
        host_info=[
            {
                "timestep": timestep,
                "num_individuals":total_pop,
                "population_name":"dogs",
                "mortality_rate": host_mortality_rate,
                "worm_death_rate": worm_death_rate,
                "worm_death_gamma_shape": worm_death_gamma_shape,
                "worm_mating_probability": 1,
                "worms_to_infect_with": 1,
                "ke":exposure_heterogeneity,
                "initial_infected": initial_infected,
                "worm_maturity_age_days": 0,
                "max_worm_age": 360+180,
                "sink_interaction_values": {"copepod": {
                    "interaction": np.full(total_pop, 1),
                }}
            } 
        ],
        intervention_info=intervention_infos,
        model_info={
            "time":0,
            "timestep": timestep,
            "endtime": endtime,
            "r0": r0,
            "NcNh": nc_nh,
            "r0_sink_to_worm": r0_sink_to_worm,
            "burnin_time_years": burnin,
            "verbose":verbose
        })

    return gw_model.iterateFullModel(output_interval_years=30/360)

def run_with_params(params_to_fit):
    processed_data = run_model(
            total_pop = params_to_fit["population"],
            larval_death_rate = 1/30, 
            host_mortality_rate = (2/5)/360,
            worm_death_rate = (12/11)/360, # Average lifespan of 11 months
            worm_death_gamma_shape = 37,
            exposure_heterogeneity=params_to_fit["exposure_heterogeneity"],
            initial_infected=int(params_to_fit["initial_infected"]*params_to_fit["population"]),
            initial_proportion_sink_infected=0,
            timestep=params_to_fit["timestep"],
            use_intervention=params_to_fit["use_intervention"],
            intervention_infos=params_to_fit["intervention_infos"],
            endtime=360*params_to_fit["max_year"],
            r0=params_to_fit["r0"],
            nc_nh=params_to_fit["nc_nh"],
            r0_sink_to_worm=params_to_fit["r0_sink_to_worm"],
            burnin=params_to_fit["burn_in"],
            verbose=params_to_fit["verbose"]
        )
    processed_data["r0"] = params_to_fit["r0"]
    processed_data["r0_sink_to_worm"] = params_to_fit["r0_sink_to_worm"]
    processed_data["run_num"] = params_to_fit["run_num"]
    processed_data["nh_nc"] = 1/params_to_fit["nc_nh"]
    processed_data["exposure_heterogeneity"] = params_to_fit["exposure_heterogeneity"]
    processed_data["initial_infected"] = params_to_fit["initial_infected"]

    write_data(
        f"output_data_{params_to_fit['desc']}/",
        processed_data,
        f"model_output_initinf_{params_to_fit['initial_infected']}_nhnc_{round(1/params_to_fit['nc_nh'], 2)}_r0_{params_to_fit['r0']}_r0sw_{round(params_to_fit['r0_sink_to_worm'], 2)}_runnum_{params_to_fit['run_num']}_exposhet_{params_to_fit['exposure_heterogeneity']}.csv"
    )
    return processed_data

def run_with_params_mse(params_to_fit):
    processed_data = run_with_params(params_to_fit)
    mse_val = calculate_mse(processed_data, params_to_fit["observed_data"])
    return {
        "r0": params_to_fit["r0"],
        "r0_sink_to_worm": params_to_fit["r0_sink_to_worm"],
        "run_num": params_to_fit["run_num"],
        "nh_nc": 1/params_to_fit["nc_nh"],
        "exposure_heterogeneity": params_to_fit["exposure_heterogeneity"],
        "mse": mse_val
    }

def fit_model(input_params, total_runs, desc):
    num_cpus = cpu_count()
    fitting_results = []
    with tqdm(total=total_runs) as pbar:
        with Pool(processes=num_cpus-2) as pool:
            for res in pool.imap_unordered(run_with_params_mse, input_params):
                fitting_results.append(res)
                pbar.update(1)
    write_data(
        f"output_data_{desc}/fit_results/",
        pd.DataFrame(fitting_results),
        "fit_results.csv"
    )