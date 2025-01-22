from pathlib import Path
from guinea_worm.model_wrapper import GuineaWormModel
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse



# Non-mated females should not emerge
def fit_model(total_pop, larval_death_rate, host_mortality_rate, worm_death_rate, worm_death_gamma_shape, exposure_heterogeneity, initial_infected, initial_proportion_sink_infected, timestep, use_intervention, intervention_infos, endtime, r0, nc_nd, r0_sink_to_worm, verbose=False):
    if not use_intervention:
        intervention_infos = []
    gw_model = GuineaWormModel(
        sink_info=[
            {#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6989452/
                "population_name":"copepod",
                "infectivity_rate": initial_proportion_sink_infected,
                "density": 250, # per liter
                "size": 4500,# liters
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
            "NcNd": nc_nd,
            "r0_sink_to_worm": r0_sink_to_worm,
            "verbose":verbose
        })

    return gw_model.iterateFullModel(output_interval_years=30/360)

def write_data(output_path, data, file_name):
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_path + file_name)

def fit_data(params_to_fit):
    processed_data = fit_model(
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
            nc_nd=params_to_fit["nc_nd"],
            r0_sink_to_worm=params_to_fit["r0_sink_to_worm"],
            verbose=params_to_fit["verbose"]
        )
    processed_data["r0"] = params_to_fit["r0"]
    processed_data["r0_sink_to_worm"] = params_to_fit["r0_sink_to_worm"]
    processed_data["run_num"] = params_to_fit["run_num"]
    processed_data["nd_nc"] = 1/params_to_fit["nc_nd"]
    processed_data["exposure_heterogeneity"] = params_to_fit["exposure_heterogeneity"]
    processed_data["initial_infected"] = params_to_fit["initial_infected"]

    write_data(
        f"output_data_{params_to_fit['desc']}/",
        processed_data,
        f"model_output_initinf_{params_to_fit['initial_infected']}_ndnc_{1/params_to_fit['nc_nd']}_r0_{params_to_fit['r0']}_r0sw_{params_to_fit['r0_sink_to_worm']}_runnum_{params_to_fit['run_num']}_exposhet_{params_to_fit['exposure_heterogeneity']}.csv"
    )

def build_intervention_info(start_time_years=10, end_time_years=11, name="teathering", value=0):
    return {
        "start_time": start_time_years * 360,
        "end_time": end_time_years * 360,
        "intervention_name": name,
        "intervention_value": value,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Guinea Worm Model")
    parser.add_argument("-n", "--num-its", action="store", help="number of iterations", default=1, type=int)
    parser.add_argument("-t", "--timestep", action="store", help="timestep of model", default=15, type=int)
    parser.add_argument("-f", "--fit", action="store", help="run over many parameters", default=False, type=bool)
    parser.add_argument("-p", "--population", action="store", help="population of model", default=800, type=int)
    parser.add_argument("--run-time", action="store", help="number of years to run for", default=20, type=int)
    parser.add_argument("-d", "--desc", action="store", help="folder description", default="", type=str)
    arguments = parser.parse_args()

    input_params = []
    if arguments.fit is True:
        r0s = np.arange(1, 4.1, 1)
        asymmetries = np.arange(1, 2.01, 0.05)
        r0_sink_to_worms = np.arange(0.1, 1, 0.1)
        nc_nds = 1 / np.array([0.1, 0.50, 0.9])
        exposure_heterogeneities = [0.39]
        initial_infecteds = [1]
        num_iters = arguments.num_its
        desc = arguments.desc
        if (desc == ""):
            desc = f"{num_iters}_its"
        total_runs = len(r0_sink_to_worms) * len(r0s) * len(nc_nds) * len(initial_infecteds) * len(exposure_heterogeneities) * num_iters
        for initial_infected in initial_infecteds:
            for exposure_heterogeneity in exposure_heterogeneities:
                for nc_nd in nc_nds:
                    for r0 in r0s:
                        for r0_sink_to_worm in r0_sink_to_worms:
                            for run_num in range(0, num_iters):
                                input_params.append({
                                    "max_year": arguments.run_time,
                                    "timestep": arguments.timestep,
                                    "initial_infected": initial_infected,
                                    "r0": r0,
                                    "nc_nd": nc_nd,
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
                                    "desc": desc
                                })

        num_cpus = cpu_count()
        with tqdm(total=total_runs) as pbar:
            with Pool(processes=num_cpus-2) as pool:
                for _ in pool.imap_unordered(fit_data, input_params):
                    pbar.update(1)
                    
    if not arguments.fit:
        processed_data = fit_model(
            total_pop=arguments.population,
            larval_death_rate = 1/30, 
            host_mortality_rate = (2/5)/360,
            exposure_heterogeneity=0.3,
            worm_death_rate = (12/11)/360, # mean of 11 months
            worm_death_gamma_shape = 37,
            initial_infected=arguments.population,
            initial_proportion_sink_infected=0,
            timestep=arguments.timestep,
            use_intervention=False,
            intervention_infos=[
                build_intervention_info(start_time_years=10, end_time_years=arguments.run_time, value=0.8), 
                build_intervention_info(start_time_years=5, end_time_years=arguments.run_time, value=0, name="abate")
            ],
            endtime=360*arguments.run_time,
            r0=2,
            nc_nd=1/0.1,
            r0_sink_to_worm=0.9,
            verbose=True
        )

        fig, axes = plt.subplots(nrows=4)
        axes[0].plot(processed_data.loc[processed_data["measure"] == "infective_larvae", "year"], processed_data.loc[processed_data["measure"] == "infective_larvae", "value"])
        axes[0].set_title("infective larvae")
        axes[1].plot(processed_data.loc[processed_data["measure"] == "female_worm_prev", "year"], processed_data.loc[processed_data["measure"] == "female_worm_prev", "value"])
        axes[1].set_title("female worm prev")
        axes[2].plot(processed_data.loc[processed_data["measure"] == "emergence_copepod", "year"], processed_data.loc[processed_data["measure"] == "emergence_copepod", "value"])
        axes[2].set_title("worm emergences")
        axes[3].plot(processed_data.loc[processed_data["measure"] == "Re", "year"], processed_data.loc[processed_data["measure"] == "Re", "value"])
        axes[3].set_title("Re")
        plt.show()
