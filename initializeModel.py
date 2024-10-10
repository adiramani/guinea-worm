from guinea_worm.model_wrapper import GuineaWormModel
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool, Manager, cpu_count

def fit_model(larval_death_rate, host_mortality_rate, worm_death_rate, initial_infected, initial_proportion_sink_infected, timestep, endtime, r0, nc_nd, transmission_asymmetry, verbose=False):
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
                "num_individuals":1000,
                "population_name":"dogs",
                "mortality_rate": host_mortality_rate,
                "worm_death_rate": worm_death_rate,
                "worm_mating_probability": 1,
                "ke":0.3,
                "initial_infected": initial_infected,
                "worm_maturity_age_days": 0,
                "max_worm_age": 360,
                "sink_interaction_values": {"copepod": {
                    "interaction": np.full(1000, 1),
                }}
            } 
        ],
        model_info={
            "time":0,
            "timestep": timestep,
            "endtime": endtime,
            "r0": r0,
            "NcNd": nc_nd,
            "transmission_asymmetry": transmission_asymmetry,
            "verbose":verbose
        })

    return gw_model.iterateFullModel()

def fit_data(params_to_fit):
    processed_data = fit_model(
            larval_death_rate = 1/30, 
            host_mortality_rate = (1/5)/360,
            worm_death_rate = 1/360,
            initial_infected=0,
            initial_proportion_sink_infected=params_to_fit["initial_infected"],
            timestep=params_to_fit["timestep"],
            endtime=360*params_to_fit["max_year"],
            r0=params_to_fit["r0"],
            nc_nd=params_to_fit["nc_nd"],
            transmission_asymmetry=params_to_fit["asymmetry"],
            verbose=params_to_fit["verbose"]
        )
    processed_data["r0"] = params_to_fit["r0"]
    processed_data["asymmetry"] = params_to_fit["asymmetry"]
    processed_data["run_num"] = params_to_fit["run_num"]
    processed_data["nd_nc"] = 1/params_to_fit["nc_nd"]
    processed_data["initial_infected"] = params_to_fit["initial_infected"]
    processed_data.to_csv(
        f"output_data/model_output_initinf_{params_to_fit['initial_infected']}_ncnd_{params_to_fit['nc_nd']}_r0_{params_to_fit['r0']}_asym_{params_to_fit['asymmetry']}_runnum_{params_to_fit['run_num']}.csv"
    )

if __name__ == '__main__':
    skip_fit = False
    max_year=10

    input_params = []
    if not skip_fit:
        r0s = np.arange(1, 5.1, 0.5)
        asymmetries = np.arange(1, 2.01, 0.1)
        nc_nds = 1 / np.array([0.05, 0.50, 0.95])
        initial_infecteds = [0.5, 0.25, 0.1]
        num_iters=10
        total_runs = len(asymmetries) * len(r0s) * len(nc_nds) * len(initial_infecteds) * num_iters
        for initial_infected in initial_infecteds:
            for nc_nd in nc_nds:
                for r0 in r0s:
                    for asymmetry in asymmetries:
                        for run_num in range(0, num_iters):
                            input_params.append({
                                "max_year": max_year,
                                "timestep": 15,
                                "initial_infected": initial_infected,
                                "r0": r0,
                                "nc_nd": nc_nd,
                                "asymmetry": asymmetry,
                                "verbose": False,
                                "run_num": run_num
                            })
                        #print(f"Initial Inf: {initial_infected}. NCND: {nc_nd}. R0: {r0}. Pi: {asymmetry}. Run Num: {run_num}")
        num_cpus = cpu_count()
        with tqdm(total=total_runs) as pbar:
            with Pool(processes=num_cpus-2) as pool:
                for _ in pool.imap_unordered(fit_data, input_params):
                    pbar.update(1)
                    
        #_ = process_map(fit_data, range(0, len(input_params)), params_to_fit=input_params, max_workers=num_cpus-2, chunksize=int(total_runs/70))

    if skip_fit:
        processed_data = fit_model(
            larval_death_rate = 1/30, 
            host_mortality_rate = (1/5)/360,
            worm_death_rate = 1/360,
            initial_infected=0,
            initial_proportion_sink_infected=0.5,
            timestep=15,
            endtime=360*20,
            r0=5,
            nc_nd=500,
            transmission_asymmetry=1,
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
