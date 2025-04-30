from pathlib import Path
from guinea_worm.model_wrapper import GuineaWormModel
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse


def plot_measure(df, axis, measure, title=""):
    if len(title) == 0:
        title = measure
    axis.plot(df.loc[df["measure"] == measure, "year"], df.loc[df["measure"] == measure, "mean_value"])
    axis.fill_between(df.loc[df["measure"] == measure, "year"], df.loc[df["measure"] == measure, "2.5_percentile"],  df.loc[df["measure"] == measure, "97.5_percentile"], alpha=0.4)
    axis.set_title(title)
    axis.set_ylim(bottom=0)

# Non-mated females should not emerge
def fit_model(total_pop, larval_death_rate, host_mortality_rate, worm_death_rate, worm_death_gamma_shape, exposure_heterogeneity, initial_infected, initial_proportion_sink_infected, timestep, use_intervention, intervention_infos, endtime, r0, nc_nh, r0_sink_to_worm, burnin, verbose=False):
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
        f"model_output_initinf_{params_to_fit['initial_infected']}_nhnc_{1/params_to_fit['nc_nh']}_r0_{params_to_fit['r0']}_r0sw_{params_to_fit['r0_sink_to_worm']}_runnum_{params_to_fit['run_num']}_exposhet_{params_to_fit['exposure_heterogeneity']}.csv"
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
                                    "burn_in": arguments.burn_in
                                })

        num_cpus = cpu_count()
        with tqdm(total=total_runs) as pbar:
            with Pool(processes=num_cpus-2) as pool:
                for _ in pool.imap_unordered(fit_data, input_params):
                    pbar.update(1)
                    
    if not arguments.fit:
        all_data = None
        for run_num in range(num_iters):

            processed_data = fit_model(
                total_pop=arguments.population,
                larval_death_rate = 1/30, 
                host_mortality_rate = (2/5)/360,
                exposure_heterogeneity=arguments.exp_het,
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
                r0=arguments.r_naught,
                nc_nh=1/arguments.ndnc,
                r0_sink_to_worm=arguments.rsw,
                verbose=True,
                burnin=arguments.burn_in
            )
            processed_data["r0"] = arguments.r_naught
            processed_data["r0_sink_to_worm"] = arguments.rsw
            processed_data["run_num"] = run_num
            processed_data["nh_nc"] = arguments.ndnc
            processed_data["exposure_heterogeneity"] = arguments.exp_het
            processed_data["initial_infected"] = 1
            if all_data is None:
                processed_data["value_0"] = processed_data["value"]
                all_data = processed_data
            else:
                all_data[f"value_{run_num}"] = processed_data["value"]

        write_data(
            f"output_data_{desc}/",
            all_data,
            f"model_output_nhnc_{arguments.ndnc}_r0_{arguments.r_naught}_r0sw_{arguments.rsw}_runnum_{run_num}_exposhet_{arguments.exp_het}.csv"
        )
        rows_for_mean = [f"value_{iter}" for iter in range(num_iters)]
        all_data["mean_value"] = all_data[rows_for_mean].mean(axis=1)
        all_data["2.5_percentile"] = all_data[rows_for_mean].quantile(axis=1, q=.025)
        all_data["97.5_percentile"] = all_data[rows_for_mean].quantile(axis=1, q=.975)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
        axes[0][0] = plot_measure(all_data, axes[0][0], "infective_larvae", "Proportion of Copepods Infected")
        axes[0][1] = plot_measure(all_data, axes[0][1], "female_worm_prev", "Female Worm Prevalence in Dogs")
        axes[1][0] = plot_measure(all_data, axes[1][0], "emergent_host_prevalence_copepod", "Emergence Prevalence in Dogs")
        axes[1][1] = plot_measure(all_data, axes[1][1], "emergences_per_infected_host_copepod", "Emergences per Infected Dog")
        plt.tight_layout()
        plot_path = Path(f"visualizations/images/{desc}/")
        plot_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"visualizations/images/{desc}/single_parameter_graph.png")
        plt.show()