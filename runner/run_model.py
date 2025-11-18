from functools import partial
from math import inf, isfinite
from guinea_worm.model_wrapper import GuineaWormModel
from guinea_worm.tools import write_data
from runner.fitting_helpers import calc_log_likelihood, log_prior
from runner.runner_utilities import calculate_sse, combine_fit_and_general_param_set
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import emcee


def run_model(
        total_pop = None,
        larval_death_rate = None,
        host_mortality_rate = None,
        worm_death_rate = None,
        worm_death_gamma_shape = None,
        exposure_heterogeneity = None,
        initial_infected = None, 
        initial_proportion_sink_infected = None,
        timestep = None,
        use_intervention = None,
        intervention_infos = None,
        endtime = None,
        r0 = None,
        a = None,
        year_over_year_r0_factor = None,
        burnin = None,
        seasonality_start = None,
        beta_dist_mu = None,
        beta_dist_kappa = None,
        verbose=False
):
    if not use_intervention:
        intervention_infos = []
    gw_model = GuineaWormModel(
        sink_info=[
            {
                #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6989452/
                "timestep": timestep,
                # "population_name":"copepod",
                "infectivity_rate": initial_proportion_sink_infected,
                # "density": 250, # per liter, unused
                # "size": 4500,# liters, unused
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
                # "worm_mating_probability": 1,
                # "worms_to_infect_with": 1,
                "k":exposure_heterogeneity,
                "initial_infected": initial_infected,
                # "worm_maturity_age_days": 0,
                "sink_interaction_values": {"copepod": {
                    "interaction": np.full(total_pop, 1),
                }}
            }
        ],
        intervention_info=intervention_infos,
        model_info={
            "timestep": timestep,
            "endtime": endtime,
            "r0": r0,
            "a": a,
            "beta_dist_mu": beta_dist_mu,
            "beta_dist_kappa": beta_dist_kappa,
            "year_over_year_r0_factor": year_over_year_r0_factor,
            "burnin_time_years": burnin,
            "seasonality_start": seasonality_start,
            "verbose":verbose
        })

    return gw_model.iterateFullModel(output_interval_years=30/360)

def run_with_params(run_num, params_to_fit):
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
            a=params_to_fit["a"],
            year_over_year_r0_factor=params_to_fit.get("year_over_year_r0_factor", None),
            beta_dist_mu=params_to_fit.get("beta_dist_mu", None),
            beta_dist_kappa=params_to_fit.get("beta_dist_kappa", None),
            burnin=params_to_fit["burn_in"],
            verbose=params_to_fit["verbose"],
            seasonality_start=params_to_fit["seasonality_start"]
        )
    processed_data["r0"] = params_to_fit["r0"]
    processed_data["run_num"] = run_num
    processed_data["a"] = params_to_fit["a"]
    processed_data["exposure_heterogeneity"] = params_to_fit["exposure_heterogeneity"]
    processed_data["initial_infected"] = params_to_fit["initial_infected"]
    processed_data["abate_efficacy"] = next((item["intervention_value"] for item in params_to_fit["intervention_infos"] if item["intervention_name"] == "abate"), None)
    processed_data["tethering_efficacy"] = next((item["intervention_value"] for item in params_to_fit["intervention_infos"] if item["intervention_name"] == "tethering"), None)

    return processed_data

def run_with_params_sse(run_num, params_to_fit, observed_data):
    processed_data = run_with_params(run_num, params_to_fit)
    sse_val = calculate_sse(processed_data, observed_data, fit_measure=params_to_fit["measure_to_fit"], start_year=params_to_fit["fit_time"])
    return {
        "sse_summary": {
            "r0": params_to_fit["r0"],
            "a": params_to_fit["a"],
            "run_num": run_num,
            "exposure_heterogeneity": params_to_fit["exposure_heterogeneity"],
            "sse": sse_val
        }, 
        "raw_data": processed_data
    }

def run_model_n_times_parallel(param_set, num_iters, desc):
    num_cpus = min(cpu_count(), num_iters)
    with tqdm(total=num_iters) as pbar:
        with Pool(processes=num_cpus) as pool:
            param_data = []
            partial_run_with_params = partial(
                run_with_params,
                params_to_fit = param_set,
            )
            for res in pool.imap_unordered(partial_run_with_params, range(0, num_iters)):
                param_data.append(res)
                pbar.update(1)

            model_outputs = pd.concat(param_data, ignore_index=True)
    return model_outputs

def fit_model_grid(input_params, total_runs, iters_per_param_set, desc, observed_data):
    num_cpus = min(cpu_count(), iters_per_param_set)
    fitting_results = []
    with tqdm(total=total_runs) as pbar:
        with Pool(processes=num_cpus-2) as pool:
            for param_set in input_params:
                param_data = []
                partial_run_with_params_sse = partial(
                    run_with_params_sse,
                    params_to_fit = param_set,
                    observed_data = observed_data,
                )
                for res in pool.imap_unordered(partial_run_with_params_sse, range(0, iters_per_param_set)):
                    fitting_results.append(res["sse_summary"])
                    param_data.append(res["raw_data"])
                    pbar.update(1)

                write_data(
                    f"output_data_{desc}/total_results/",
                    pd.concat(param_data, ignore_index=True),
                    f"model_output_initinf_{param_set['initial_infected']}_r0_{param_set['r0']}_a_{round(param_set['a'], 4)}_exposhet_{param_set['exposure_heterogeneity']}.csv"
                )
    write_data(
        f"output_data_{desc}/fit_results/",
        pd.DataFrame(fitting_results),
        "fit_results.csv"
    )

def run_model_for_mcmc(params, general_param_set, iters_per_param_set, observed_data, desc=""):
    """
    @param fit_params_list: params list, in order r0, r0_s_w, nc_nh, exp_het, sigma
    @param general_param_set: a dictionary of general parameters for the model
    @param iters_per_param_set: total number of iterations per mcmc iteration
    @param observed_data: The observed data to compare to
    @param pool: Pool to do parallel processing with
    """
    theta = params[:-1]
    sigma = params[-1]

    # raw_model_output = []
    fitting_results = []
    combined_param_set = combine_fit_and_general_param_set(theta, general_param_set)
    for iter_val in range(iters_per_param_set):
        res = run_with_params_sse(
            run_num = iter_val,
            params_to_fit = combined_param_set,
            observed_data = observed_data,
        )
        fitting_results.append(res["sse_summary"])
        # raw_model_output.append(res["raw_data"])
    
    lp = log_prior(theta, sigma)
    if not isfinite(lp):
        return -inf
    
    ll = calc_log_likelihood(fitting_results, sigma, observed_data.shape[0])
    
    # write_data(
    #     f"output_data_{desc}/total_results/",
    #     pd.concat(raw_model_output, ignore_index=True),
    #     f"model_output_initinf_{general_param_set['initial_infected']}_nhnc_{round(1/theta[2], 2)}_r0_{theta[0]}_r0sw_{round(theta[1], 2)}_exposhet_{theta[3]}.csv"
    # )

    return lp + ll

def fit_model_mcmc(init_thetas, init_sigma, general_input_params, num_walkers, iters_per_param_set, desc, observed_data):
    ndim = len(init_thetas) + 1
    nwalkers = num_walkers
    rng = np.random.default_rng(7281998)

    # Initial guess for parameters
    init_theta = [list(d.values())[0] for d in init_thetas]

    # Start walkers in a small ball around initial guess
    p0 = np.array(init_theta + [init_sigma]) + 1e-2*rng.standard_normal((nwalkers, ndim))
    
    num_cpus = cpu_count()
    with Pool(processes=num_cpus-2) as pool:
        log_post = partial(
            run_model_for_mcmc, 
            general_param_set = general_input_params,
            observed_data=observed_data, 
            iters_per_param_set=iters_per_param_set, 
            desc=desc
        )
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post, pool=pool)

        # Burn-in
        state = sampler.run_mcmc(p0, 2000, progress=True)
        sampler.reset()

        # Production run
        sampler.run_mcmc(state, 5000, progress=True)

        # Extract chains once
        chains = sampler.get_chain()  # shape: (nsteps, nwalkers, ndim)
        flat_samples = chains.reshape(-1, chains.shape[-1])

        # Posterior summaries
        theta_samples = flat_samples[:, :-1]
        sigma_samples = flat_samples[:, -1]

        print("Posterior mean theta:", theta_samples.mean(axis=0))
        print("Posterior median sigma:", np.median(sigma_samples))

        # Save burn-in and production separately
        cols = [list(d.keys())[0] for d in init_thetas] + ["sigma"]
        write_data(
            f"output_data_{desc}/mcmc_results/",
            pd.DataFrame(chains[:2000,:,:].reshape(-1, ndim), columns=cols),
            "burnin_samples.csv"
        )
        write_data(
            f"output_data_{desc}/mcmc_results/",
            pd.DataFrame(chains[2000:,:,:].reshape(-1, ndim), columns=cols),
            "final_samples.csv"
        )
