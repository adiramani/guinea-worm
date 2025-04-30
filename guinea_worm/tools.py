import pandas as pd

def process_data(model_output):
    processed_rows = []
    for data_set in model_output:
        for population in data_set["stats"].keys():
            year_pop_stat = data_set["stats"][population]
            if population == "copepod":
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "infective_larvae",
                    "value": year_pop_stat["infective_larvae"],
                })
            else:    
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "female_worm_prev",
                    "value": year_pop_stat["female_worm_prev"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "total_worm_load",
                    "value": year_pop_stat["total_worm_load_per_person"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "female_worm_load",
                    "value": year_pop_stat["female_worm_load_per_person"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "emergence_copepod",
                    "value": year_pop_stat["emergence_copepod"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "emergence_hosts_copepod",
                    "value": year_pop_stat["emergence_hosts_copepod"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "emergences_per_host_copepod",
                    "value": year_pop_stat["emergences_per_host_copepod"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "emergence_average_worm_age_copepod",
                    "value": year_pop_stat["emergence_average_worm_age_copepod"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "mean_worm_rate",
                    "value": year_pop_stat["mean_worm_rate"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "mating_prob",
                    "value": year_pop_stat["mating_prob"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "mean_age",
                    "value": year_pop_stat["mean_age"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "num_infected_host",
                    "value": year_pop_stat["num_infected_host"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "emergences_per_infected_host_copepod",
                    "value": year_pop_stat["emergences_per_infected_host_copepod"],
                })
                processed_rows.append({
                    "year": data_set["year"],
                    "population": population,
                    "measure": "emergent_host_prevalence_copepod",
                    "value": year_pop_stat["emergent_host_prevalence_copepod"],
                })
    return pd.DataFrame(processed_rows)