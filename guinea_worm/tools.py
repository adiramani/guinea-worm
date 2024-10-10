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
                    "measure": "Re",
                    "value": year_pop_stat["Re"],
                })
    return pd.DataFrame(processed_rows)