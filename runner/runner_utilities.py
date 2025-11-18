import pandas as pd

def extract_observed_data(region="", cluster=-1, fit_measure="emergence_copepod", start_year=2014, end_year=2017):
    df = pd.read_csv("/Users/adi/Documents/RVC/Guinea Worm/carter center data/clean/case_data_single_case_with_clusters.csv")
    if (cluster >= 0):
        df = df[df['cluster'] == cluster].copy()
    elif (region != "" and region is not None):
        df = df[df['snu1_detect'] == region].copy()

    df['date_emerge'] = pd.to_datetime(df['date_emerge'])
    df['year'] = df['date_emerge'].dt.year
    df['month'] = df['date_emerge'].dt.month

    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    if fit_measure in ["emergence_copepod", "emergences_per_host_copepod"]:
        summary = (
            df.groupby(['year', 'month'], as_index=False)['worm_emerge']
              .sum()
              .rename(columns={'worm_emerge': 'cases'})
        )
        if fit_measure == "emergences_per_host_copepod":
            summary["cases"] = summary["cases"] / 5000  # TODO: Make dynamic
        return summary
    elif fit_measure == "emergences_per_infected_host_copepod":
        return (
            df.groupby(['year', 'month', "id_event"], as_index=False)['worm_emerge']
              .sum()
              .groupby(['year', 'month'], as_index=False)['worm_emerge']
              .mean()
              .rename(columns={'worm_emerge': 'cases'})
        )


def calculate_sse(model_data, observed, fit_measure="emergence_copepod", start_year=14):
    """
    @params fit_measure: options are "emergence_copepod", "emergences_per_host_copepod", or "emergences_per_infected_host_copepod"
    """   
    model_data["year"] = model_data["year"] - 1/12
    total_years = len(observed["year"].unique())
    trimmed_model_data = model_data[
        ((model_data['year'] >= start_year) & (model_data['year'] < (start_year + total_years))) &
        (model_data["measure"] == fit_measure)
    ].copy()
    trimmed_model_data["year"] = round(trimmed_model_data["year"] - min(trimmed_model_data["year"]), 2)
    
    observed["year"] = round(observed["year"] - min(observed["year"]) + ((observed["month"] - 1) / 12), 2)
    
    merged_data = pd.merge(trimmed_model_data, observed, left_on='year', right_on='year')
    diff = (observed["cases"] - merged_data["value"])
    merged_data['sse'] = ((diff) ** 2) # * np.where(diff < 1, 1.5, 1)
    return(sum(merged_data["sse"]))

def build_intervention_info(start_time_years=10, end_time_years=11, name="tethering", value=0, months_to_apply=[]):
    return {
        "start_time": start_time_years * 360,
        "end_time": end_time_years * 360,
        "months_to_apply": months_to_apply,
        "intervention_name": name,
        "intervention_value": value,
    }

def combine_fit_and_general_param_set(fit_params_list, general_param_set):
    return {
        **general_param_set,
        "r0": fit_params_list[0],
        "r0_sink_to_worm": fit_params_list[1],
        "nc_nh": 1 / fit_params_list[2],
        "exposure_heterogeneity": fit_params_list[3]
    }