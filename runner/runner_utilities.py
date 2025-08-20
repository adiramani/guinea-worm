import pandas as pd

def extract_observed_data(region="Cb"):
    df = pd.read_csv("/Users/adi/Documents/RVC/Guinea Worm/carter center data/clean/case_data_single_case.csv")

    df = df[df['snu1_detect'] == region].copy()

    df['date_emerge'] = pd.to_datetime(df['date_emerge'])
    df['year'] = df['date_emerge'].dt.year
    df['month'] = df['date_emerge'].dt.month

    df = df[(df['year'] >= 2014) & (df['year'] <= 2017)]

    summary = df.groupby(['year', 'month'], as_index=False)['worm_emerge'].sum()
    summary.rename(columns={'worm_emerge': 'cases'}, inplace=True)
    return(summary)

def calculate_mse(model_data, observed, fit_to_prev=False):
    fit_variable = "emergence_copepod" if not fit_to_prev else "emergences_per_host_copepod"
    
    model_data["year"] = model_data["year"] - 1/12
    trimmed_model_data = model_data[
        ((model_data['year'] >= 15) & (model_data['year'] < 19)) &
        (model_data["measure"] == fit_variable)
    ].copy()
    trimmed_model_data["year"] = round(trimmed_model_data["year"] - min(trimmed_model_data["year"]), 2)
    
    observed["year"] = round(observed["year"] - min(observed["year"]) + ((observed["month"] - 1) / 12), 2)
    if fit_to_prev:
        observed["cases"] = observed["cases"] / 4000 # TODO: Make dynamic?
    
    merged_data = pd.merge(trimmed_model_data, observed, left_on='year', right_on='year')
    diff = (observed["cases"]) - merged_data["value"]
    merged_data['mse'] = ((diff) ** 2) # * np.where(diff < 1, 1.5, 1)
    return(sum(merged_data["mse"]))

def build_intervention_info(start_time_years=10, end_time_years=11, name="teathering", value=0):
    return {
        "start_time": start_time_years * 360,
        "end_time": end_time_years * 360,
        "intervention_name": name,
        "intervention_value": value,
    }