import pandas as pd
from result_analysis import *
# from plot_results import *
import networkx as nx

def parse_if_number(s):
    try: return float(s)
    except: return True if s=="true" else False if s=="false" else s if s else None

def parse_ndarray(s):
    return np.fromstring(s, sep=' ') if s else None

def read_result_multi_variable_multiaircraft_multi_runs(csv_filename, variable_values, variable_name, no_simulation_runs, module_names, kpi_name):
    # Read the CSV file
    result = pd.read_csv(csv_filename)
    # Select and reshape the iteration variables
    iter_vars = result.loc[result.type == 'itervar', ['run', 'attrname', 'attrvalue']]
    iter_vars_pivot = iter_vars.pivot(index='run', columns='attrname', values='attrvalue')
    # Merge the result with the iteration variables
    result_merged = result.merge(iter_vars_pivot, left_on='run', right_index=True, how='outer')
    # Filter the rows based on specified criteria
    output_row_in_table_format = result_merged[
        (getattr(result_merged, variable_name).isin(variable_values)) &
        (result_merged.type == 'scalar') &
        (result_merged.module.isin(module_names)) &
        (result_merged.name == kpi_name)
    ]
    # Initialize the array
    no_elements_variable_values = len(variable_values)
    array_shape = (no_elements_variable_values, no_simulation_runs, len(output_row_in_table_format) // (no_elements_variable_values * no_simulation_runs))
    result_array = np.zeros(array_shape)
    # Iterate over each commRange in variable_values and then each run
    for i, variable_value in enumerate(variable_values):
        variable_value_df = output_row_in_table_format[output_row_in_table_format[variable_name] == variable_value]
        unique_runs = variable_value_df['run'].unique()
        for j, run in enumerate(unique_runs):
            run_df = variable_value_df[variable_value_df['run'] == run]
            result_array[i, j, :len(run_df)] = run_df['value'].values

    return result_array

def read_result_multi_modules_aggregated(csv_filename, variable_values, variable_name, no_simulation_runs, module_names, kpi_name):
    # this function is useful when number of aircraft is variable
    # Read the CSV file
    result = pd.read_csv(csv_filename)
    # Select and reshape the iteration variables
    iter_vars = result.loc[result.type == 'itervar', ['run', 'attrname', 'attrvalue']]
    iter_vars_pivot = iter_vars.pivot(index='run', columns='attrname', values='attrvalue')
    # Merge the result with the iteration variables
    result_merged = result.merge(iter_vars_pivot, left_on='run', right_index=True, how='outer')
    # Filter the rows based on specified criteria
    output_row_in_table_format = result_merged[
        (getattr(result_merged, variable_name).isin(variable_values)) &
        (result_merged.type == 'scalar') &
        (result_merged.module.isin(module_names)) &
        (result_merged.name == kpi_name)
    ]
    # Initialize the 2D array
    no_elements_variable_values = len(variable_values)
    array_shape = (no_elements_variable_values, no_simulation_runs)
    result_array = np.zeros(array_shape)

    # Create a copy to avoid the warning and ensure independent modifications
    result_df = output_row_in_table_format.copy()
    # Extract aircraft and application numbers
    result_df['aircraft'] = result_df['module'].str.extract(r'aircraft\[(\d+)]').astype(int)
    # result_df['app'] = result_df['module'].str.extract(r'app\[(\d+)]').astype(int)

    # Pivot the table to sum values across all aircraft and applications for each run and variable
    pivot_df = result_df.pivot_table(
        index=['run', variable_name],
        values='value',
        aggfunc='sum'
    ).reset_index()
    # Extract the numeric part between the hyphens
    pivot_df['run_id'] = pivot_df['run'].str.extract(r'-(\d+)-')
    # Convert the extracted part to an integer
    pivot_df['run_id'] = pivot_df['run_id'].astype(int)

    # Create a mapping for variable_values
    variable_index = {v: i for i, v in enumerate(variable_values)}

    # Iterate over each unique value in numAircrafts
    for var_value in variable_values:
        # Filter pivot_df for the current variable value
        filtered_df = pivot_df[pivot_df[variable_name] == var_value]

        # Sort filtered_df by run_id
        sorted_df = filtered_df.sort_values(by='run_id')

        # Get the index for the current variable value
        var_idx = variable_index[var_value]

        # Using advanced indexing to fill the array
        # Ensure that the number of runs does not exceed no_simulation_runs
        num_runs = min(len(sorted_df), no_simulation_runs)

        # Get the first 'num_runs' values from sorted_df and place them in the array
        result_array[var_idx, :num_runs] = sorted_df['value'].iloc[:num_runs].to_numpy()
    return result_array

def read_result_vector_averages_multi_modules_aggregated(csv_filename, variable_values, variable_name, no_simulation_runs, module_names, kpi_name):
    result = pd.read_csv(csv_filename)
    # Select and reshape the iteration variables
    iter_vars = result.loc[result.type == 'itervar', ['run', 'attrname', 'attrvalue']]
    iter_vars_pivot = iter_vars.pivot(index='run', columns='attrname', values='attrvalue')
    # Merge the result with the iteration variables
    result_merged = result.merge(iter_vars_pivot, left_on='run', right_index=True, how='outer')
    # Filter the rows based on specified criteria
    output_row_in_table_format = result_merged[
        (getattr(result_merged, variable_name).isin(variable_values)) &
        (result_merged.type == 'vector') &
        (result_merged.module.isin(module_names)) &
        (result_merged.name == kpi_name)
    ]
    # Initialize the 2D array
    no_elements_variable_values = len(variable_values)
    array_shape = (no_elements_variable_values, no_simulation_runs)
    result_array = np.zeros(array_shape)

    # Create a copy to avoid the warning and ensure independent modifications
    result_df = output_row_in_table_format.copy()
    # Extract aircraft and application numbers
    result_df['aircraft'] = result_df['module'].str.extract(r'aircraft\[(\d+)]').astype(int)

    # Convert string values in 'vecvalue' to lists of integers
    result_df['vecvalue'] = result_df['vecvalue'].apply(lambda x: [int(i) for i in x.split()] if isinstance(x, str) else x)

    # Create a DataFrame to store the averages for each group
    averages_df = pd.DataFrame(columns=['run', variable_name, 'average'])

    # Iterate over each group and calculate the average
    for (run, var), group in result_df.groupby(['run', variable_name]):
        concatenated = np.concatenate(group['vecvalue'].dropna().tolist())
        
        # Calculate the average for this group
        average = np.nan if concatenated.size == 0 else np.mean(concatenated)

        # Create a temporary DataFrame for the current group's average and concatenate it
        temp_df = pd.DataFrame({
            'run': [run],
            variable_name: [var],
            'average': [average]
        })
        
        averages_df = pd.concat([averages_df, temp_df], ignore_index=True)
        # # Append the average to the DataFrame
        # averages_df = averages_df.append({
        #     'run': run,
        #     variable_name: var,
        #     'average': average
        # }, ignore_index=True)

    # Create a mapping for variable_values
    variable_index = {v: i for i, v in enumerate(variable_values)}

    # Initialize the result array with NaNs (or zeros if preferable)
    result_array = np.full((len(variable_values), no_simulation_runs), np.nan)

    # Iterate over each unique value in variable_values
    for var_value in variable_values:
        # Filter averages_df for the current variable value
        filtered_df = averages_df[averages_df[variable_name] == var_value]

        # Sort filtered_df by 'run'
        sorted_df = filtered_df.sort_values(by='run')

        # Get the index for the current variable value
        var_idx = variable_index[var_value]

        # Using advanced indexing to fill the array
        # Ensure that the number of runs does not exceed no_simulation_runs
        num_runs = min(len(sorted_df), no_simulation_runs)

        # Get the first 'num_runs' averages from sorted_df and place them in the array
        result_array[var_idx, :num_runs] = sorted_df['average'].iloc[:num_runs].to_numpy()
    return result_array