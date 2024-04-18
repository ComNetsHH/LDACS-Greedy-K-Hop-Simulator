from py_performence_evaluation import *


# Define variables
variable_values = ['250', '300', '350', '400', '450', '500']
x_data = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
no_simulation_runs = 50
max_no_of_aircraft = 500

dir_path = './simresults/scenario_EquipageFraction'
plots_dir_path = './scenario_EquipageFraction'
file = f'hc-greedy-EquipageFraction_{min(x_data)}-{max(x_data)}'

variable_name='numAircrafts'

# Process data for different strategies
strategies = ['Greedy-1', f'Greedy-FFT ($m=4$)', f'Greedy-EFFT ($m=4$)', f'Greedy-Random ($m=4$)', f'Greedy-FFT ($m=6$)', f'Greedy-EFFT ($m=6$)', f'Greedy-Random ($m=6$)']
mean_rates = {}
margin_errors = {}
csv_filenames_hop_count = {
    'Greedy-1': f'{dir_path}/greedy-forwarding/hop_count_vector.csv',
    f'Greedy-FFT ($m=4$)': f'{dir_path}/greedy-forwarding-fft-m-4/hop_count_vector.csv',
    f'Greedy-EFFT ($m=4$)': f'{dir_path}/greedy-forwarding-efft-m-4/hop_count_vector.csv',
    f'Greedy-Random ($m=4$)': f'{dir_path}/greedy-forwarding-random-m-4/hop_count_vector.csv',
    f'Greedy-FFT ($m=6$)': f'{dir_path}/greedy-forwarding-fft-m-6/hop_count_vector.csv',
    f'Greedy-EFFT ($m=6$)': f'{dir_path}/greedy-forwarding-efft-m-6/hop_count_vector.csv',
    f'Greedy-Random ($m=6$)': f'{dir_path}/greedy-forwarding-random-m-6/hop_count_vector.csv'
}

module_names_hop_count = [f'scenarioRandom_forwarding.aircraft[{aircraft}].routing' for aircraft in range(max_no_of_aircraft)]

for index, strategy in enumerate(strategies):
    print(f"currently running for strategy: {strategy}")
    csv_filename_hop_count = os.path.abspath(csv_filenames_hop_count[strategy])
    mean_rates[strategy], margin_errors[strategy] = hc_read_and_process_data(csv_filename_hop_count,
                                                                              module_names_hop_count, 
                                                                              variable_values, 
                                                                              no_simulation_runs, 
                                                                              variable_name)
    print(f"HC: {mean_rates[strategy]}")

# Plot settings
xlabel = r'Equipage Fraction $(\rho$)'
ylabel = 'Hop Count (hops)'
title = ''

colors = ["#FFBB6F", "#A00000", "#B8B8B8"] # Gold, Red and Gray

style_combinations = {
    strategies[0]: [colors[1], ''], 
    strategies[1]: [colors[2], '///'],
    strategies[2]: [colors[2], '-'],  
    strategies[3]: [colors[2], 'o'], 
    strategies[4]: [colors[0], '///'],  
    strategies[5]: [colors[0], '-'],   
    strategies[6]: [colors[0], 'o']   
}
legend_info = [
    ('Greedy-1', colors[1], ''),
    ('Greedy ($m=4$)', colors[2], ''),
    ('Greedy ($m=6$)', colors[0], ''),
    ('FFT', 'none', '///'),
    ('EFFT', 'none', '-'),
    ('Random', 'none', 'o'),
]

plot_error_bar(mean_rates, margin_errors, strategies, x_data, xlabel, ylabel, plots_dir_path, file, set_ylim=None, width=0.1, figsize=(12, 8), style_combinations=style_combinations, enable_legend=True, capsize=2, legend_info=legend_info, bar_spacing=0.025)
