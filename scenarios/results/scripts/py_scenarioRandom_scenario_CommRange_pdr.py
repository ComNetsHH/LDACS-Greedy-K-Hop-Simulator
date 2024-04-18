from py_performence_evaluation import *

# Define variables
variable_values = ['250', '300', '350', '400', '450', '500']
x_data = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
# variable_values = ['300', '350', '400', '450', '500']
# x_data = [0.6, 0.7, 0.8, 0.9, 1]
no_simulation_runs = 50
max_no_of_aircraft = 500
# variable_values = ['250', '350', '400', '450', '500']
# x_data = [0.5, 0.7, 0.8, 0.9, 1]

dir_path = './results/simresults/scenarioRandom/scenario_CommRange=100km_FR=2.5km_BI=5s_thr_0.707'
plots_dir_path = './results/scenarioRandom/scenario_CommRange=100km_FR=2.5km_BI=5s_thr_0.707'
file = 'pdr-greedy-numAircrafts'
# file = 'pdr-greedy-numAircrafts-Equipage_0.5-1_Greedy-1'
# dir_path = './results/simresults/scenarioRandom/scenario_CommRange=90km_FR=2.5km_BI=5s_thr_0.707'
# plots_dir_path = './results/scenarioRandom/scenario_CommRange=90km_FR=2.5km_BI=5s_thr_0.707'
file = 'pdr-greedy-numAircrafts-Equipage_0.5-1-fft'

variable_name='numAircrafts'

# Process data for different strategies
# strategies = ['Greedy-1', 'Greedy-2', f'Greedy-EFFT ($m=4$)', f'Greedy-Random ($m=4$)', '3-Hop', f'Greedy-EFFT ($m=6$)', f'Greedy-Random ($m=6$)']
# strategies = [f'Greedy-1', f'Greedy-EFFT ($m=4$)', f'Greedy-Random ($m=4$)', f'Greedy-EFFT ($m=6$)', f'Greedy-Random ($m=6$)']
strategies = [f'Greedy-1', f'Greedy-FFT ($m=4$)', f'Greedy-EFFT ($m=4$)', f'Greedy-Random ($m=4$)', f'Greedy-FFT ($m=6$)', f'Greedy-EFFT ($m=6$)', f'Greedy-Random ($m=6$)']
# strategies = [f'Greedy-1']

mean_rates = {}
margin_errors = {}
csv_filenames_received = {
    'Greedy-1': f'{dir_path}/greedy-forwarding/packet_received_count.csv',
    'GPSR': f'{dir_path}/gpsr-forwarding/packet_received_count.csv',
    'Greedy-2': f'{dir_path}/greedy-forwarding-2hop/packet_received_count.csv',
    f'Greedy-FFT ($m=4$)': f'{dir_path}/greedy-forwarding-2hop-fft/packet_received_count.csv',
    f'Greedy-EFFT ($m=4$)': f'{dir_path}/greedy-forwarding-2hop-efft/packet_received_count.csv',
    f'Greedy-Random ($m=4$)': f'{dir_path}/greedy-forwarding-2hop-randomSubset/packet_received_count.csv',
    '3-Hop': f'{dir_path}/greedy-forwarding-3hop/packet_received_count.csv',
    f'Greedy-FFT ($m=6$)': f'{dir_path}/greedy-forwarding-3hop-fft/packet_received_count.csv',
    f'Greedy-EFFT ($m=6$)': f'{dir_path}/greedy-forwarding-3hop-efft/packet_received_count.csv',
    f'Greedy-Random ($m=6$)': f'{dir_path}/greedy-forwarding-3hop-randomSubset/packet_received_count.csv'
}

csv_filenames_sent = {
    'Greedy-1': f'{dir_path}/greedy-forwarding/packet_sent_count_multiapp.csv',
    'GPSR': f'{dir_path}/gpsr-forwarding/packet_sent_count_multiapp.csv',
    'Greedy-2': f'{dir_path}/greedy-forwarding-2hop/packet_sent_count_multiapp.csv',
    f'Greedy-FFT ($m=4$)': f'{dir_path}/greedy-forwarding-2hop-fft/packet_sent_count_multiapp.csv',
    f'Greedy-EFFT ($m=4$)': f'{dir_path}/greedy-forwarding-2hop-efft/packet_sent_count_multiapp.csv',
    f'Greedy-Random ($m=4$)': f'{dir_path}/greedy-forwarding-2hop-randomSubset/packet_sent_count_multiapp.csv',
    '3-Hop': f'{dir_path}/greedy-forwarding-3hop/packet_sent_count_multiapp.csv',
    f'Greedy-FFT ($m=6$)': f'{dir_path}/greedy-forwarding-3hop-fft/packet_sent_count_multiapp.csv',
    f'Greedy-EFFT ($m=6$)': f'{dir_path}/greedy-forwarding-3hop-efft/packet_sent_count_multiapp.csv',
    f'Greedy-Random ($m=6$)': f'{dir_path}/greedy-forwarding-3hop-randomSubset/packet_sent_count_multiapp.csv'
}

module_name_packet_received =['scenarioRandom_forwarding.groundStation[0].app[0]']
module_names_packet_sent = [f'scenarioRandom_forwarding.aircraft[{aircraft}].app[0]' for aircraft in range(max_no_of_aircraft)]
for index, strategy in enumerate(strategies):
    print(f"currently running for strategy: {strategy}")
    csv_filename_received = os.path.abspath(csv_filenames_received[strategy])
    csv_filename_sent = os.path.abspath(csv_filenames_sent[strategy])
    mean_rates[strategy], margin_errors[strategy] = pdr_read_and_process_data(csv_filename_received,
                                                                              csv_filename_sent,
                                                                              module_name_packet_received, 
                                                                              module_names_packet_sent, 
                                                                              variable_values, 
                                                                              no_simulation_runs, 
                                                                              variable_name)
    print(f"PDR: {mean_rates[strategy]}")


# # Reverse the order of x_data and corresponding data for plotting
# x_data = x_data[::-1]
# for strategy in strategies:
#     mean_rates[strategy] = mean_rates[strategy][::-1]
#     margin_errors[strategy] = margin_errors[strategy][::-1]

# Plot settings
xlabel = r'Equipage Fraction $(\rho$)'
ylabel = 'Packet Delivery Ratio'
title = ''

# colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]
# colors = ["#5E4C5F", "#999999", "#FFBB6F"] # Dull purple, Med grey and Gold
# colors = ["#298C8C", "#A00000", "#B8B8B8"] # Teal, Red and Gray
# colors = ["#298C8C", "#A00000", "#B8B8B8", "#6A0DAD"]  # Added purple
colors = ["#FFBB6F", "#A00000", "#B8B8B8"] # Gold, Red and Gray


# style_combinations = {
#     strategies[0]: [colors[2], ''],    # Red color with diagonal hatching
#     strategies[1]: [colors[1], 'xx'],  # Blue color with back diagonal hatching
#     strategies[2]: [colors[0], 'xx'],   # Green color with cross hatching
#     strategies[3]: [colors[1], 'ooo'],   # Green color with cross hatching
#     strategies[4]: [colors[0], 'ooo'],   # Green color with cross hatching
# }
# style_combinations = {
#     strategies[0]: [colors[2], ''],    # Red color with diagonal hatching
#     strategies[1]: [colors[2], 'xx'],  # Blue color with back diagonal hatching
#     strategies[2]: [colors[1], 'xx'],   # Green color with cross hatching
#     strategies[3]: [colors[0], 'xx'],   # Green color with cross hatching
#     strategies[4]: [colors[2], 'ooo'],   # Green color with cross hatching
#     strategies[5]: [colors[1], 'ooo'],   # Green color with cross hatching
#     strategies[6]: [colors[0], 'ooo']   # Green color with cross hatching
# }
# style_combinations = {
#     strategies[0]: [colors[1], ''],    # Red color with diagonal hatching
#     strategies[1]: [colors[2], 'xxxx'],  # Blue color with back diagonal hatching
#     strategies[2]: [colors[2], '-'],   # Green color with cross hatching
#     strategies[3]: [colors[2], 'o'],   # Green color with cross hatching
#     strategies[4]: [colors[0], 'xxxx'],   # Green color with cross hatching
#     strategies[5]: [colors[0], '-'],   # Green color with cross hatching
#     strategies[6]: [colors[0], 'o']   # Green color with cross hatching
# }
# style_combinations = {
#     strategies[0]: [colors[2], ''],    # Red color with diagonal hatching
# }

# legend_info = [
#     ('Greedy-1', colors[1], ''),
#     ('Greedy ($m=4$)', colors[2], ''),
#     ('Greedy ($m=6$)', colors[0], ''),
#     ('FFT', 'none', 'xxxx'),
#     ('EFFT', 'none', '-'),
#     ('Random', 'none', 'o'),
#     # Add more legend entries as needed
# ]
style_combinations = {
    strategies[0]: [colors[1], ''],    # Red color with diagonal hatching
    strategies[1]: [colors[2], '///'],  # Blue color with back diagonal hatching
    strategies[2]: [colors[2], '-'],   # Green color with cross hatching
    strategies[3]: [colors[2], 'o'],   # Green color with cross hatching
    strategies[4]: [colors[0], '///'],   # Green color with cross hatching
    strategies[5]: [colors[0], '-'],   # Green color with cross hatching
    strategies[6]: [colors[0], 'o']   # Green color with cross hatching
}
legend_info = [
    ('Greedy-1', colors[1], ''),
    ('Greedy ($m=4$)', colors[2], ''),
    ('Greedy ($m=6$)', colors[0], ''),
    ('FFT', 'none', '///'),
    ('EFFT', 'none', '-'),
    ('Random', 'none', 'o'),
    # Add more legend entries as needed
]

# Call the plotting function
# plot_error_bar(mean_rates, margin_errors, strategies, x_data, xlabel, ylabel, plots_dir_path, file, width=0.3, figsize=(8, 6), style_combinations=style_combinations)
plot_error_bar(mean_rates, margin_errors, strategies, x_data, xlabel, ylabel, plots_dir_path, file, width=0.1, figsize=(12, 8), style_combinations=style_combinations, enable_legend=True, capsize=2, legend_info=legend_info, bar_spacing=0.025)
