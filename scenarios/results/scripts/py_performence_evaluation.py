import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from py_plot_functions import plot_error_bar
from result_analysis_init import *

import seaborn as sns
# sys.path.append("../../..")
from result_analysis_init import *
from py_plot_functions import *
import os

# def pdr_read_and_process_data(csv_filename_packet_received, 
#                           csv_filename_packet_sent, 
#                           module_name_packet_received,
#                           module_names_packet_sent,
#                           variable_values, 
#                           no_simulation_runs, 
#                           variable_name,
#                           no_of_aircraft=371,
#                           no_of_data_applications=10):
    
#     # module_name_packet_received =['greedy_forwarding.groundStation[0].app[0]']
#     # module_names_packet_sent = [f'greedy_forwarding.aircraft[{aircraft}].app[{app}]' 
#     #                       for aircraft in range(no_of_aircraft) 
#     #                       for app in range(no_of_data_applications)]
#     kpi_name_packet_received='packetReceived:count'
#     kpi_name_packet_sent='packetSent:count'

#     packet_sent_sum = read_result_multi_variable_multiaircraft_multiapps_multi_runs(csv_filename_packet_sent, 
#                                                                                     variable_values, 
#                                                                                     variable_name, 
#                                                                                     no_of_data_applications, 
#                                                                                     no_simulation_runs, 
#                                                                                     module_names_packet_sent, 
#                                                                                     kpi_name_packet_sent)
#     # Apply np.squeeze() if the fourth dimension size is 1, otherwise sum along the third and fourth axes
#     if packet_sent_sum.shape[3] == 1:
#         packet_sent_sum = np.squeeze(packet_sent_sum, axis=3)
#         packet_sent_sum = np.sum(packet_sent_sum, axis=2)
#     else:
#         packet_sent_sum = np.sum(packet_sent_sum, axis=(2, 3))
#     # Read and process the specified data
#     packet_received_sum = read_result_multi_variable_multiaircraft_multi_runs(csv_filename_packet_received, variable_values, variable_name, no_simulation_runs, module_name_packet_received, kpi_name_packet_received)
#     # Apply np.squeeze() if the third dimension size is 1, otherwise sum along the third axis
#     if packet_received_sum.shape[2] == 1:
#         packet_received_sum = np.squeeze(packet_received_sum, axis=2)
#     else:
#         packet_received_sum = np.sum(packet_received_sum, axis=2)
#     pdr = np.divide(packet_received_sum, packet_sent_sum)
#     mean, _, margin_of_error = confidence_interval_init(pdr, confidence=0.95)
#     return mean, margin_of_error

def pdr_read_and_process_data(csv_filename_packet_received, 
                             csv_filename_packet_sent, 
                             module_name_packet_received,
                             module_names_packet_sent,
                             variable_values, 
                             no_simulation_runs, 
                             variable_name):
    
    # module_name_packet_received =['greedy_forwarding.groundStation[0].app[0]']
    # module_names_packet_sent = [f'greedy_forwarding.aircraft[{aircraft}].app[{app}]' 
    #                       for aircraft in range(no_of_aircraft) 
    #                       for app in range(no_of_data_applications)]
    kpi_name_packet_received='packetReceived:count'
    kpi_name_packet_sent='packetSent:count'

    packet_sent_sum = read_result_multi_modules_aggregated(csv_filename_packet_sent, 
                                                           variable_values, 
                                                           variable_name, 
                                                           no_simulation_runs, 
                                                           module_names_packet_sent, 
                                                           kpi_name_packet_sent)
    # Read and process the specified data
    packet_received_sum = read_result_multi_variable_multiaircraft_multi_runs(csv_filename_packet_received, 
                                                                              variable_values, 
                                                                              variable_name, 
                                                                              no_simulation_runs, 
                                                                              module_name_packet_received, 
                                                                              kpi_name_packet_received)
    # Apply np.squeeze() if the third dimension size is 1, otherwise sum along the third axis
    if packet_received_sum.shape[2] == 1:
        packet_received_sum = np.squeeze(packet_received_sum, axis=2)
    else:
        packet_received_sum = np.sum(packet_received_sum, axis=2)
    pdr = np.divide(packet_received_sum, packet_sent_sum)
    mean, _, margin_of_error = confidence_interval_init(pdr, confidence=0.95)
    return mean, margin_of_error

# def ro_read_and_process_data(csv_filename_beacon_sent, 
#                           csv_filename_packet_sent, 
#                           variable_values, 
#                           no_simulation_runs, 
#                           variable_name):
#     no_of_data_applications = 10
#     no_of_aircraft = 371
#     module_name_beacon_sent =[f'greedy_forwarding.aircraft[{aircraft}].routing' for aircraft in range(no_of_aircraft)]
#     module_names_packet_sent = [f'greedy_forwarding.aircraft[{aircraft}].app[{app}]' 
#                           for aircraft in range(no_of_aircraft) 
#                           for app in range(no_of_data_applications)]
#     kpi_name_beacon_sent='beaconSentBytes:sum'
#     kpi_name_packet_sent='packetSent:sum(packetBytes)'

#     beacon_sent_sum = read_result_multi_variable_multiaircraft_multi_runs(csv_filename_beacon_sent, 
#                                                                           variable_values, 
#                                                                           variable_name, 
#                                                                           no_simulation_runs, 
#                                                                           module_name_beacon_sent, 
#                                                                           kpi_name_beacon_sent)
    
#     packet_sent_sum = read_result_multi_variable_multiaircraft_multiapps_multi_runs(csv_filename_packet_sent, 
#                                                                                     variable_values, 
#                                                                                     variable_name, 
#                                                                                     no_of_data_applications, 
#                                                                                     no_simulation_runs, 
#                                                                                     module_names_packet_sent, 
#                                                                                     kpi_name_packet_sent)
    
#     # Apply np.squeeze() if the third dimension size is 1, otherwise sum along the third axis
#     if beacon_sent_sum.shape[2] == 1:
#         beacon_sent_sum = np.squeeze(beacon_sent_sum, axis=2)
#     else:
#         beacon_sent_sum = np.sum(beacon_sent_sum, axis=2)
    
#     # Apply np.squeeze() if the fourth dimension size is 1, otherwise sum along the third and fourth axes
#     if packet_sent_sum.shape[3] == 1:
#         packet_sent_sum = np.squeeze(packet_sent_sum, axis=3)
#     else:
#         packet_sent_sum = np.sum(packet_sent_sum, axis=(2, 3))

#     ro = np.divide(beacon_sent_sum, (beacon_sent_sum + packet_sent_sum))
#     mean, _, margin_of_error = confidence_interval_init(ro, confidence=0.95)
#     return mean, margin_of_error

def ro_read_and_process_data(csv_filename_beacon_sent, 
                             csv_filename_packet_sent,
                             module_name_beacon_sent,
                             module_names_packet_sent, 
                             variable_values, 
                             no_simulation_runs, 
                             variable_name):
    kpi_name_beacon_sent='beaconSentBytes:sum'
    kpi_name_packet_sent='packetSent:sum(packetBytes)'

    beacon_sent_sum = read_result_multi_modules_aggregated(csv_filename_beacon_sent, 
                                                           variable_values,
                                                           variable_name,
                                                           no_simulation_runs,
                                                           module_name_beacon_sent,
                                                           kpi_name_beacon_sent)
    
    packet_sent_sum = read_result_multi_modules_aggregated(csv_filename_packet_sent, 
                                                           variable_values,
                                                           variable_name,
                                                           no_simulation_runs,
                                                           module_names_packet_sent,
                                                           kpi_name_packet_sent)

    ro = np.divide(beacon_sent_sum, (beacon_sent_sum + packet_sent_sum))
    mean, _, margin_of_error = confidence_interval_init(ro, confidence=0.95)
    return mean, margin_of_error

def hc_read_and_process_data(csv_filename_hop_count, 
                             module_names_hop_count, 
                             variable_values, 
                             no_simulation_runs, 
                             variable_name):
    kpi_name_hop_count = 'hopCount:vector'

    average_hop_count = read_result_vector_averages_multi_modules_aggregated(csv_filename_hop_count, 
                                                                             variable_values,
                                                                             variable_name,
                                                                             no_simulation_runs,
                                                                             module_names_hop_count,
                                                                             kpi_name_hop_count)
    
    mean, _, margin_of_error = confidence_interval_init(average_hop_count, confidence=0.95)
    return mean, margin_of_error

def br_read_and_process_data(strategies,
                             csv_filenames_beacon_sent, 
                             module_name_beacon_sent,
                             variable_values, 
                             no_simulation_runs, 
                             variable_name):
    kpi_name_beacon_sent='beaconSentBytes:sum'
    kpi_name_packet_sent='packetSent:sum(packetBytes)'
    mean = {}
    margin_of_error = {}
    beacon_sent_sums = {}
    beacon_reductions = {}
    
    for index, strategy in enumerate(strategies):
        print(f"currently running for strategy: {strategy}")
        csv_filename_beacon_sent = os.path.abspath(csv_filenames_beacon_sent[strategy])        

        beacon_sent_sums[strategy] = read_result_multi_modules_aggregated(csv_filename_beacon_sent, 
                                                            variable_values,
                                                            variable_name,
                                                            no_simulation_runs,
                                                            module_name_beacon_sent,
                                                            kpi_name_beacon_sent)
    beacon_reductions['2-Hop'] = np.divide((beacon_sent_sums['2-Hop']-beacon_sent_sums['1-Hop']), beacon_sent_sums['1-Hop'])
    beacon_reductions['3-Hop'] = np.divide(np.abs(beacon_sent_sums['3-Hop']-beacon_sent_sums['1-Hop']), beacon_sent_sums['1-Hop'])
    # beacon_reductions['2-hop (FFT)'] = np.divide(np.abs(beacon_sent_sums['2-hop (FFT)']-beacon_sent_sums['2-hop']), beacon_sent_sums['2-hop'])
    beacon_reductions[f'EFFT ($m=4$)'] = np.divide(np.abs(beacon_sent_sums[f'EFFT ($m=4$)']-beacon_sent_sums['2-Hop']), beacon_sent_sums['2-Hop'])
    beacon_reductions[f'2-Hop Random ($m=4$)'] = np.divide(np.abs(beacon_sent_sums[f'2-Hop Random ($m=4$)']-beacon_sent_sums['2-Hop']), beacon_sent_sums['2-Hop'])
    beacon_reductions[f'EFFT ($m=8$)'] = np.divide(np.abs(beacon_sent_sums[f'EFFT ($m=8$)']-beacon_sent_sums['3-Hop']), beacon_sent_sums['3-Hop'])
    beacon_reductions[f'3-Hop Random ($m=8$)'] = np.divide(np.abs(beacon_sent_sums[f'3-Hop Random ($m=8$)']-beacon_sent_sums['3-Hop']), beacon_sent_sums['3-Hop'])

    for index, strategy in enumerate(strategies):
        if strategy != '1-Hop':
            mean[strategy], _, margin_of_error[strategy] = confidence_interval_init(beacon_reductions[strategy] * 100, confidence=0.95)
    return mean, margin_of_error