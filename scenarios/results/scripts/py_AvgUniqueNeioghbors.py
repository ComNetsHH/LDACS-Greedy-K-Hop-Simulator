import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.ticker as ticker
import os
from sklearn.metrics.pairwise import euclidean_distances
from multiprocessing import Pool, cpu_count
from itertools import product
import csv
from result_analysis_init import *
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'lmodern',
    # "font.serif": 'Times',
    'font.size': 30,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'text.latex.preamble': r'\usepackage{lmodern}'
})

# Settings and parameters
area_length, area_side = 3000, 3000
comm_range = 100  # Communication range in km
densities = np.linspace(0.0002, 0.0005, 30)  # Node densities in node/km^2
repititions = 50
k = 4  # the subset size
# k = 6 
path = 'AverageCaptureRatio'
filename_2hop_fraction=f'AvgCaptureRatio-R{comm_range}km-thr0.707-m={k}'
filename_number_neighbors=f'AvgNumberOfNeighbors-R{comm_range}km-thr0.707-m={k}'
# Specify the output directory and filename
output_directory = 'AverageCaptureRatio'
output_filename = f'AvgCaptureRatio-R{comm_range}km-thr0.707-m={k}.csv'
full_path = os.path.join(output_directory, output_filename)
# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

colors = ["#298C8C", "#A00000", "#B8B8B8"] # Teal, Red and Gray

def calculate_pairwise_distances(points, selected_points):
    # Convert points and selected_points to NumPy arrays
    points_array = np.array([p['pos'] for p in points])
    selected_points_array = np.array([sp['pos'] for sp in selected_points])

    # Compute pairwise distances
    return euclidean_distances(points_array, selected_points_array)

def select_random_neighbors(neighbors, k):
    """Select up to k unique elements randomly from a set of neighbors."""
    neighbors_list = list(neighbors)  # Convert the set to a list
    selected_neighbors = []
    for _ in range(min(k, len(neighbors))):
        chosen = random.choice(neighbors_list)
        selected_neighbors.append(chosen)
        neighbors_list.remove(chosen)  # Remove the selected neighbor
    return selected_neighbors

def select_FFT_neighbors(points, m):
    """Select up to k unique nodes from a set of neighbors in graph G using standard Farthest-First Traversal."""
    if not points:
        return []
    if len(points) <= m:
        return [point['id'] for point in points if 'id' in point]
    selected_points = []
    points = np.array(points)  # Ensure points is a NumPy array for easier manipulation

    selected_indices = set()  # Keep track of selected indices instead of deleting
    random_index = random.randint(0, len(points) - 1)
    selected_indices.add(random_index)
    selected_points.append(points[random_index])

    while len(selected_points) < m:
        # Only consider points that have not been selected yet
        remaining_indices = [i for i in range(len(points)) if i not in selected_indices]
        if not remaining_indices:
            break  # Break if there are no remaining points to consider

        remaining_points = points[remaining_indices]
        # Calculate distances from remaining points to the selected points
        distances_to_selected = calculate_pairwise_distances(remaining_points, selected_points)
        min_distances = np.min(distances_to_selected, axis=1)

        # Select the farthest point candidate
        farthest_point_idx = np.argmax(min_distances)
        selected_idx = remaining_indices[farthest_point_idx]
        selected_indices.add(selected_idx)
        selected_points.append(points[selected_idx])

    # Extract the list of ids
    selected_points_ids = [point['id'] for point in selected_points if 'id' in point]
    return selected_points_ids

def select_FFT_neighbors_remaining(points, m, preselected_points):
    """Select up to k unique nodes from a set of neighbors in graph G using standard Farthest-First Traversal."""
    # print(f"We want to select {m - len(preselected_points)} nodes from a pool of {len(points)} nodes.")
    if not points:
        return []
    if len(points) <= m - len(preselected_points):
        return [point['id'] for point in points if 'id' in point]
    selected_points = []
    combined_points = preselected_points + [p for p in points if p not in preselected_points]
    points = np.array(points)  # Ensure points is a NumPy array for easier manipulation
    preselected_points = np.array(preselected_points)  
    combined_points = np.array(combined_points)

    
    selected_indices = set()  # Keep track of selected indices instead of deleting

    # Loop through each index of preselected_points within combined_points
    for index in range(len(preselected_points)):
        # Directly add the preselected point's index and position
        selected_indices.add(index)
        selected_points.append(combined_points[index])

    while len(selected_points) < m:
        # Only consider points that have not been selected yet
        remaining_indices = [i for i in range(len(combined_points)) if i not in selected_indices]
        if not remaining_indices:
            break  # Break if there are no remaining points to consider

        remaining_points = combined_points[remaining_indices]
        # Calculate distances from remaining points to the selected points
        distances_to_selected = calculate_pairwise_distances(remaining_points, selected_points)
        min_distances = np.min(distances_to_selected, axis=1)

        # Select the farthest point candidate
        farthest_point_idx = np.argmax(min_distances)
        selected_idx = remaining_indices[farthest_point_idx]
        selected_indices.add(selected_idx)
        selected_points.append(combined_points[selected_idx])
    # Extract the list of ids
    selected_points_ids = [point['id'] for point in selected_points if point not in preselected_points and 'id' in point]
    return selected_points_ids

def select_EFFT_neighbors(neighbor_attr, points, m, R=100):
    """Select up to k unique nodes from a set of neighbors in graph G using standard Farthest-First Traversal."""
    threshold = 1 / np.sqrt(2) * R
    if not points:
        return []
    if len(points) <= m:
        return [point['id'] for point in points if 'id' in point]
    
    neighbor_pos = np.array(neighbor_attr['pos'])
    points = np.array(points)  # Ensure points is a NumPy array for easier manipulation
    selected_indices = set()
    selected_points = []

    random_index = random.randint(0, len(points) - 1)
    selected_indices.add(random_index)
    selected_points.append(points[random_index])

    while len(selected_points) < m:
        remaining_indices = [i for i in range(len(points)) if i not in selected_indices]
        if not remaining_indices:
            break  # Break if there are no remaining points to consider

        remaining_points = points[remaining_indices]
        distances_to_selected = calculate_pairwise_distances(remaining_points, selected_points)
        min_distances = np.min(distances_to_selected, axis=1)

        farthest_point_idx = np.argmax(min_distances)
        selected_idx = remaining_indices[farthest_point_idx]
        farthest_point = points[selected_idx]

        # Assuming farthest_point and neighbor_pos are both numpy arrays 
        distance_to_center = np.linalg.norm(np.array(farthest_point['pos']) - neighbor_pos)

        distance_to_selected_points = np.min(calculate_pairwise_distances([farthest_point], selected_points), axis=1)[0]
        if distance_to_selected_points > threshold and distance_to_center > 1 / np.sqrt(2) * threshold:
            selected_indices.add(selected_idx)
            selected_points.append(farthest_point)
        else:
            break

    selected_points_ids = [point['id'] for point in selected_points if 'id' in point]
    return selected_points_ids

def run_simulation(density, rep):
    # Insert the body of your simulation here.
    num_nodes = int(density * area_length * area_side)
    G = nx.Graph()
    pos = [(random.uniform(0, area_length), random.uniform(0, area_side)) for _ in range(num_nodes)]
    for i in range(num_nodes):
        G.add_node(i, pos=pos[i], id=i)
    for i, pos1 in enumerate(pos):
        for j, pos2 in enumerate(pos[i + 1:], start=i + 1):
            if np.linalg.norm(np.array(pos1) - np.array(pos2)) <= comm_range:
                G.add_edge(i, j)

    # Filter nodes from G.nodes() based on position criteria
    h_fil = 3.5
    filtered_nodes = [node for node, attrs in G.nodes(data=True) if (0 + h_fil * comm_range) <= attrs['pos'][0] <= (area_length - h_fil * comm_range) and
                                                                    (0 + h_fil * comm_range) <= attrs['pos'][1] <= (area_side - h_fil * comm_range)]
    num_nodes = len(filtered_nodes)

    total_one_hop = 0
    total_unique_two_hop = 0
    total_unique_random_two_hop = 0
    total_unique_fft_two_hop = 0
    total_unique_efft_two_hop = 0
    total_unique_three_hop= 0
    total_unique_efft_three_hop = 0

    for node in filtered_nodes:
        # print(f"Node ID: {node}, Position: {G.nodes[node]['pos']}")
        neighbors = set(nx.all_neighbors(G, node))
        all_two_hop_neighbors = []
        all_random_two_hop_neighbors = []
        all_fft_two_hop_neighbors = []
        all_efft_two_hop_neighbors = []
        all_three_hop_neighbors = []
        all_efft_three_hop_neighbors = []

        for neighbor in neighbors:
            neighbor_neighbors = set(nx.all_neighbors(G, neighbor))
            neighbor_neighbors_attr = [G.nodes[i] for i in neighbor_neighbors]
            neighbor_attr = G.nodes[neighbor]
            unique_two_hop = neighbor_neighbors - neighbors - {node}
            local_three_hop_neighbors = []
            local_efft_three_hop_neighbors = []

            # For each 2-hop neighbor, find their neighbors (3-hop)
            for two_hop_neighbor in unique_two_hop:
                neighbor_neighbors_neighbors = set(nx.all_neighbors(G, two_hop_neighbor))
                neighbor_neighbors_neighbors_attr = [G.nodes[i] for i in neighbor_neighbors_neighbors]
                neighbor_neighbor_attr = G.nodes[two_hop_neighbor]
                unique_three_hop = neighbor_neighbors_neighbors - unique_two_hop - neighbors - {node}
                efft_three_hop_neighbors = select_EFFT_neighbors(neighbor_neighbor_attr, neighbor_neighbors_neighbors_attr, k, R=comm_range)
                local_efft_three_hop_neighbors.extend(efft_three_hop_neighbors)
                all_three_hop_neighbors.extend(unique_three_hop)
                local_three_hop_neighbors.extend(unique_three_hop)
            
            unique_local_efft_three_hop_neighbors = set(local_efft_three_hop_neighbors)
            local_three_hop_neighbors = set(local_three_hop_neighbors)
            local_efft_three_hop_neighbors_attrs = [G.nodes[i] for i in unique_local_efft_three_hop_neighbors]
            # print("This is me: ", unique_local_efft_three_hop_neighbors)


            random_neighbors = set(select_random_neighbors(neighbor_neighbors, k))
            fft_neighbors = set(select_FFT_neighbors(neighbor_neighbors_attr, k))
            efft_neighbors = set(select_EFFT_neighbors(neighbor_attr, neighbor_neighbors_attr, k, R=comm_range))
            efft_neighbors_attrs = [G.nodes[i] for i in efft_neighbors]
            
            unique_random_two_hop = random_neighbors - neighbors - {node}
            unique_fft_two_hop = fft_neighbors - neighbors - {node}
            unique_efft_two_hop = efft_neighbors - neighbors - {node}
            unique_efft_three_hop = set()
            
            k_remaining = 0
            if len(efft_neighbors) > 0:
                k_remaining = max(0, k - len(efft_neighbors))
                if k_remaining > 0:
                    if len(local_efft_three_hop_neighbors_attrs) > 0:
                        efft_neighbors_three = set(select_FFT_neighbors_remaining(local_efft_three_hop_neighbors_attrs, k, efft_neighbors_attrs))
                        unique_efft_three_hop = efft_neighbors_three - neighbors - {node}

                        for element in unique_efft_three_hop.copy():
                            if element in unique_efft_two_hop:
                                continue  # Skip if the element is already in unique_efft_two_hop
                            if element in unique_two_hop:
                                unique_efft_two_hop.add(element)
                                unique_efft_three_hop.discard(element)  # Remove the element from unique_efft_three_hop

            all_two_hop_neighbors.extend(unique_two_hop)
            all_random_two_hop_neighbors.extend(unique_random_two_hop)
            all_fft_two_hop_neighbors.extend(unique_fft_two_hop)
            all_efft_two_hop_neighbors.extend(unique_efft_two_hop)
            all_efft_three_hop_neighbors.extend(unique_efft_three_hop)

        unique_two_hop_count = len(set(all_two_hop_neighbors))
        unique_random_two_hop_count = len(set(all_random_two_hop_neighbors))
        unique_fft_two_hop_count = len(set(all_fft_two_hop_neighbors))
        unique_efft_two_hop_count = len(set(all_efft_two_hop_neighbors))
        unique_three_hop_count = len(set(all_three_hop_neighbors))
        all_efft_three_hop_neighbors = [element for element in all_efft_three_hop_neighbors if element in all_three_hop_neighbors]
        unique_efft_three_hop_count = len(set(all_efft_three_hop_neighbors))

        total_one_hop += G.degree(node) 
        total_unique_two_hop += unique_two_hop_count
        total_unique_random_two_hop += unique_random_two_hop_count
        total_unique_fft_two_hop += unique_fft_two_hop_count
        total_unique_efft_two_hop += unique_efft_two_hop_count
        total_unique_three_hop += unique_three_hop_count
        total_unique_efft_three_hop += unique_efft_three_hop_count

    # Add to the total for each repetition
    avg_unique_rand_ratio = (total_unique_random_two_hop / total_unique_two_hop
                             if total_unique_two_hop else 0)
    avg_unique_fft_ratio = (total_unique_fft_two_hop / total_unique_two_hop
                            if total_unique_two_hop else 0)
    avg_unique_efft_ratio = (total_unique_efft_two_hop / total_unique_two_hop
                             if total_unique_two_hop else 0)
    avg_unique_efft_3hop_ratio = (total_unique_efft_three_hop / total_unique_three_hop
                             if total_unique_three_hop else 0)
    avg_1_hop = total_one_hop / num_nodes
    avg_2_hop = total_unique_two_hop / num_nodes
    avg_3_hop = total_unique_three_hop / num_nodes

    print(
        f"Completed: Density {round(density, 5)}, "
        f"Repetition {rep}, "
        f"avg_rand {round(avg_unique_rand_ratio, 2)}, "
        f"avg_fft {round(avg_unique_fft_ratio, 2)}, "
        f"avg_efft {round(avg_unique_efft_ratio, 2)}, "
        f"avg_efft_3hop {round(avg_unique_efft_3hop_ratio, 2)}, "
        f"avg_1_hop: {round(avg_1_hop, 1)}, "  
        f"avg_2_hop: {round(avg_2_hop, 1)}, "
        f"avg_3_hop: {round(avg_3_hop, 1)}"
    )

    # Return the results including the repetition number.
    return density, rep, avg_unique_rand_ratio, avg_unique_fft_ratio, avg_unique_efft_ratio, avg_unique_efft_3hop_ratio, avg_1_hop, avg_2_hop, avg_3_hop


if __name__ == "__main__":
    parameters = list(product(densities, range(repititions)))

    # ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    ncpus = cpu_count() # if not on the cluster you should do this instead

    print(f"cpus_per_task: {ncpus}")  # output is coherent with my slurm script

    # with Pool() as pool:
    with Pool(ncpus) as pool:
        print(f"cpus_per_task: {ncpus}")
        # results = pool.map(run_simulation, parameters)
        results = pool.starmap(run_simulation, parameters, chunksize=10)
        

    # Initialize a more detailed structure for storing results
    aggregated_results = {density: {rep: {'rand': None, 'fft': None, 'efft': None, 'efft_3hop': None, '1hop': None, '2hop': None, '3hop': None} for rep in range(repititions)} for
                          density in densities}

    # Populate this structure with results
    for density, rep, rand_ratio, fft_ratio, efft_ratio, efft_3hop_ratio, avg_1_hop, avg_2_hop, avg_3_hop in results:
        aggregated_results[density][rep]['rand'] = rand_ratio
        aggregated_results[density][rep]['fft'] = fft_ratio
        aggregated_results[density][rep]['efft'] = efft_ratio
        aggregated_results[density][rep]['efft_3hop'] = efft_3hop_ratio
        aggregated_results[density][rep]['1hop'] = avg_1_hop
        aggregated_results[density][rep]['2hop'] = avg_2_hop
        aggregated_results[density][rep]['3hop'] = avg_3_hop

    # Initialize structure for final averages
    final_averages = {density: {'rand': 0, 'fft': 0, 'efft': 0, 'efft_3hop': 0, '1hop': 0, '2hop': 0, '3hop': 0} for density in densities}

    # Compute averages for each density
    for density in densities:
        total_reps = repititions  # Assuming you know the count ahead, otherwise calculate dynamically
        for rep in range(total_reps):
            final_averages[density]['rand'] += aggregated_results[density][rep]['rand']
            final_averages[density]['fft'] += aggregated_results[density][rep]['fft']
            final_averages[density]['efft'] += aggregated_results[density][rep]['efft']
            final_averages[density]['efft_3hop'] += aggregated_results[density][rep]['efft_3hop']
            final_averages[density]['1hop'] += aggregated_results[density][rep]['1hop']
            final_averages[density]['2hop'] += aggregated_results[density][rep]['2hop']
            final_averages[density]['3hop'] += aggregated_results[density][rep]['3hop']

        # Divide by total number of repetitions to get the average
        final_averages[density]['rand'] /= total_reps
        final_averages[density]['fft'] /= total_reps
        final_averages[density]['efft'] /= total_reps
        final_averages[density]['efft_3hop'] /= total_reps
        final_averages[density]['1hop'] /= total_reps
        final_averages[density]['2hop'] /= total_reps
        final_averages[density]['3hop'] /= total_reps
    # Define the fieldnames (column headers) for the CSV
    headers = ['Density', 'Avg_Rand', 'MoE_Rand', 'Avg_FFT', 'MoE_FFT', 'Avg_EFFT', 'MoE_EFFT', 'Avg_EFFT_3HOP', 'MoE_EFFT_3HOP', 'Avg_1HOP', 'MoE_1HOP', 'Avg_2HOP', 'MoE_2HOP', 'Avg_3HOP', 'MoE_3HOP']

    # Open the CSV file for writing
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()  # Write the header row

        # Write data rows
        for density in final_averages:
            # Collect all results for this density into lists
            all_rand = [aggregated_results[density][rep]['rand'] for rep in range(repititions)]
            all_fft = [aggregated_results[density][rep]['fft'] for rep in range(repititions)]
            all_efft = [aggregated_results[density][rep]['efft'] for rep in range(repititions)]
            all_efft_3hop = [aggregated_results[density][rep]['efft_3hop'] for rep in range(repititions)]
            all_1hop = [aggregated_results[density][rep]['1hop'] for rep in range(repititions)]
            all_2hop = [aggregated_results[density][rep]['2hop'] for rep in range(repititions)]
            all_3hop = [aggregated_results[density][rep]['3hop'] for rep in range(repititions)]

            print(all_rand)
            mean_rand, _,  moe_rand = confidence_interval_init(all_rand, confidence=0.95)
            mean_fft, _, moe_fft = confidence_interval_init(all_fft, confidence=0.95)
            mean_efft, _, moe_efft = confidence_interval_init(all_efft, confidence=0.95)
            mean_efft_3hop, _, moe_efft_3hop = confidence_interval_init(all_efft_3hop, confidence=0.95)
            mean_1hop, _, moe_1hop = confidence_interval_init(all_1hop, confidence=0.95)
            mean_2hop, _, moe_2hop = confidence_interval_init(all_2hop, confidence=0.95)
            mean_3hop, _, moe_3hop = confidence_interval_init(all_3hop, confidence=0.95)

            writer.writerow({
                'Density': density,
                'Avg_Rand': mean_rand, 'MoE_Rand': moe_rand,
                'Avg_FFT': mean_fft, 'MoE_FFT': moe_fft,
                'Avg_EFFT': mean_efft, 'MoE_EFFT': moe_efft,
                'Avg_EFFT_3HOP': mean_efft_3hop, 'MoE_EFFT_3HOP': moe_efft_3hop,
                'Avg_1HOP': mean_1hop, 'MoE_1HOP': moe_1hop,
                'Avg_2HOP': mean_2hop, 'MoE_2HOP': moe_2hop,
                'Avg_3HOP': mean_3hop, 'MoE_3HOP': moe_3hop
            })

    print(f"Saved final averages to {full_path}")

    # Initialize lists to hold your data
    densities = []
    final_avg_unique_rand_neighbor_ratios, moe_rand = [], []
    final_avg_unique_fft_neighbor_ratios, moe_fft = [], []
    final_avg_unique_efft_neighbor_ratios, moe_efft = [], []
    final_avg_unique_efft_neighbor_3hop_ratios, moe_efft_3hop = [], []
    final_avg_unique_1hop, moe_1hop = [], []
    final_avg_unique_2hop, moe_2hop = [], []
    final_avg_unique_3hop, moe_3hop = [], []

    # Read data from CSV
    with open(full_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            densities.append(float(row['Density']))
            final_avg_unique_rand_neighbor_ratios.append(float(row['Avg_Rand']))
            final_avg_unique_fft_neighbor_ratios.append(float(row['Avg_FFT']))
            final_avg_unique_efft_neighbor_ratios.append(float(row['Avg_EFFT']))
            final_avg_unique_efft_neighbor_3hop_ratios.append(float(row['Avg_EFFT_3HOP']))
            final_avg_unique_1hop.append(float(row['Avg_1HOP']))
            final_avg_unique_2hop.append(float(row['Avg_2HOP']))
            final_avg_unique_3hop.append(float(row['Avg_3HOP']))
            moe_rand.append(float(row['MoE_Rand']))
            moe_fft.append(float(row['MoE_FFT']))
            moe_efft.append(float(row['MoE_EFFT']))
            moe_efft_3hop.append(float(row['MoE_EFFT_3HOP']))
            moe_1hop.append(float(row['MoE_1HOP']))
            moe_2hop.append(float(row['MoE_2HOP']))
            moe_3hop.append(float(row['MoE_3HOP']))

    # Ensure densities are sorted along with their corresponding values (important if the data isn't already sorted)
    densities = np.array(densities)
    sorted_indices = np.argsort(densities)
    densities = densities[sorted_indices]

    avg_rand = np.array(final_avg_unique_rand_neighbor_ratios)[sorted_indices]
    avg_rand_3hop = np.zeros_like(densities)
    avg_fft = np.array(final_avg_unique_fft_neighbor_ratios)[sorted_indices]
    avg_fft_3hop = np.zeros_like(densities)
    avg_efft = np.array(final_avg_unique_efft_neighbor_ratios)[sorted_indices]
    avg_efft_3hop = np.array(final_avg_unique_efft_neighbor_3hop_ratios)[sorted_indices]
    avg_1hop = np.array(final_avg_unique_1hop)[sorted_indices]
    avg_2hop = np.array(final_avg_unique_2hop)[sorted_indices]
    avg_3hop = np.array(final_avg_unique_3hop)[sorted_indices]
    moe_rand = np.array(moe_rand)[sorted_indices]
    moe_rand_3hop = np.zeros_like(densities)
    moe_fft = np.array(moe_fft)[sorted_indices]
    moe_fft_3hop = np.zeros_like(densities)
    moe_efft = np.array(moe_efft)[sorted_indices]
    moe_efft_3hop = np.array(moe_efft_3hop)[sorted_indices]
    moe_1hop = np.array(moe_1hop)[sorted_indices]
    moe_2hop = np.array(moe_2hop)[sorted_indices]

    # Plotting results
    # Set figure size
    plt.figure(figsize=(12, 9))
    capsize=4
    markersize=9
    lw=5
    FFT_linestyle = 'solid'
    EFFT_linestyle = (0, (0.5, 0.5))
    Random_linestyle = (0, (1.5, 1.5))
    legend_info = [
    ('FFT', colors[2], '', FFT_linestyle),
    ('EFFT', colors[0], '', EFFT_linestyle),
    ('Random', colors[1], '', Random_linestyle),
    # Add more legend entries as needed
    ]

    # Creating a separate error bar for legend of 2-hop and 3-hop
    err_2_hop = plt.errorbar([0], [0], yerr=[0.1], fmt='o', ecolor='black', markersize=markersize, markeredgecolor='black', markerfacecolor='none', lw=lw, capsize=capsize, label='2-Hop')
    err_3_hop = plt.errorbar([0], [0], yerr=[0.1], fmt='s', ecolor='black', markersize=markersize, markeredgecolor='black', markerfacecolor='none', lw=lw, capsize=capsize, label='3-Hop')

    # Plot data with error bars
    plt.errorbar(densities, avg_fft, yerr=moe_fft, label='FFT (2-hop)', linestyle=FFT_linestyle, fmt='o', markersize=markersize, markeredgecolor='black', lw=lw, capsize=capsize, color=colors[2])
    plt.errorbar(densities, avg_fft_3hop, yerr=moe_fft_3hop, label='FFT (3-hop)', linestyle=FFT_linestyle, fmt='s', markersize=markersize, markeredgecolor='black', lw=lw, capsize=capsize, color=colors[2])
    plt.errorbar(densities, avg_efft, yerr=moe_efft, label='EFFT (2-hop)', linestyle=EFFT_linestyle,  fmt='o', markersize=markersize, markeredgecolor='black', lw=lw, capsize=capsize, color=colors[0])
    plt.errorbar(densities, avg_efft_3hop, yerr=moe_efft_3hop, label='EFFT (3-hop)', linestyle=EFFT_linestyle, fmt='s', markersize=markersize, markeredgecolor='black', lw=lw, capsize=capsize, color=colors[0])
    plt.errorbar(densities, avg_rand, yerr=moe_rand, label='Random (2-hop)', linestyle=Random_linestyle, fmt='o', markersize=markersize, markeredgecolor='black', lw=lw, capsize=capsize, color=colors[1])
    plt.errorbar(densities, avg_rand_3hop, yerr=moe_rand_3hop, label='Random (3-hop)', linestyle=Random_linestyle, fmt='s', markersize=markersize, markeredgecolor='black', lw=lw, capsize=capsize, color=colors[1])

    # Set axis labels and tick formatting
    plt.xlabel('Density (nodes/km²)')
    plt.ylabel('Average Capture Ratio')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # Configure y-axis locators
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # Configure x-axis locators
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.00005))
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(0.000025))

    # Set up grid
    plt.gca().xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    plt.gca().yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

    # Create and configure the secondary x-axis
    ax2 = plt.gca().secondary_xaxis('top')
    ax2.set_xlabel('Equipage Fraction ($\\rho$)', labelpad=10)
    ax2.set_xticks([0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005])
    ax2.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])

    # Set x-axis limits
    plt.xlim([0.00025, 0.0005])
    # Set y-axis limits
    plt.ylim([-0.01, 1])

    # Add a legend
    if legend_info:
        legend_patches = [Line2D([0], [0], color=color, marker=marker, linestyle=linestyle, markersize=markersize, label=label, lw=lw, markeredgecolor='black') for label, color, marker, linestyle in legend_info]
        # Add the error bar entry manually to the list of handles
        legend_patches.append(err_2_hop)
        legend_patches.append(err_3_hop)
        plt.legend(handles=legend_patches, loc='best')
    else:
        plt.legend(loc='best')

    # Tight layout for better spacing
    plt.tight_layout()

    # Save plots if the path is specified
    if path and filename_2hop_fraction:
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f"{filename_2hop_fraction}.pdf"), format='pdf', bbox_inches="tight")
        plt.savefig(os.path.join(path, f"{filename_2hop_fraction}.png"), format='png', bbox_inches="tight")

    # Display the plot
    # plt.show()
    plt.close()

    # Plotting results
    # plt.figure(figsize=(8, 7))
    plt.figure(figsize=(12, 9))

    # Plotting results with error bars for 1-hop, 2-hop, and 3-hop neighbors
    plt.errorbar(densities, avg_1hop, yerr=moe_1hop, label='$1$-Hop', fmt='-o', markersize=markersize, lw=lw, capsize=capsize, markeredgecolor='black', color=colors[0])
    plt.errorbar(densities, avg_2hop, yerr=moe_2hop, label='$2$-Hop', fmt='-o', markersize=markersize, lw=lw, capsize=capsize, markeredgecolor='black', color=colors[1])
    plt.errorbar(densities, avg_3hop, yerr=moe_3hop, label='$3$-Hop', fmt='-o', markersize=markersize, lw=lw, capsize=capsize, markeredgecolor='black', color=colors[2])

    # Axis labels and formatting
    plt.xlabel('Density (nodes/km²)')
    plt.ylabel('Number of $k$-Hop Neighbors')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Adjusting tick locators for better tick distribution
    plt.gca().yaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(15))  # Ensuring major ticks at reasonable intervals

    # Grid setup for better readability
    plt.gca().xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    plt.gca().yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

    # Create and configure the secondary x-axis
    ax2 = plt.gca().secondary_xaxis('top')
    ax2.set_xlabel('Equipage Fraction ($\\rho$)', labelpad=10)
    ax2.set_xticks([0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005])
    ax2.set_xticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1"])

    # Adding a legend
    plt.legend(loc='best')

    # Setting x-axis limits for clarity
    plt.xlim([0.00025, 0.0005])

    # Ensuring layout is tight for aesthetics
    plt.tight_layout()

    # Saving plots conditionally based on provided path and filename
    if path and filename_number_neighbors:
        os.makedirs(path, exist_ok=True)  # Ensure directory exists
        plt.savefig(os.path.join(path, f'{filename_number_neighbors}.pdf'), format='pdf', bbox_inches="tight")
        plt.savefig(os.path.join(path, f'{filename_number_neighbors}.png'), format='png', bbox_inches="tight")

    # Display the plot
    # plt.show()
    plt.close()