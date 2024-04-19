# [LDACS-Greedy-K-Hop-Simulator](https://github.com/ComNetsHH/LDACS-Greedy-K-Hop-Simulator)

## Overview
The LDACS-Greedy-K-Hop-Simulator is an open-source simulation model developed to enhance and evaluate geographic greedy routing in sparse LDACS air-to-air networks. It specifically simulates the application of a geographic greedy routing protocol that leverages k-hop neighborhood information. The simulator evaluates three distinct methods for selecting k-hop subsets: a random method, Farthest First Traversal (FFT), and Enhanced Farthest First Traversal (EFFT). These methods are tested within simulated French airspace scenario, allowing for comprehensive analysis of routing efficiency and the impact of different neighbor selection strategies on network performance. The simulation is implemented using OMNeT++ and Python, ensuring detailed and scalable simulation capabilities. Additionally, it is compatible with HPC environments using Slurm, making it suitable for large-scale simulation tasks.

## Components
- **[LDACS Greedy K-Hop Routing](https://github.com/ComNetsHH/LDACS-Greedy-K-Hop-Routing)**: Implements the greedy routing algorithm with a subset of k-hop neighborhood information for efficient forwarding in sparse environments.
- **[LDACS Abstract TDMA MAC](https://github.com/ComNetsHH/LDACS-Abstract-TDMA-MAC)**: Models an abstract LDACS air-to-air TDMA-based MAC protocol.
- **[LDACS Abstract Radio](https://github.com/ComNetsHH/LDACS-Abstract-Radio)**: Abstract model of LDACS radio communications for accurate air-to-air transmission simulation.

## Installation
Clone the repository and set up the simulation environment with the following commands:
```bash
git clone https://github.com/yourusername/LDACS-Greedy-K-Hop-Simulator.git
cd LDACS-Greedy-K-Hop-Simulator
./install.sh  # Installs all dependencies and sets up necessary submodules
```

## Configuration of Environment Variables for OMNeT++

To run simulations with OMNeT++, you need to ensure that the OMNeT++ binaries are included in your system's PATH. You have two options:

1. **Permanent PATH Configuration**:
   Add OMNeT++ to your system's environment variables permanently:
   - For Linux/MacOS, add the following line to your `~/.bashrc` or `~/.zshrc`:
     ```bash
     export PATH="/path/to/omnetpp-5.6.2/bin:$PATH"
     ```

2. **Temporary PATH Configuration**:
   You can also configure the PATH temporarily for each session where you run simulations:
   ```bash
   cd omnetpp-5.6.2/
   WORKDIR=$(pwd)
   export PATH=$(pwd)/bin:$PATH


## System Dependencies
Before running the project, ensure you have all necessary system dependencies installed. For Ubuntu/Linux users, `dvipng` is required for rendering plots with Matplotlib when LaTeX is used. Install it using the following command:
```bash
sudo apt-get install dvipng
```

# Scenario Evaluation Guide

## Average PDR and HC with varying Equipage Fractions 

### Overview
In this experiment, we examine the impact of varying equipage fractions on the average packet delivery ratio and hop count. The equipage fraction, denoted by ρ, represents the proportion of aircraft equipped with LDACS. This analysis helps in understanding how different levels of equipages can affect the reliability and effectiveness of the Greedy-1, Greedy-FFT, Greedy-EFFT, and Greedy-Random routing algorithms.

### Targets and Execution
You can execute these simulations either locally or on a High-Performance Computing (HPC) environment, depending on your available resources and needs.

#### Local Execution
To run the simulations locally on your PC, use the following targets in the Makefile:

- `greedy-forwarding-equipageFraction-A2G`: Runs the basic Greedy Forwarding simulation for different equipage fractions.
- `greedy-forwarding-equipageFraction-fft-m-4-A2G`: Runs the Greedy-FFT algorithm with m=4.
- `greedy-forwarding-equipageFraction-efft-m-4-A2G`: Runs the Greedy-EFFT algorithm with m=4.
- `greedy-forwarding-equipageFraction-random-m-4-A2G`: Runs the Greedy-Random algorithm with m=4.

Similarly, add targets for m=6 variants if applicable.

Example command to run a specific simulation:
```bash
cd scenarios/results
make greedy-forwarding-equipageFraction-A2G
```

#### High-Performance Computing (HPC) Execution
For large-scale simulations, using an HPC environment is recommended. This allows for faster execution times and handling of more extensive data sets.

- `run-all-simulations-from-hpc-equipageFraction`: This target submits batch jobs for all Greedy Forwarding scenarios to an HPC system using the `sbatch` command.

Example command to run all simulations on HPC:
```bash
cd scenarios/results
make run-all-simulations-from-hpc-equipageFraction
```
### Results and Analysis
Targets in the Makefile that analyze the results:
- `random-generate-all-equipageFraction-csv`: Creates CSV files for further analysis.
- `plot-equipage-fraction-metrics`: Generates detailed plots for Packet Delivery Ratio (PDR) and Hop Count from the CSV files.

Example commands:
```bash
cd scenarios/results
make -j16 random-generate-all-equipageFraction-csv
make plot-equipage-fraction-metrics
```

## Average Capture Ratio with varying Equipage Fractions 

### Overview
The Average Capture Ratio experiment investigates the efficiency of the three proposed selection methods (Random, FFT, EFFT) within a network by measuring the proportion of 2nd and 3rd order neighbors identified by each selection method against the total number of neighbors present. This ratio is calculated for each node and then averaged across the entire network to evaluate the performance of each method under different levels of LDACS equipage. 

### Targets and Execution
You can execute this simulation locally on your PC.

To run the simulations locally on your PC and plot the results, use the following targets in the Makefile:

- `run-average-khop-neighbors-simulations`: Run simulations and plot the results of the average number of unique k-hop neighbors from the local PC.

Example commands:
```bash
cd scenarios/results
make run-average-khop-neighbors-simulations
```
