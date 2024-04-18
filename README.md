# [LDACS-Greedy-K-Hop-Simulator](https://github.com/ComNetsHH/LDACS-Greedy-K-Hop-Simulator)

## Overview
The LDACS-Greedy-K-Hop-Simulator is an open-source toolkit developed to simulate and enhance geographic greedy routing in sparse LDACS air-to-air networks. It specifically simulates the application of a geographic greedy routing protocol that leverages k-hop neighborhood data. The simulator evaluates three distinct methods for selecting k-hop subsets: a random method, Farthest First Traversal (FFT), and Enhanced Farthest First Traversal (EFFT). These methods are tested within simulated French airspace scenarios, allowing for comprehensive analysis of routing efficiency and the impact of different neighbor selection strategies on network performance. The simulation is implemented using OMNeT++ and Python, ensuring detailed and scalable simulation capabilities. Additionally, it is compatible with HPC environments using Slurm, making it suitable for large-scale simulation tasks.

## Components
- **[LDACS Greedy K-Hop Routing](https://github.com/ComNetsHH/LDACS-Greedy-K-Hop-Routing)**: Advanced algorithm for efficient network traffic management in sparse environments.
- **[LDACS Abstract TDMA MAC](https://github.com/ComNetsHH/LDACS-Abstract-TDMA-MAC)**: Models an LDACS-specific MAC protocol using time-division multiple access.
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
   cd path/to/omnetpp-5.6.2/
   export PATH=$(pwd)/bin:$PATH


## System Dependencies
Before running the project, ensure you have all necessary system dependencies installed. For Ubuntu/Linux users, `dvipng` is required for rendering plots with Matplotlib when LaTeX is used. Install it using the following command:
```bash
sudo apt-get install dvipng
```