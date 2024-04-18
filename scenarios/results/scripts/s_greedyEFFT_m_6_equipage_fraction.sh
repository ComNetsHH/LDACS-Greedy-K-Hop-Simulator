#!/bin/bash -l
#SBATCH -p smp    	 
#SBATCH --ntasks 1           
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 5G  
#SBATCH --array=0-300
#SBATCH --time=5-12:00:00

# Execute simulation
opp_runall -j1 ../../omnetpp-5.6.2/workspace/ldacs_greedy_k_hop_routing/out/gcc-release/src/ldacs_greedy_k_hop_routing ../scenario_equipage_fraction.ini -c greedy-forwarding-efft-m-6 -n ../:./:../../omnetpp-5.6.2/workspace/inet4/src/:../../omnetpp-5.6.2/workspace/ldacs_abstract_radio/src:../../omnetpp-5.6.2/workspace/ldacs_abstract_tdma_mac/src:../../omnetpp-5.6.2/workspace/ldacs_greedy_k_hop_routing/src -r ${SLURM_ARRAY_TASK_ID}

# Exit job
exit7