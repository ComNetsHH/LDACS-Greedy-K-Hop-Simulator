#!/bin/bash -l
#SBATCH -p smp    	 
#SBATCH --ntasks 1           
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 7G  
#SBATCH --array=0-300
#SBATCH --time=5-12:00:00

# Execute simulation
opp_runall -j1 ../out/gcc-release/src/greedy-forwarding scenarioRandom_communication_range.ini -c greedy-forwarding-2hop-fft -n ../src:./:../../inet4/src/:../../ldacs-abstract-radio/src:../../ldacs-abstract-tdma/src:../../intairnet-tracebasedapp/src -r ${SLURM_ARRAY_TASK_ID}

# Exit job
exit7