# LDACS Greedy k-Hop Simulator Makefile
# The LDACS Greedy k-Hop simulator includes an installation script that automates the downloading of 
# components, sets up simulation scenarios, and handles result evaluation and graph generation.
# Copyright (C) 2024 Musab Ahmed, Konrad Fuger
# Institute of Communication Networks, Hamburg University of Technology, Hamburg, Germany.
#
# Licensed under the GNU Lesser General Public License v3.0 or later (LGPLv3+).
# This program is distributed without any warranty; see <https://www.gnu.org/licenses/lgpl-3.0.html> for details.
#
# The license grants the ability to modify and redistribute this software under certain conditions.
# A copy of the LGPLv3 should be included with this program; if not, it is available via the provided URL.
#
# This file is responsible for providing the full toolchain of creating result graphs.
# Each target within this Makefile is dedicated to a specific simulation scenario. Targets are responsible for:
# - Starting the simulations
# - Parsing the resulting .sca and .vec files
# - Plotting the parsed results into .pdf files
#
# Brief comments before each target will describe what the scenario entails.
# Variables specifying parameters for targets may be shared among scenarios; they should be named following the SCENARIO_VARNAME syntax.

# create a virtual Python environment to install all packages that are required for plotting results
install-python-env:
	pipenv install

NUM_CPUS = 4 # number of CPUs to allow OMNeT++ to use

# Build debug versions of the LDACS components and Greedy Forwarding algorithm.
# This target compiles all components with debug symbols and links against the debug versions of dependent libraries.
build-debug-greedy:
	cd ../../omnetpp-5.6.2/workspace/; \
	echo -e "\nBuilding LDACS Abstract Radio Component in Debug Mode"; \
	cd ldacs_abstract_radio/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \
	echo -e "\nBuilding LDACS Abstract TDMA Component in Debug Mode"; \
	cd ldacs_abstract_tdma_mac/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \
	echo -e "\nBuilding Greedy Forwarding Component in Debug Mode"; \
	cd ldacs_greedy_k_hop_routing/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-debug/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-debug/src/ -lINET_dbg -lldacs_abstract_radio_dbg -lldacs_abstract_tdma_mac_dbg; make -j$(NUM_CPUS) MODE=debug; \

# Build release versions of the LDACS components and Greedy Forwarding algorithm.
# This target compiles all components for release, optimizing performance and linking against the release versions of dependent libraries.
build-release-greedy:
	cd ../../omnetpp-5.6.2/workspace/; \
	echo -e "\nBuilding LDACS Abstract Radio Component in Release Mode"; \
	cd ldacs_abstract_radio/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET; make -j8 MODE=release; cd ../..; \
	echo -e "\nBuilding LDACS Abstract TDMA Component in Release Mode"; \
	cd ldacs_abstract_tdma_mac/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET; make -j8 MODE=release; cd ../..; \
	echo -e "\nBuilding Greedy Forwarding Component in Release Mode"; \
	cd ldacs_greedy_k_hop_routing/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-release/src/ -lINET -lldacs_abstract_radio -lldacs_abstract_tdma_mac; make -j$(NUM_CPUS) MODE=release

# Build debug versions of the LDACS components and Dijkstra algorithm.
# This target compiles all components with debug symbols and links against the debug versions of dependent libraries.
build-debug-dijkstra:
	cd ../../omnetpp-5.6.2/workspace/; \
	echo -e "\nBuilding LDACS Abstract Radio Component in Debug Mode"; \
	cd ldacs_abstract_radio/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \
	echo -e "\nBuilding LDACS Abstract TDMA Component in Debug Mode"; \
	cd ldacs_abstract_tdma_mac/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET_dbg; make -j8 MODE=debug; cd ../..; \
	echo -e "\nBuilding Greedy Forwarding Component in Debug Mode"; \
	cd ldacs_dijkstra_routing/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-debug/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-debug/src/ -lINET_dbg -lldacs_abstract_radio_dbg -lldacs_abstract_tdma_mac_dbg; make -j$(NUM_CPUS) MODE=debug; \

# Build release versions of the LDACS components and Dijkstra algorithm.
# This target compiles all components for release, optimizing performance and linking against the release versions of dependent libraries.
build-release-dijkstra:
	cd ../../omnetpp-5.6.2/workspace/; \
	echo -e "\nBuilding LDACS Abstract Radio Component in Release Mode"; \
	cd ldacs_abstract_radio/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET; make -j8 MODE=release; cd ../..; \
	echo -e "\nBuilding LDACS Abstract TDMA Component in Release Mode"; \
	cd ldacs_abstract_tdma_mac/src; opp_makemake --make-so -f --deep -KINET_PROJ=../../inet4 -DINET_IMPORT -I../../inet4/src -L../../inet4/src -lINET; make -j8 MODE=release; cd ../..; \
	echo -e "\nBuilding Dijkstra Component in Release Mode"; \
	cd ldacs_dijkstra_routing/src; opp_makemake -f --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-release/src/ -lINET -lldacs_abstract_radio -lldacs_abstract_tdma_mac; make -j$(NUM_CPUS) MODE=release

build-debug:
	make build-debug-greedy; \
	make build-debug-dijkstra

build-release:
	make build-release-greedy; \
	make build-release-dijkstra

# GLOBAL VARIABLES
# Define paths for the release and debug binaries of the ldacs_greedy_k_hop_routing simulation.
SIM_BINARY_GREEDY_RELEASE = ../../omnetpp-5.6.2/workspace/ldacs_greedy_k_hop_routing/out/gcc-release/src/ldacs_greedy_k_hop_routing
SIM_BINARY_GREEDY_DBG = ../../omnetpp-5.6.2/workspace/ldacs_greedy_k_hop_routing/out/gcc-debug/src/ldacs_greedy_k_hop_routing_dbg

# Define paths for the release and debug binaries of the ldacs_dijkstra_routing simulation.
SIM_BINARY_DIJKSTRA_RELEASE = ../../omnetpp-5.6.2/workspace/ldacs_dijkstra_routing/out/gcc-release/src/ldacs_dijkstra_routing
SIM_BINARY_DIJKSTRA_DBG = ../../omnetpp-5.6.2/workspace/ldacs_dijkstra_routing/out/gcc-debug/src/ldacs_dijkstra_routing_dbg

# Include directories for .ned files across various components of the simulation environment.
INCL_GREEDY = -n ../:./:../../omnetpp-5.6.2/workspace/inet4/src/:../../omnetpp-5.6.2/workspace/ldacs_abstract_radio/src:../../omnetpp-5.6.2/workspace/ldacs_abstract_tdma_mac/src:../../omnetpp-5.6.2/workspace/ldacs_greedy_k_hop_routing/src
INCL_DIJKSTRA = -n ../:./:../../omnetpp-5.6.2/workspace/inet4/src/:../../omnetpp-5.6.2/workspace/ldacs_abstract_radio/src:../../omnetpp-5.6.2/workspace/ldacs_abstract_tdma_mac/src:../../omnetpp-5.6.2/workspace/ldacs_dijkstra_routing/src

# END-TO-END SIMULATIONS
# Definition of variable names for each simulation scenario.
GREEDY_FORWARDING_1HOP = greedy-forwarding-1hop
GREEDY_FORWARDING_2HOP = greedy-forwarding-2hop
GREEDY_FORWARDING_FFT_m_4 = greedy-forwarding-fft-m-4
GREEDY_FORWARDING_EFFT_m_4 = greedy-forwarding-efft-m-4
GREEDY_FORWARDING_RANDOM_m_4 = greedy-forwarding-random-m-4  
GREEDY_FORWARDING_FFT_m_6 = greedy-forwarding-fft-m-6
GREEDY_FORWARDING_EFFT_m_6 = greedy-forwarding-efft-m-6
GREEDY_FORWARDING_RANDOM_m_6 = greedy-forwarding-random-m-6  
DIJKSTRA = dijkstra 

# Notes:
# - Use these variables to refer to specific simulation binaries and include paths in your Makefile targets.
# - Ensure that each target that uses these variables properly handles the paths and includes for compiling and linking.

# Create a symbolic link named 'results' to the directory on an HPC cluster where simulation results are stored.
# This target checks if a file or link named 'results' already exists to avoid overwriting or duplicating links.
result-directory-hpc:
	@if [ ! -L simresults ] && [ ! -e simresults ]; then \
		ln -s /path/to/your/hpc/results/directory simresults; \
		echo "Symbolic link created to HPC simresults directory."; \
	else \
		echo "The symbolic link 'simresults' already exists or the name is used by another file."; \
	fi

# Uncomment lines in the .ned file related to Dijkstra
uncomment-dijkstra:
	@sed -i '/^\/\/ import dspr.Dspr;/s|^\/\/ ||' ../scenarioNAC.ned; \
	sed -i '/^\/\/ import dspr.NodeManager;/s|^\/\/ ||' ../scenarioNAC.ned; \
	sed -i 's|^        // nodeManager: NodeManager;|        nodeManager: NodeManager;|' ../scenarioNAC.ned; \

# Comment lines in the .ned file related to Dijkstra
comment-dijkstra:
	@sed -i 's|^import dspr.Dspr;|// import dspr.Dspr;|' ../scenarioNAC.ned
	sed -i 's|^import dspr.NodeManager;|// import dspr.NodeManager;|' ../scenarioNAC.ned
	sed -i 's|^        nodeManager: NodeManager;|        // nodeManager: NodeManager;|' ../scenarioNAC.ned


# Run the greedy-forwarding-1hop communication range simulation on local PC.
greedy-forwarding-1hop-equipageFraction-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_1HOP) $(INCL_GREEDY) 

# Debug the greedy-forwarding-1hop communication range simulation on local PC.
greedy-forwarding-1hop-equipageFraction-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_1HOP) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding-1hop communication range simulation.
greedy-forwarding-1hop-equipageFraction-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_1HOP); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the greedy-forwarding-2hop communication range simulation on local PC.
greedy-forwarding-2hop-equipageFraction-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_2HOP) $(INCL_GREEDY) 

# Debug the greedy-forwarding-2hop communication range simulation on local PC.
greedy-forwarding-2hop-equipageFraction-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_2HOP) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding-2hop communication range simulation.
greedy-forwarding-2hop-equipageFraction-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_2HOP); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the greedy-forwarding using FFT with m=4 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-fft-m-4-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_FFT_m_4) $(INCL_GREEDY) 

# Debug the greedy-forwarding using FFT with m=4 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-fft-m-4-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_FFT_m_4) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding using FFT with m=4 equipage fraction simulation.
greedy-forwarding-equipageFraction-fft-m-4-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_FFT_m_4); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the greedy-forwarding using EFFT with m=4 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-efft-m-4-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_EFFT_m_4) $(INCL_GREEDY) 

# Debug the greedy-forwarding using EFFT with m=4 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-efft-m-4-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_EFFT_m_4) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding using EFFT with m=4 equipage fraction simulation.
greedy-forwarding-equipageFraction-efft-m-4-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_EFFT_m_4); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the greedy-forwarding using Random with m=4 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-random-m-4-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_RANDOM_m_4) $(INCL_GREEDY) 

# Debug the greedy-forwarding using Random with m=4 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-random-m-4-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_RANDOM_m_4) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding using Random with m=4 equipage fraction simulation.

greedy-forwarding-equipageFraction-random-m-4-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_RANDOM_m_4); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the greedy-forwarding using FFT with m=6 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-fft-m-6-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_FFT_m_6) $(INCL_GREEDY) 

# Debug the greedy-forwarding using FFT with m=6 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-fft-m-6-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_FFT_m_6) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding using FFT with m=6 equipage fraction simulation.
greedy-forwarding-equipageFraction-fft-m-6-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_FFT_m_6); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the greedy-forwarding using EFFT with m=6 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-efft-m-6-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_EFFT_m_6) $(INCL_GREEDY) 

# Debug the greedy-forwarding using EFFT with m=6 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-efft-m-6-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_EFFT_m_6) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding using EFFT with m=6 equipage fraction simulation.
greedy-forwarding-equipageFraction-efft-m-6-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_EFFT_m_6); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the greedy-forwarding using Random with m=6 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-random-m-6-A2G:
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_GREEDY_RELEASE) ../scenario_equipage_fraction.ini -c $(GREEDY_FORWARDING_RANDOM_m_6) $(INCL_GREEDY) 

# Debug the greedy-forwarding using Random with m=6 equipage fraction simulation on local PC.
greedy-forwarding-equipageFraction-random-m-6-A2G-debug:
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(GREEDY_FORWARDING_RANDOM_m_6) $(INCL_GREEDY)

# Generate CSV files from results of the greedy-forwarding using Random with m=6 equipage fraction simulation.
greedy-forwarding-equipageFraction-random-m-6-A2G_csv:
	cd simresults/scenario_EquipageFraction/$(GREEDY_FORWARDING_RANDOM_m_6); \
	echo -e "Generate csv files for greedy-forwarding scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run the dijkstra routing equipage fraction simulation on local PC.
dijkstra-equipageFraction-A2G:
    make uncomment-dijkstra; \
	opp_runall -j$(NUM_CPUS) $(SIM_BINARY_DIJKSTRA_RELEASE) ../scenario_equipage_fraction.ini -c $(DIJKSTRA) $(INCL_GREEDY); \
    make comment-dijkstra 

# Debug the dijkstra equipage fraction simulation on local PC.
dijkstra-equipageFraction-debug:
    make uncomment-dijkstra; \
	gdbgui --args $(SIM_BINARY_GREEDY_DBG) ../scenario_equipage_fraction.ini -u Cmdenv -c $(DIJKSTRA) $(INCL_GREEDY); \
    make comment-dijkstra

# Generate CSV files from results of the dijkstra equipage fraction simulation.
dijkstra-equipageFraction_csv:
	cd simresults/scenario_EquipageFraction/$(DIJKSTRA); \
	echo -e "Generate csv files for dijkstra scenario"; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:count")' -o packet_sent_count_multiapp.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("packetReceived:count")' -o packet_received_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].app[*]) AND name("packetSent:sum(packetBytes)")' -o packet_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentBytes:sum")' -o beacon_sentBytes_sum.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("beaconSentCount:count")' -o beacon_sent_count.csv numAircrafts\=*.sca; \
	scavetool export -f 'module(scenarioRandom_forwarding.groundStation[0].app[0]) AND name("endToEndDelay:vector")' -o e2e_delay_vector.csv numAircrafts\=*.vec; \
	scavetool export -f 'module(scenarioRandom_forwarding.aircraft[*].routing) AND name("hopCount:vector")' -o hop_count_vector.csv numAircrafts\=*.vec

# Run all simulations from the HPC cluster command range.
run-all-simulations-from-hpc-equipageFraction:
	make uncomment-dijkstra; \
	sleep 10; \
	sbatch scripts/s_dijkstra_equipage_fraction.sh; \
	sleep 600; \
	sbatch scripts/s_greedy_1hop_equipage_fraction.sh; \
	# sleep 300; \
	sbatch scripts/s_greedy_2hop_equipage_fraction.sh; \
	# sleep 300; \
	sbatch scripts/s_greedyFFT_m_4_equipage_fraction.sh; \
	# sleep 500; \
	sbatch scripts/s_greedyEFFT_m_4_equipage_fraction.sh; \
	# sleep 300; \
	sbatch scripts/s_greedyRandom_m_4_equipage_fraction.sh; \
	# sleep 300; \
	sbatch scripts/s_greedyFFT_m_6_equipage_fraction.sh; \
	# sleep 300; \
	sbatch scripts/s_greedyEFFT_m_6_equipage_fraction.sh; \
	# sleep 300; \
	sbatch scripts/s_greedyRandom_m_6_equipage_fraction.sh 

# This target aggregates the generation of all CSV files for different configurations of the equipage fraction simulations.
random-generate-all-equipageFraction-csv: dijkstra-equipageFraction_csv greedy-forwarding-1hop-equipageFraction-A2G_csv greedy-forwarding-2hop-equipageFraction-A2G_csv greedy-forwarding-equipageFraction-fft-m-4-A2G_csv greedy-forwarding-equipageFraction-efft-m-4-A2G_csv greedy-forwarding-equipageFraction-random-m-4-A2G_csv greedy-forwarding-equipageFraction-fft-m-6-A2G_csv greedy-forwarding-equipageFraction-efft-m-6-A2G_csv greedy-forwarding-equipageFraction-random-m-6-A2G_csv

# This target removes all non-CSV files from the simulation results directory for the equipage fraction scenarios,
# ensuring that only CSV files are retained for data analysis and storage efficiency.
random-remove-collected-result-files-all-equipageFraction:
	find ./simresults/scenario_EquipageFraction/../ -type f ! -name "*.csv" -exec rm {} +

# Run simulations to calculate the average number of unique k-hop neighbors from the local PC.
run-average-khop-neighbors-simulations:
	pipenv run python3 scripts/py_AvgUniqueNeioghbors.py

# This target runs Python scripts to generate plots for the Packet Delivery Ratio (PDR) and Hop Count (HC) 
# for the Equipage Fraction scenario. Each script processes simulation data to produce visualizations.
plot-equipage-fraction-metrics: run-pdr-plot run-hc-plot

run-pdr-plot:
	pipenv run python3 scripts/py_scenario_EquipageFraction_pdr.py


run-hc-plot:
	pipenv run python3 scripts/py_scenario_EquipageFraction_hc.py
