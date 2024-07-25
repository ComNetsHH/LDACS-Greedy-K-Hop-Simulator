#!/bin/bash

# The LDACS-Greedy-K-Hop-Simulator includes a comprehensive installation script that sets up the simulation environment, downloads necessary components, and configures initial simulation scenarios. It facilitates both the evaluation of results and the generation of graphs to visualize outcomes. This toolkit is designed for efficient simulation and analysis of greedy k-hop routing in LDACS air-to-air communications.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

NUM_CPUS=10  # set to your no. of cores

LOC_OMNET=https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.6.2/omnetpp-5.6.2-src-linux.tgz
LOC_OMNET_MAC=https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.6.2/omnetpp-5.6.2-src-macosx.tgz
LOC_INET=https://github.com/eltayebmusab/inet/archive/refs/tags/v4.2.5.tar.gz
LOC_RADIO=https://zenodo.org/records/12826802/files/ComNetsHH/LDACS-Abstract-Radio-v1.0.5.zip
LOC_TDMA=https://zenodo.org/records/12826801/files/ComNetsHH/LDACS-Abstract-TDMA-MAC-v1.0.5.zip
LOC_GREEDY_ROUTING=https://zenodo.org/records/12826688/files/ComNetsHH/LDACS-Greedy-K-Hop-Routing-v1.0.5.zip
LOC_DIJKSTRA_ROUTING=https://zenodo.org/records/12826320/files/ComNetsHH/LDACS-Dijkstra-v1.0.2.zip

# Download OMNeT++ v5.6.2, unpack and go to directory.
echo -n "Downloading OMNeT++ "
if [ $1 = "mac" ]; then
	echo "for Mac"
	wget $LOC_OMNET_MAC
	echo -e "\n\nUnpacking OMNeT++"
	tar -xvzf omnetpp-5.6.2-src-macosx.tgz
	rm omnetpp-5.6.2-src-macosx.tgz
else
	echo "for Linux"
	wget $LOC_OMNET
	echo -e "\n\nUnpacking OMNeT++"
	tar -xvzf omnetpp-5.6.2-src-linux.tgz
	rm omnetpp-5.6.2-src-linux.tgz
fi
echo -e "\n\nCompiling OMNeT++"
cd omnetpp-5.6.2/
# Set PATH, configure and build.
WORKDIR=$(pwd)
export PATH=${WORKDIR}/bin:$PATH
./configure CC=gcc CXX=g++ WITH_OSG=no WITH_OSGEARTH=no WITH_QTENV=no
make -j$NUM_CPUS MODE=release base
#make -j$NUM_CPUS MODE=debug base

# Download INET, unpack and go to directory.
echo -e "\n\nDownloading INET"
mkdir workspace
cd workspace
wget $LOC_INET
echo -e "\n\nUnpacking INET"
tar -xvzf v4.2.5.tar.gz 
rm -R v4.2.5.tar.gz
mv inet-4.2.5/ inet4
cd inet4/
# Build.
echo -e "\n\nCompiling INET"
make makefiles
make -j$NUM_CPUS MODE=release
#make -j$NUM_CPUS MODE=debug
cd ..

# Compile LDACS Abstract Radio
echo -e "\n\nDownloading LDACS Abstract Radio"
mkdir ldacs_abstract_radio
wget $LOC_RADIO
umask 000
unzip LDACS-Abstract-Radio-v1.0.5.zip -d tmp_extract
mv tmp_extract/*/* ldacs_abstract_radio/
mv tmp_extract/*/.* ldacs_abstract_radio/
rm -r tmp_extract
rm -r LDACS-Abstract-Radio-v1.0.5.zip
cd ldacs_abstract_radio/src
opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET 
# opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I. -I../../inet4/src -L../../inet4/src -lINET_dbg
make MODE=release -j$NUM_CPUS
# make MODE=debug -j$NUM_CPUS
cd ../..

# Compile LDACS Abstract TDMA MAC
echo -e "\n\nDownloading LDACS Abstract TDMA MAC"
mkdir ldacs_abstract_tdma_mac
wget $LOC_TDMA
umask 000
unzip LDACS-Abstract-TDMA-MAC-v1.0.5.zip -d tmp_extract
mv tmp_extract/*/* ldacs_abstract_tdma_mac/
mv tmp_extract/*/.* ldacs_abstract_tdma_mac/
rm -r tmp_extract
rm -r LDACS-Abstract-TDMA-MAC-v1.0.5.zip
cd ldacs_abstract_tdma_mac/src
opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../ldacs_abstract_radio/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -lINET -lldacs_abstract_radio
# opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../ldacs_abstract_radio/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -lINET_dbg -lldacs_abstract_radio_dbg
make MODE=release -j$NUM_CPUS
# make MODE=debug -j$NUM_CPUS
cd ../..

# Compile LDACS-Greedy-K-Hop-Routing
echo -e "\n\nDownloading LDACS-Greedy-K-Hop-Routing"
mkdir ldacs_greedy_k_hop_routing
wget $LOC_GREEDY_ROUTING
umask 000
unzip LDACS-Greedy-K-Hop-Routing-v1.0.5.zip -d tmp_extract
mv tmp_extract/*/* ldacs_greedy_k_hop_routing/
mv tmp_extract/*/.* ldacs_greedy_k_hop_routing/
rm -r tmp_extract
rm -r LDACS-Greedy-K-Hop-Routing-v1.0.5.zip
cd ldacs_greedy_k_hop_routing/src
opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-release/src/ -lINET -lldacs_abstract_radio -lldacs_abstract_tdma_mac
# opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-release/src/ -lINET_dbg -lldacs_abstract_radio_dbg -lldacs_abstract_tdma_mac_dbg
make MODE=release -j$NUM_CPUS
# make MODE=debug -j$NUM_CPUS
cd ../..

# Compile LDACS-Dijkstra
echo -e "\n\nDownloading LDACS-Dijkstra"
mkdir ldacs_dijkstra_routing
wget $LOC_DIJKSTRA_ROUTING
umask 000
unzip LDACS-Dijkstra-v1.0.2.zip -d tmp_extract
mv tmp_extract/*/* ldacs_dijkstra_routing/
mv tmp_extract/*/.* ldacs_dijkstra_routing/
rm -r tmp_extract
rm -r LDACS-Dijkstra-v1.0.2.zip
cd ldacs_dijkstra_routing/src
opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-release/src/ -lINET -lldacs_abstract_radio -lldacs_abstract_tdma_mac
# opp_makemake -f -s --deep -O out -KINET4_PROJ=../../inet4 -DINET_IMPORT -I../../inet4 -I../../ldacs_abstract_radio/src -I../../ldacs_abstract_tdma_mac/src -I. -I../../inet4/src -L../../inet4/src -L../../ldacs_abstract_radio/out/gcc-release/src/ -L../../ldacs_abstract_tdma_mac/out/gcc-release/src/ -lINET_dbg -lldacs_abstract_radio_dbg -lldacs_abstract_tdma_mac_dbg
make MODE=release -j$NUM_CPUS
# make MODE=debug -j$NUM_CPUS
cd ../..


cd ../../scenarios/results
echo -e "\n\nInstall python packages into local pipenv environment"
make install-python-env

