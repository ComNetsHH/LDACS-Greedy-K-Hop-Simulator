//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
// 

import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.ManetRouter;
import inet.node.inet.StandardHost;
import inet.physicallayer.contract.packetlevel.IRadioMedium;
import inet.visualizer.integrated.IntegratedMultiVisualizer;
import inet.visualizer.contract.IIntegratedVisualizer;
import inet.node.ethernet.EtherSwitch;
import inet.common.scenario.ScenarioManager;
import ned.DatarateChannel;
import ldacs_abstract_tdma.scheduler.AbstractLdacsTdmaScheduler;
import ldacs_abstract_radio.LdacsAbstractRadio;
import greedy_forwarding.Greedy;


network scenarioRandom
{
    parameters:
        int numAircrafts; // Number of aircrafts, set this in your .ini file
        int numGroundStations = default(1);  // Set a default value
		@statistic[sumBeaconMsg](source=count(beacon); record=count);
		@statistic[sumPacketSent](source=count(packetSent); record=count);
		@statistic[sumAppPacketSent](source=count(appPacketSent); record=count);
		
        @display("bgb=10000000,10000000;bgl=2");
        @figure[title](type=label; pos=0,-1; anchor=sw; color=darkblue);
    submodules:
        scenarioManager: ScenarioManager;
        //		responsible of assigning slots to the aircrafts based in their transmission buffer		
        scheduler: AbstractLdacsTdmaScheduler {
            parameters:
                @display("p=100,400;is=s");
        }
        
        radioMedium: <default("UnitDiskRadioMedium")> like IRadioMedium {
            @display("p=580,275");
        }

        visualizer: IntegratedMultiVisualizer {
            @display("p=100,200");
        }
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,100;is=s");
        }

        aircraft[numAircrafts]: ManetRouter {
        }

        groundStation[numGroundStations]: ManetRouter {
        }
}

network scenarioRandom_forwarding extends scenarioRandom
{
    types:
        channel Eth1G extends DatarateChannel
        {
            parameters: 
            datarate = 1Gbps;
            delay = 0s;
            ber = 0;
        }
    submodules:
        switchGS: EtherSwitch {
            parameters:
                @display("p=1150000,380000");
            gates: 
                ethg[numAircrafts + numGroundStations]; // Number of aircrafts + ground stations
        }
    connections allowunconnected:
        for i=0..numAircrafts-1 {
            aircraft[i].ethg++ <--> Eth1G <--> switchGS.ethg++ ;
        }
        groundStation[0].ethg++ <--> Eth1G <--> switchGS.ethg++ ;
}