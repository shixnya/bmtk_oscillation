# %%
from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar
import random


net = NetworkBuilder("v1")
ncells_v1 = 1000
net.add_nodes(
    N=ncells_v1,
    positions=positions_columinar(
        N=ncells_v1, center=[0, 50.0, 0], max_radius=30.0, height=100.0
    ),
    pop_name="Scnn1a",
    location="VisL4",
    ei="e",
    model_type="point_process",
    model_template="nest:iaf_psc_alpha",
    dynamics_params="IntFire1_exc_point.json",
)


def connect_prob(source, target, prob):
    if random.random() < prob:
        return 1
    return 0


src_type = {"ei": "e"}
trg_type = {"ei": "e"}

weights = 12.5

cm = net.add_edges(
    source=src_type,
    target=trg_type,
    connection_rule=connect_prob,
    connection_params={"prob": 0.03},
    syn_weight=weights,
    delay=1.0,
    dynamics_params="ExcToExc.json",
    model_template="static_synapse",
)

""" 
# use this section if we need to prepare weight for individual connections
def random_syn_weight(source, target, mu, sigma):
    return random.gauss(mu, sigma)


cm.add_properties(
    "syn_weight",
    rule=random_syn_weight,
    rule_params={
        "mu": 1.0,
        "sigma": 0.0,
    },
    dtypes=float,
)
"""

net.build()
net.save_nodes(output_dir="network")
net.save_edges(output_dir="network")

# %% now the input structure

input = NetworkBuilder("input")
ncells_input = 1000
input.add_nodes(N=ncells_input, pop_name="input", potential="exc", model_type="virtual")
input.add_edges(
    source=input.nodes(),
    target=net.nodes(pop_name="Scnn1a"),
    connection_rule=connect_prob,
    connection_params={"prob": 0.02},
    syn_weight=weights,
    delay=1.0,
    dynamics_params="ExcToExc.json",
    model_template="static_synapse",
)
input.build()
input.save_nodes(output_dir="network")
input.save_edges(output_dir="network")


# %% setting up point net
from bmtk.utils.sim_setup import build_env_pointnet

dt = 0.1  # ms
tstop = 3000.0
# delete existing config files: config.json
import os

# os.remove("config.json")
# os.remove("simulation_config.json")
# os.remove("circuit_config.json")

build_env_pointnet(
    base_dir=".",
    network_dir="network",
    tstop=tstop,
    dt=dt,
    include_examples=True,
)

# append the input section for the simulation_config.json
import json

sim_config = json.load(open("simulation_config.json", "r"))
sim_config["inputs"] = {
    "input_spikes": {
        "input_type": "spikes",
        "module": "sonata",
        "input_file": "input/spikes.h5",
        "node_set": "input",
    }
}

json.dump(sim_config, open("simulation_config.json", "w"), indent=2)


# %% generate random input spikes
input_fr = 100  # Hz
import numpy as np

# steps = int(tstop / dt)
# spike_array = np.random.rand(steps, ncells_input) < input_fr * dt / 1000.0
# store in the sonata format
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator

psg = PoissonSpikeGenerator(population="input")
psg.add(node_ids=range(ncells_input), firing_rate=input_fr, times=(0.0, tstop / 1000.0))
psg.to_sonata("input/spikes.h5")


# %% run simulation
from bmtk.simulator import pointnet

configure = pointnet.Config.from_json("config.json")
configure.build_env()
network = pointnet.PointNetwork.from_config(configure)
sim = pointnet.PointSimulator.from_config(configure, network)
sim.run()

# %% make a raster plot
import h5py
import matplotlib.pyplot as plt
import numpy as np

# load spikes.h5
# f = h5py.File("output/spikes.h5", "r")
# spikes = f["input/spikes"]
with h5py.File("output/spikes.h5", "r") as f:
    spikes = f["/spikes/v1"]
    node_ids = np.array(spikes["node_ids"])
    timestamps = np.array(spikes["timestamps"])

# plot
mean_fr = len(node_ids) / ncells_v1 / tstop * 1000
plt.figure()
plt.plot(timestamps, node_ids, ".", markersize=2)
plt.xlabel("Time (s)")
plt.ylabel("Neuron ID")
plt.title(f"Mean firing rate: {mean_fr:.2f} Hz")
# plt.show()
plt.savefig("raster_plot.png")

# %%
