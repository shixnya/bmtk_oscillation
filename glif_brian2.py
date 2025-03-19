# %%
#####################
# NEST Simulation (Using Default GLIF3 Parameters)
#####################
import nest
import nest.voltage_trace
import matplotlib.pyplot as plt
import numpy as np

# Reset the NEST kernel
nest.ResetKernel()

# Set the resolution of the NEST kernel to 0.01 ms
resolution = 0.01
nest.SetKernelStatus({"resolution": resolution})

# Create a GLIF3 neuron using "glif_psc" with mechanism flags that select GLIF3.
neuron_nest = nest.Create(
    "glif_psc",
    params={
        "spike_dependent_threshold": False,
        "after_spike_currents": True,
        "adapting_threshold": False,
    },
)
# Note: We do not override any other parameters so that the defaults from the C++ code are used.
# Override the refractory period to be 0 ms.
# nest.SetStatus(neuron_nest, {"t_ref": 0.001})

# Create a step current generator that injects 500 pA between 100 ms and 400 ms.
current_generator = nest.Create(
    "step_current_generator",
    params={
        "amplitude_times": [99.0, 399.0],
        "amplitude_values": [500.0, 0.0],
    },
)
nest.Connect(current_generator, neuron_nest)

# Set up recording devices.
voltmeter = nest.Create("voltmeter", params={"interval": resolution})
nest.Connect(voltmeter, neuron_nest)

# The after‐spike current (ASC) is recorded as "ASCurrents_sum".
multimeter = nest.Create(
    "multimeter", params={"interval": resolution, "record_from": ["ASCurrents_sum"]}
)
nest.Connect(multimeter, neuron_nest)

# Run the NEST simulation for 500 ms.
nest.Simulate(500.0)

# Retrieve NEST data.
nest_data = nest.GetStatus(voltmeter)[0]
t_nest = nest_data["events"]["times"]
V_nest = nest_data["events"]["V_m"]

multimeter_data = nest.GetStatus(multimeter)[0]
t_w_nest = multimeter_data["events"]["times"]
w_nest = multimeter_data["events"]["ASCurrents_sum"]

#####################
# Brian2 Simulation (Matching NEST Defaults)
#####################
from brian2 import *

# Clear any previous Brian2 simulation.
start_scope()

# Update the simulation time step to 0.01 ms
defaultclock.dt = resolution * ms

# --- NEST Default Parameters ---
# From glif_psc.cpp defaults:
C_b = 58.72 * pF  # Membrane capacitance
G_b = 9.43 * nS  # Leak conductance
tau_m_b = C_b / G_b  # Membrane time constant ≈ 6.225 ms
E_b = -78.85 * mV  # Resting potential
Vt_b = -51.68 * mV  # Spike threshold
Vreset_b = -78.85 * mV  # Reset potential (E_L + 0)
t_ref_b = 3.75 * ms  # Refractory period
# t_ref_b = 0.001 * ms  # Refractory period

# --- After-Spike Current (ASC) Components ---
# Two ASC components as given by defaults:
tau_asc1 = (1 / 0.003) * ms  # ≈ 333.33 ms
tau_asc2 = (1 / 0.1) * ms  # 10 ms
b1 = -9.18 * pA  # ASC increment component 1
b2 = -198.94 * pA  # ASC increment component 2

# --- Simulation and Input Current ---
duration = 500 * ms
dt_sim = resolution * ms

# Define a step current injection: 500 pA between 100 ms and 400 ms.
times_b = (
    np.arange(0, float(duration / ms) + float(dt_sim / ms), float(dt_sim / ms)) * ms
)
I_values_b = np.zeros(len(times_b)) * pA
I_values_b[(times_b >= 100 * ms) & (times_b < 400 * ms)] = 500 * pA
I_inj = TimedArray(I_values_b, dt=dt_sim)

# --- Brian2 Model Equations ---
# We add the two ASC components (w1 and w2) to the input current.
eqs = """
dV/dt = (-(V - E_b)/tau_m_b + (I_inj(t) + (w1 + w2))/C_b) : volt (unless refractory)
dw1/dt = -w1/tau_asc1 : amp (unless refractory)
dw2/dt = -w2/tau_asc2 : amp (unless refractory)
"""

reset = """
V = Vreset_b
w1 = b1 + w1 * exp(-t_ref_b / tau_asc1)
w2 = b2 + w2 * exp(-t_ref_b / tau_asc2)
"""

neuron = NeuronGroup(
    1,
    eqs,
    threshold="V > Vt_b",
    reset=reset,
    refractory=t_ref_b,
    method="exact",  # Changed solver to exponential Euler
)
neuron.V = E_b
neuron.w1 = 0 * amp
neuron.w2 = 0 * amp

# Set up monitors.
M = StateMonitor(neuron, ["V", "w1", "w2"], record=True)
spikemon = SpikeMonitor(neuron)

run(duration)

#####################
# Comparison Plot
#####################
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)


# Plot the membrane potential.
axs[0].plot(t_nest - t_nest[0], V_nest, label="NEST (glif_psc default)")
axs[0].plot(M.t / ms, M.V[0] / mV, label="Brian2", linestyle="--")
axs[0].set_ylabel("Membrane potential (mV)")
axs[0].set_title("Membrane Potential Comparison")
axs[0].legend()

# Plot the after-spike current (ASC): sum the two Brian2 components.
w_total_b = M.w1[0] + M.w2[0]
axs[1].plot(
    t_w_nest, w_nest / 1000, label="NEST (ASCurrents_sum, nA)"
)  # NEST values are in pA; convert to nA.
axs[1].plot(M.t / ms, w_total_b / nA, label="Brian2 (w1+w2)", linestyle="--")
axs[1].set_xlabel("Time (ms)")
axs[1].set_ylabel("After-Spike Current (nA)")
axs[1].set_title("After-Spike Current Comparison")
axs[1].legend()

# Set x-axis limits for both subplots.
# axs[0].set_xlim(100, 110)
# axs[1].set_xlim(100, 110)

plt.tight_layout()
plt.show()
