import os
import sys
import matplotlib.pyplot as plt
from brian2 import (
    NeuronGroup,
    StateMonitor,
    SpikeMonitor,
    run,
    defaultclock,
    pA,
    ms,
    second,
    volt,
    amp,
    mV,
    nS,
    pF,
    siemens,
    farad,
)  # Import necessary units and base types


# --- Determine project root and add src to sys.path ---
def get_project_root():
    """Traverse up to find the project root marked by a known file/dir."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != "/":  # Avoid infinite loop at filesystem root
        # Heuristic: Check for a known top-level file or directory
        if os.path.exists(os.path.join(current_dir, "README.md")) or os.path.exists(
            os.path.join(current_dir, ".git")
        ):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root
            break
        current_dir = parent_dir
    # Fallback if marker not found (adjust as needed)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


PROJECT_ROOT = get_project_root()
SRC_DIR = os.path.join(PROJECT_ROOT, "glif_brian2", "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Now import local modules
try:
    from parameters import load_glif_params
    from neuron_models import get_glif_asc_equations
except ImportError as e:
    print(f"Error importing modules from {SRC_DIR}: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Simulation Configuration ---
# Use relative path from script location to cell_models
# These three cell models are typical mouse V1 L2/3 pyramidal cells
CELL_MODEL_FILENAME = "489751692_glif_lif_asc_config.json"
# CELL_MODEL_FILENAME = "490376252_glif_lif_asc_config.json"
# CELL_MODEL_FILENAME = "505512874_glif_lif_asc_config.json"
CELL_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "glif_brian2", "cell_models", CELL_MODEL_FILENAME
)

SIMULATION_DURATION = 500 * ms
INJECTED_CURRENT_AMP = 250 * pA  # Example current injection amplitude
CURRENT_START_TIME = 100 * ms
CURRENT_END_TIME = 400 * ms
DT = 0.1 * ms  # Simulation time step

# --- Load Parameters ---
try:
    params = load_glif_params(CELL_MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Cell model file not found at {CELL_MODEL_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading parameters: {e}")
    sys.exit(1)

# --- Setup Brian2 Model ---
defaultclock.dt = DT

# Get equations based on the number of ASC components
eqs, reset_eqs = get_glif_asc_equations(params["num_asc"])

# Create NeuronGroup
# Ensure all parameters used in equations are defined either here or as variables
neuron = NeuronGroup(
    1,
    model=eqs,
    threshold="V>V_th",
    reset=reset_eqs,
    refractory="t_ref",
    method="exact",
)  # Using Euler for simplicity

# --- Set Initial Conditions and Parameters ---
neuron.V = params["V_init"]  # Use V_init from params
neuron.I_inj = 0 * pA  # Initialize injected current

# Assign parameters from the loaded dictionary to the NeuronGroup
neuron.g = params["g"]
neuron.E_L = params["E_L"]
neuron.C_m = params["C_m"]
neuron.V_th = params["V_th"]
neuron.V_reset = params["V_reset"]
neuron.t_ref = params["t_ref"]

# Assign ASC parameters if present
if params["asc_present"]:
    for i in range(params["num_asc"]):
        # Initialize state variables for ASC currents
        setattr(neuron, f"I_asc{i}", params["asc_init"][i])
        # Assign parameters for ASC dynamics
        setattr(neuron, f"asc_decay{i}", params["asc_decay"][i])
        setattr(neuron, f"asc_amp{i}", params["asc_amps"][i])
else:
    # Ensure I_asc_total is initialized if no ASCs (though model eq should handle this)
    if "I_asc_total" in neuron.variables:
        neuron.I_asc_total = 0 * pA

# --- Setup Monitors ---
# Determine variables to monitor
vars_to_monitor = ["V"]
if params["asc_present"]:
    vars_to_monitor.append("I_asc_total")
    vars_to_monitor.extend([f"I_asc{i}" for i in range(params["num_asc"])])

state_monitor = StateMonitor(
    neuron, vars_to_monitor, record=0
)  # Record from the first (only) neuron
spike_monitor = SpikeMonitor(neuron)

# --- Run Simulation ---
print(f"Running simulation for {SIMULATION_DURATION}...")
# Initial run with no current
run(CURRENT_START_TIME)
# Apply current injection
neuron.I_inj = INJECTED_CURRENT_AMP
run(CURRENT_END_TIME - CURRENT_START_TIME)
# Run after current stops
neuron.I_inj = 0 * pA
run(SIMULATION_DURATION - CURRENT_END_TIME)
print("Simulation finished.")

# --- Plot Results ---
plt.style.use("seaborn-v0_8-darkgrid")  # Use a nice style
num_plots = 2 + (1 if params["asc_present"] else 0)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
ax_idx = 0

# Plot Voltage
ax = axes[ax_idx]
ax.plot(state_monitor.t / ms, state_monitor.V[0] / mV, label="Vm", color="royalblue")
ax.set_ylabel("Voltage (mV)")
ax.set_title(f"Single GLIF Neuron Simulation ({CELL_MODEL_FILENAME})")
ax.grid(True)
ax_idx += 1

# Plot ASC components if present
if params["asc_present"]:
    ax = axes[ax_idx]
    for i in range(params["num_asc"]):
        ax.plot(
            state_monitor.t / ms,
            getattr(state_monitor, f"I_asc{i}")[0] / pA,
            label=f"I_asc{i}",
        )
    ax.plot(
        state_monitor.t / ms,
        state_monitor.I_asc_total[0] / pA,
        label="I_asc_total",
        linestyle="--",
        color="black",
    )
    ax.set_ylabel("ASC (pA)")
    ax.legend()
    ax.grid(True)
    ax_idx += 1

# Plot Injected Current
ax = axes[ax_idx]
injected_current_trace = [
    (INJECTED_CURRENT_AMP if CURRENT_START_TIME <= t < CURRENT_END_TIME else 0 * pA)
    for t in state_monitor.t
]
ax.plot(
    state_monitor.t / ms,
    [c / pA for c in injected_current_trace],
    label="Injected Current",
    color="firebrick",
)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Current (pA)")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()

print(f"Number of spikes: {spike_monitor.num_spikes}")
print(f"Spike times (ms): {spike_monitor.t / ms}")
