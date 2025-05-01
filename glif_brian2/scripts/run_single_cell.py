import os
import sys
import argparse  # Import argparse for command-line arguments
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
)

# --- Setup Project Path --- #
# This block assumes utils.py is in ../src relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir_relative = os.path.join(script_dir, "..", "src")
sys.path.insert(0, os.path.abspath(src_dir_relative))

try:
    from utils import get_project_root, add_src_to_path, plot_simulation_results
    from parameters import load_glif_params
    from neuron_models import get_glif_asc_equations
except ImportError as e:
    print(
        f"Error importing modules. Ensure 'utils.py', 'parameters.py', and 'neuron_models.py' are in the src directory ({src_dir_relative}). Error: {e}"
    )
    sys.exit(1)

# --- Simulation Configuration --- #
# Default values
DEFAULT_CELL_MODEL_FILENAME = (
    "489751692_glif_lif_asc_config.json"  # One of the typical ones
)
SIM_CONFIG = {
    "SIMULATION_DURATION": 500 * ms,
    "INJECTED_CURRENT_AMP": 250 * pA,
    "CURRENT_START_TIME": 100 * ms,
    "CURRENT_END_TIME": 400 * ms,
    "DT": 0.1 * ms,
}


def run_simulation(cell_model_filename, config):
    """Runs a single GLIF cell simulation with the given configuration."""

    PROJECT_ROOT = get_project_root()  # Find project root using utils
    CELL_MODEL_PATH = os.path.join(
        PROJECT_ROOT, "glif_brian2", "cell_models", cell_model_filename
    )

    # --- Load Parameters --- #
    try:
        params = load_glif_params(CELL_MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Cell model file not found at {CELL_MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading parameters from {cell_model_filename}: {e}")
        sys.exit(1)

    # --- Setup Brian2 Model --- #
    defaultclock.dt = config["DT"]
    eqs, reset_eqs = get_glif_asc_equations(params["num_asc"])

    neuron = NeuronGroup(
        1,
        model=eqs,
        threshold="V>V_th",
        reset=reset_eqs,
        refractory="t_ref",
        method="euler",
    )

    # --- Set Initial Conditions and Parameters --- #
    neuron.V = params["V_init"]
    neuron.I_inj = 0 * pA
    neuron.g = params["g"]
    neuron.E_L = params["E_L"]
    neuron.C_m = params["C_m"]
    neuron.V_th = params["V_th"]
    neuron.V_reset = params["V_reset"]
    neuron.t_ref = params["t_ref"]

    if params["asc_present"]:
        for i in range(params["num_asc"]):
            setattr(neuron, f"I_asc{i}", params["asc_init"][i])
            setattr(neuron, f"asc_decay{i}", params["asc_decay"][i])
            setattr(neuron, f"asc_amp{i}", params["asc_amps"][i])
    elif "I_asc_total" in neuron.variables:
        neuron.I_asc_total = 0 * pA

    # --- Setup Monitors --- #
    vars_to_monitor = ["V"]
    if params["asc_present"]:
        vars_to_monitor.append("I_asc_total")
        vars_to_monitor.extend([f"I_asc{i}" for i in range(params["num_asc"])])

    state_monitor = StateMonitor(neuron, vars_to_monitor, record=0)
    spike_monitor = SpikeMonitor(neuron)

    # --- Run Simulation --- #
    print(
        f"Running simulation for {config['SIMULATION_DURATION']} using {cell_model_filename}..."
    )
    run(config["CURRENT_START_TIME"])
    neuron.I_inj = config["INJECTED_CURRENT_AMP"]
    run(config["CURRENT_END_TIME"] - config["CURRENT_START_TIME"])
    neuron.I_inj = 0 * pA
    run(config["SIMULATION_DURATION"] - config["CURRENT_END_TIME"])
    print("Simulation finished.")

    # --- Plot Results --- #
    plot_simulation_results(
        state_monitor, spike_monitor, params, config, cell_model_filename
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single GLIF cell simulation using Brian2."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_CELL_MODEL_FILENAME,
        help=f"Filename of the cell model JSON in cell_models directory (default: {DEFAULT_CELL_MODEL_FILENAME})",
    )
    parser.add_argument(
        "-i",
        "--current",
        type=float,
        default=SIM_CONFIG["INJECTED_CURRENT_AMP"] / pA,
        help=f"Injected current amplitude in pA (default: {SIM_CONFIG['INJECTED_CURRENT_AMP']/pA})",
    )
    parser.add_argument(
        "--t_start",
        type=float,
        default=SIM_CONFIG["CURRENT_START_TIME"] / ms,
        help=f"Start time of current injection in ms (default: {SIM_CONFIG['CURRENT_START_TIME']/ms})",
    )
    parser.add_argument(
        "--t_end",
        type=float,
        default=SIM_CONFIG["CURRENT_END_TIME"] / ms,
        help=f"End time of current injection in ms (default: {SIM_CONFIG['CURRENT_END_TIME']/ms})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=SIM_CONFIG["SIMULATION_DURATION"] / ms,
        help=f"Total simulation duration in ms (default: {SIM_CONFIG['SIMULATION_DURATION']/ms})",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=SIM_CONFIG["DT"] / ms,
        help=f"Simulation time step in ms (default: {SIM_CONFIG['DT']/ms})",
    )

    args = parser.parse_args()

    # Update config from command-line arguments
    current_config = {
        "SIMULATION_DURATION": args.duration * ms,
        "INJECTED_CURRENT_AMP": args.current * pA,
        "CURRENT_START_TIME": args.t_start * ms,
        "CURRENT_END_TIME": args.t_end * ms,
        "DT": args.dt * ms,
    }

    run_simulation(args.model, current_config)
