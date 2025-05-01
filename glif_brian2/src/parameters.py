import json
from brian2 import mV, nS, pF, ms, pA, second  # Added second for asc_decay units


def load_glif_params(filepath):
    """Loads GLIF parameters from a JSON file and assigns units."""
    with open(filepath, "r") as f:
        params_raw = json.load(f)

    params = {}
    # Membrane properties
    # V_m is the initial membrane potential, often set to E_L
    params["V_init"] = params_raw.get("V_m", params_raw["E_L"]) * mV
    params["V_th"] = params_raw["V_th"] * mV
    params["g"] = params_raw["g"] * nS
    params["E_L"] = params_raw["E_L"] * mV
    params["C_m"] = params_raw["C_m"] * pF
    params["t_ref"] = params_raw["t_ref"] * ms
    params["V_reset"] = params_raw["V_reset"] * mV

    # After-spike currents (ASC)
    # Check if 'after_spike_currents' key exists and is True
    params["asc_present"] = params_raw.get("after_spike_currents", False)
    if params["asc_present"] and "asc_init" in params_raw:
        params["asc_init"] = [val * pA for val in params_raw["asc_init"]]
        # asc_decay is often given in seconds in Allen SDK models
        params["asc_decay"] = [val * second for val in params_raw["asc_decay"]]
        params["asc_amps"] = [val * pA for val in params_raw["asc_amps"]]
        params["num_asc"] = len(params["asc_init"])
    else:
        params["asc_present"] = False  # Ensure it's false if keys are missing
        params["num_asc"] = 0
        params["asc_init"] = []
        params["asc_decay"] = []
        params["asc_amps"] = []

    # Other flags (can be used for model selection later)
    params["spike_dependent_threshold"] = params_raw.get(
        "spike_dependent_threshold", False
    )
    params["adapting_threshold"] = params_raw.get("adapting_threshold", False)

    # Note: Synaptic parameters are ignored in this single-cell setup

    return params
