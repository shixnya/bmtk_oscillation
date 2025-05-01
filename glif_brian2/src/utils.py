import os
import sys
import matplotlib.pyplot as plt
from brian2 import ms, mV, pA, second, volt, amp  # Import necessary units


def get_project_root():
    """Traverse up to find the project root marked by a known file/dir."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from src to glif_brian2 directory
    parent_dir = os.path.dirname(current_dir)
    # Go up one level from glif_brian2 to the main project directory
    project_root_candidate = os.path.dirname(parent_dir)

    # Basic check: does it contain expected top-level items?
    if os.path.exists(
        os.path.join(project_root_candidate, "README.md")
    ) or os.path.exists(os.path.join(project_root_candidate, "glif_brian2")):
        return project_root_candidate

    # Fallback if the structure is different than expected
    # Traverse up until a marker is found or root is hit
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != "/":
        if os.path.exists(os.path.join(current_dir, "README.md")) or os.path.exists(
            os.path.join(current_dir, ".git")
        ):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached root
            break
        current_dir = parent_dir
    # If no marker found, return a reasonable default (e.g., two levels up from src)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_src_to_path(project_root):
    """Adds the src directory to the Python path."""
    src_dir = os.path.join(project_root, "glif_brian2", "src")
    if src_dir not in sys.path:
        sys.path.append(src_dir)


def plot_simulation_results(
    state_monitor, spike_monitor, params, config, cell_model_filename
):
    """Generates plots for the single-cell simulation results."""
    plt.style.use("seaborn-v0_8-darkgrid")
    num_plots = 2 + (1 if params["asc_present"] else 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)
    ax_idx = 0

    # Plot Voltage
    ax = axes[ax_idx] if num_plots > 1 else axes  # Handle single subplot case
    ax.plot(
        state_monitor.t / ms, state_monitor.V[0] / mV, label="Vm", color="royalblue"
    )
    # Add spike markers
    if spike_monitor.num_spikes > 0:
        ax.vlines(
            spike_monitor.t / ms,
            ax.get_ylim()[0],
            ax.get_ylim()[1],
            color="red",
            linestyle="--",
            alpha=0.7,
            label="Spikes",
        )
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(
        f"Single GLIF Simulation ({cell_model_filename})\n"
        f"I_inj={config['INJECTED_CURRENT_AMP']/pA:.1f} pA ({config['CURRENT_START_TIME']/ms:.0f}-{config['CURRENT_END_TIME']/ms:.0f} ms)"
    )
    ax.legend(loc="upper right")
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
        ax.legend(loc="upper right")
        ax.grid(True)
        ax_idx += 1

    # Plot Injected Current
    ax = axes[ax_idx]
    injected_current_trace = [
        (
            config["INJECTED_CURRENT_AMP"]
            if config["CURRENT_START_TIME"] <= t < config["CURRENT_END_TIME"]
            else 0 * pA
        )
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
    ax.legend(loc="upper right")
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to prevent title overlap
    plt.show()

    print(f"Number of spikes: {spike_monitor.num_spikes}")
    if spike_monitor.num_spikes > 0:
        print(f"Spike times (ms): {spike_monitor.t / ms}")
