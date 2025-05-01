from brian2 import pA, amp, volt, second, exp  # Import exp


def get_glif_asc_equations(num_asc):
    """
    Generates Brian2 equations for a GLIF model with after-spike currents (ASC).
    Defines ASC dynamics and I_asc_total before the V equation.
    """
    # Start building the equation string
    eqs = ""

    # Add equations for each ASC component first
    asc_sum_terms = []
    for i in range(num_asc):
        eqs += f"""
        dI_asc{i}/dt = -I_asc{i} / asc_decay{i} : amp (unless refractory)
        asc_decay{i} : second # Decay time constant for ASC {i}
        asc_amp{i} : amp     # Amplitude increment for ASC {i}
        """
        asc_sum_terms.append(f"I_asc{i}")

    # Define the I_asc_total subexpression (alias)
    if asc_sum_terms:
        # Ensure newline before adding the subexpression if ASC equations exist
        if num_asc > 0:
            eqs += "\n"
        eqs += f'I_asc_total = {" + ".join(asc_sum_terms)} : amp'
    else:
        # If no ASCs, define I_asc_total as 0
        eqs += "\nI_asc_total = 0*pA : amp"

    # Now add the main V equation and I_inj declaration
    # Ensure newline before adding V equation
    eqs += """
    dV/dt = (-g*(V - E_L) + I_inj + I_asc_total) / C_m : volt (unless refractory)
    I_inj : amp # Injected current
    # Add parameter declarations back
    g : siemens # Membrane conductance
    E_L : volt # Reversal potential
    C_m : farad # Membrane capacitance
    V_th : volt # Spike threshold
    V_reset : volt # Reset potential
    t_ref : second # Refractory period
    """
    # Parameters like g, E_L, C_m, V_th, V_reset, t_ref will be inferred from NeuronGroup arguments

    # Reset equations
    # Include decay during refractory period
    reset_eqs = "V = V_reset"
    for i in range(num_asc):
        # Apply decay during t_ref before adding the increment
        reset_eqs += f"\nI_asc{i} = asc_amp{i} + I_asc{i} * exp(-t_ref / asc_decay{i})"

    return eqs, reset_eqs
