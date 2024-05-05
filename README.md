# BMTK simulation environment for oscillation study

To make an environment to work on, please use the environment.yml.

```bash
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

If you omit `<env_name>` it'll become `bmtk_nest`.

To run simulation, simply run the script.

```bash
python build_and_run.py
```

It'll create a network, run the simulation, and make a raster plot of the simulation.
Please separate scripts into different parts (building, running, plotting) as needed.

