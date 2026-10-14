## O-PIUS: System Model for Organic-Cooled and Organic-Moderated PIUS-type Reactor

Transient simulation model for the passive shutdown mechanism in an organic-cooled (HB-40) PIUS-type reactor.
This simulation tool was developed as part of our Master's Thesis in Nuclear Engineering at KTH:
For complete mathematical derivations, model benchmark validations, and numerical discretization schemes, please refer to the full thesis manuscript.

*Note: The thesis document will be linked / made available upon final publication/defense.*

## Requirements & Setup

Install the required Python dependencies:
```bash
pip install numpy scipy matplotlib jupyter
```

## Repository Structure

- opius/: Core simulation package
  - config/: Plant geometry and steady-state operating parameters (system_configuration.py)
  - components/: Hydraulic components, 1D channels, and density lock mixing balances
  - physics/: Constitutive relations, HB-40 coolant properties, fuel conduction, and point kinetics
  - core/: Transient stepping driver and state management
  - data/: Core burnup tables and axial/radial peaking factor data (.npy files)
  - postprocessing/: Plotting and diagnostic routines
- notebooks/: Interactive exploratory notebook (`O-PIUS_V2.ipynb`) and light water PIUS notebook (its code not polished) (`PIUS_V5_code_not_polished.ipynb`)
- main.py: Top-level execution entry point
- Neutronics: Contains the neutronic study input generators, input files and results.
- Economics: Contains the economical study coolant cost model and results.

## How to Run

### 1. Configure the Simulation
Before running, open 'opius/config/system_configuration.py' to select the input parameters:

### 2. Execute
Run the standard transient simulation from the workspace root:
```bash
python main.py
```
Outputs (`results.npz` and summary plots) will be saved directly in the project root.

### 3. Interactive Analysis
Alternatively, open the Jupyter notebook to run the interactive stepping loop:
```bash
jupyter notebook notebooks/O-PIUS_V2.ipynb
```

## Authors
- David Ploumhans
- Alessio Sarcletti