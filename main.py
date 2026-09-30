import numpy as np
from opius.config.system_configuration import Time, enable_plots
from opius.core.driver import OPIUSDriver
from opius.postprocessing.plotting import plot_transient

def main():
    sim = OPIUSDriver()
    sim.initialize_steady_state()

    print(f"Simulating up to {Time} s...")
    while sim.t[-1] < Time:
        sim.step()
        print(f"Simulation progress: {sim.t[-1] / Time * 100:.1f} %")
    sim.finalize()
    print("Simulation complete!")

    # 1) Save results
    np.savez_compressed("results.npz", **sim.history)
    print("Saved results to results.npz")

    # 2) Render plots
    if enable_plots:
        plot_transient(sim.history, sim)

if __name__ == "__main__":
    main()