import os
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry, PDPlotter

# --- Configuration ---
PATH = os.path.join("/Users", "nicholaskryger-nelson")
FE_DATA_DIR = os.path.join(PATH, "Ternary", "FE")
BG_DATA_DIR = os.path.join(PATH, "Ternary", "BG")
OUTPUT_DIR = os.path.join(PATH, "Ternary", "ternary_plots_BG_styled")

ELEMENTS = [
    "Ag", "Al", "Au", "B", "Ba", "Bi", "Be", "Ca", "Cd", "Co", "Cr", "Cs",
    "Cu", "Fe", "Ga", "Ge", "K", "Hf", "Hg", "In", "Ir", "La", "Li", "Mg",
    "Mn", "Mo", "Na", "Nb", "Ni", "Os", "Pb", "Pd", "Pt", "Rb", "Re", "Rh",
    "Ru", "Sb", "Sc", "Sr", "Sn", "Ta", "Tc", "Ti", "Tl", "W", "Y", "Zr", "Zn"
]

# Ensure the main output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Main Loop ---
for element in ELEMENTS:
    print(f"--- Processing: {element}-P-S System ---")

    fe_file = os.path.join(FE_DATA_DIR, f"{element}_FE.csv")
    bg_file = os.path.join(BG_DATA_DIR, f"{element}_BG.csv")

    if not os.path.exists(fe_file) or not os.path.exists(bg_file):
        print(f"Data file missing for {element}. Skipping.")
        continue

    # Load data
    df_fe = pd.read_csv(fe_file)
    df_bg = pd.read_csv(bg_file)

    # Prepare PDEntry list and bandgap dictionary
    entries = []
    for _, row in df_fe.iterrows():
        try:
            energy = row['FE']
            comp = Composition(row['formula'])
            entries.append(PDEntry(comp, energy * comp.num_atoms))
        except Exception as e:
            print(f"Error processing FE data for {row['formula']}: {e}")

    bandgaps = {}
    for _, row in df_bg.iterrows():
        try:
            formula = Composition(row['formula']).reduced_formula
            bandgaps[formula] = row['BG']
        except Exception as e:
            print(f"Error processing BG data for {row['formula']}: {e}")

    if not entries:
        print(f"No valid entries for {element}. Skipping plot.")
        continue

    # Create Phase Diagram
    pd_obj = PhaseDiagram(entries)

    # --- Generate and save the 2D plot with custom styling ---
    try:
        # Define your custom styling for markers
        plot_kwargs = {
            "markerfacecolor": "orange",
            "markersize": 12,
            "linewidth": 4,
        }
        
        # Create plotter instance with custom styles and bandgap data
        plotter = PDPlotter(pd_obj, backend="matplotlib", bandgap_data=bandgaps, **plot_kwargs)
        
        # Define the colormap for bandgap visualization
        energy_colormap = plt.get_cmap('coolwarm')

        # Create a new figure to avoid overlaying plots
        plt.figure()
        
        # Call the get_plot function with the desired colormap and other settings
        ax = plotter.get_plot(
            label_unstable=False,
            label_stable=True,
            energy_colormap=energy_colormap,
            ordering=[element, "P", "S"]
        )
        
        # Additional customizations
        ax.set_title(f"Band Gap Phase Diagram for {element}-P-S", fontsize=20)

        # Main output file
        output_file = os.path.join(OUTPUT_DIR, f"bg_ternary_plot_{element}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)

        # Save to group-specific folder
        group = Element(element).group
        group_dir = os.path.join(OUTPUT_DIR, str(group))
        os.makedirs(group_dir, exist_ok=True)
        group_output_file = os.path.join(group_dir, f"bg_ternary_plot_{element}.png")
        plt.savefig(group_output_file, bbox_inches='tight', dpi=300)

        plt.close()  # Close the figure to free memory
        print(f"Saved styled 2D plot for {element}.")

    except Exception as e:
        print(f"Failed to generate plot for {element}: {e}")
        if plt.gcf().get_axes(): # Check if a figure is open before trying to close
            plt.close()

print("\nAll plotting complete.")