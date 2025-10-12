import os
import pandas as pd
from pymatgen.core.composition import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter, PDEntry

# Define the base path where your generated data is stored
DATA_PATH = os.path.join("/Users", "nicholaskryger-nelson", "Ternary", "FE")

# Define the list of elements you have generated data for
elements = ["Ag","Al","Au","B","Ba","Bi","Be","Ca","Cd","Co","Cr","Cs","Cu","Fe","Ga","Ge","K","Hf","Hg","In","Ir","La","Li","Mg","Mn","Mo","Na","Nb","Ni","Os","Pb","Pd","Pt","Rb","Re","Rh","Ru","Sb","Sc","Sr","Sn","Ta","Tc","Ti","Tl","W","Y","Zr","Zn"]

# Loop through each element to generate and save its phase diagram
for element in elements:
    print(f"Generating 3D phase diagram for the {element}-P-S system...")

    # Construct the full path to the input CSV file
    input_file = os.path.join(DATA_PATH, f"{element}_FE.csv")

    # Check if the data file exists before proceeding
    if not os.path.exists(input_file):
        print(f"Warning: Data file not found for {element} at {input_file}. Skipping.")
        continue

    # Read the ternary data from the CSV file
    ternary_data = pd.read_csv(input_file)

    # Create a list of PDEntry objects from the dataframe
    # The PhaseDiagram requires total energy, so we multiply the predicted
    # formation energy per atom (FE) by the number of atoms in the formula.
    entries = []
    for _, row in ternary_data.iterrows():
        try:
            composition = Composition(row['formula'])
            total_energy = row['FE'] * composition.num_atoms
            entries.append(PDEntry(composition, total_energy))
        except Exception as e:
            print(f"Could not process formula '{row['formula']}': {e}")

    # Ensure there are entries to plot
    if not entries:
        print(f"No valid chemical compositions found for {element}. Skipping plot generation.")
        continue

    # Create the PhaseDiagram object
    phase_diagram = PhaseDiagram(entries)

    # Initialize the plotter with the 3D style
    plotter = PDPlotter(phase_diagram, ternary_style='3d')

    # Define the output path for the HTML file
    output_file = os.path.join(DATA_PATH, f"ternary_plot_3d_{element}.html")

    # Generate the plot and write it to an HTML file
    # The 'ordering' argument ensures the elements are on the correct corners of the plot
    plot = plotter.get_plot(ordering=["P", "S", element])
    plot.write_html(output_file)

    print(f"Successfully saved plot to {output_file}")

print("\nAll 3D plots have been generated.")