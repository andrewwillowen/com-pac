"""
Running the theta calculations by hand
to explore what the data looks like.
"""

from com_pac.parser import parse_input_file
from com_pac.diagonalize import get_theta_values, get_principal_axes
from com_pac.dataframes import get_theta_df

# Load the data
# with open("two-isos.txt", "r") as infile:
with open("./full-isos.txt", "r") as infile:
    input_file = infile.read()

iso_names, iso_dict, n_atoms, atom_symbols, mol_coords, mol_dipoles, atom_numbs = (
    parse_input_file(input_file)
)

(
    atom_masses,
    rot_consts,
    pa_dipoles,
    pa_coords,
    pa_inertias,
    com_coords,
    com_inertias,
    evecs,
    evals,
    com_vals,
) = get_principal_axes(
    iso_names, iso_dict, n_atoms, atom_symbols, mol_coords, mol_dipoles
)

theta_data = get_theta_values(
    iso_names, atom_masses, pa_coords
)

theta_df = get_theta_df(iso_names, theta_data)
