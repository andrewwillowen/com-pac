#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #
from com_pac.writer import generate_output_file, generate_csv_output
from com_pac.isotopologues import get_dataframes
from com_pac.diagonalize import get_principal_axes
from com_pac.parser import parse_input_file

from sys import argv
import pandas as pd
from pathlib import Path


def read_args(num_of_decimals):
    """
    Manually parsing arguments...
    """
    # TODO: replace with argparse

    # reading arguments from shell
    if len(argv) < 2:
        print("""
        Usage:
            principal_axes_calculator pac-input-file.txt num_of_decimals

        Uses the information provided in the input file to determine the principal axes systems
        and corresponding rotational constants & dipole moments for each isotopologue provided.
        The output file summarizes the intermediate and final results to assist in replication,
        if necessary, while the .csv file contains only the rotational constants, dipole moments,
        and principal axes coordinates of each isotopologue.

        The input file has the following format:
            Coordinates
            C 0.0 0.0 0.0
            O 1.2 0.0 0.0
            H -0.63 0.63 0
            H -0.63 -0.63 0

            Dipole
            1.6 0.0 0.0

            Isotopologues
            12 16 1 1 name1
            12 16 2 1 name2
            12 16 2 2 name3

        The headings are required to be at the start of the corresponding section (but
        are case insensitive), but otherwise comments can be placed on any other line,
        or at the end of any line in the file. Names for the isotopologues must be provided,
        and the order of the mass numbers provided must match the order of the atoms
        listed in the Coordinates section.

        By default, the numbers in the output _pac.out will be reported to 6 decimal places.
        This can be changed by providing replacing `num_of_decimals` in the usage line above
        with a positive integer.
        
        To use an experimental dipole value, first use this tool to obtain the principal axes 
        Cartesian coordinates for the corresponding isotopologue. That guarantees that the
        axes are in A, B, C ordering and you can simply set the dipole values to the 
        experimental mu_A, mu_B, mu_C.  
        """)
        quit()
    else:
        input_file_path = Path(argv[1])
        if len(argv) >= 3:
            raw_num_of_decimals = argv[2]
            try:
                num_of_decimals = int(raw_num_of_decimals)
                if not (num_of_decimals >= 0):
                    raise ValueError
            except (Exception,):
                print("Make sure that num_of_decimals is a positive integer.")
            if len(argv) > 3:
                print("The following arguments will be ignored: {}".format(argv[3:]))

        return input_file_path, num_of_decimals


def main():
    input_file_path, num_of_decimals = read_args(6)

    # ================================ #
    #  reading contents of input file  #
    # ================================ #

    if input_file_path is not None:
        input_file_dir = input_file_path.parent
        input_file_name = input_file_path.name
    else:
        raise ValueError("Failure to import file path.")

    with open(input_file_path, "r") as infile:
        input_file = infile.read()

    (
        isotopologue_names,
        isotopologue_dict,
        n_atoms,
        atom_symbols,
        mol_coordinates,
        mol_dipole,
        atom_numbering,
    ) = parse_input_file(input_file)

    (
        atom_masses,
        rotational_constants,
        pa_dipoles,
        pa_coordinates,
        pa_inertias,
        com_coordinates,
        com_inertias,
        eigenvectors,
        eigenvalues,
    ) = get_principal_axes(
        isotopologue_names,
        isotopologue_dict,
        n_atoms,
        atom_symbols,
        mol_coordinates,
        mol_dipole,
    )

    (
        atom_masses_df,
        rotational_constants_df,
        dipole_components_df,
        com_coordinates_df_dict,
        com_inertias_df_dict,
        eigenvectors_df_dict,
        pa_inertias_df_dict,
        pa_coordinates_df_dict,
    ) = get_dataframes(
        atom_masses,
        atom_symbols,
        rotational_constants,
        pa_dipoles,
        isotopologue_names,
        com_coordinates,
        atom_numbering,
        com_inertias,
        eigenvectors,
        pa_inertias,
        pa_coordinates,
    )

    # ==================== #
    #  Outputting results  #
    # ==================== #

    if input_file_name.count(".") != 1:
        input_file_base_name = str(input_file_name)
    else:
        input_file_base_name = str(input_file_name).split(".")[0]

    text_output_name = input_file_base_name + "_pac.out"
    csv_output_name = input_file_base_name + "_pac.csv"

    text_output_path = input_file_dir.joinpath(text_output_name)
    csv_output_path = input_file_dir.joinpath(csv_output_name)

    generate_output_file(
        num_of_decimals,
        csv_output_name,
        input_file,
        atom_masses_df,
        rotational_constants_df,
        dipole_components_df,
        isotopologue_names,
        com_coordinates_df_dict,
        atom_symbols,
        com_inertias_df_dict,
        eigenvectors_df_dict,
        eigenvalues,
        pa_inertias_df_dict,
        pa_coordinates_df_dict,
        text_output_path,
    )

    generate_csv_output(
        pa_coordinates_df_dict,
        rotational_constants_df,
        dipole_components_df,
        atom_masses_df,
        csv_output_path,
    )


if __name__ == "__main__":
    main()
