#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #
from com_pac.writer import generate_output_file, generate_csv_output
from com_pac.dataframes import get_dataframes
from com_pac.diagonalize import get_principal_axes, get_theta_values
from com_pac.parser import parse_input_file

import argparse
from pathlib import Path

from com_pac.__about__ import __version__


def _non_negative_int(value: str) -> int:
    """Validate and convert a string to a non-negative integer for argparse."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer")
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"'{value}' is not a non-negative integer")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for com-pac."""
    parser = argparse.ArgumentParser(
        prog="com-pac",
        description=(
            "Uses the information provided in the input file to determine the principal axes "
            "systems and corresponding rotational constants & dipole moments for each "
            "isotopologue provided. The output file summarizes the intermediate and final "
            "results to assist in replication, if necessary, while the .csv file contains "
            "only the rotational constants, dipole moments, and principal axes coordinates "
            "of each isotopologue."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
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

To use an experimental dipole value, first use this tool to obtain the principal axes
Cartesian coordinates for the corresponding isotopologue. That guarantees that the
axes are in A, B, C ordering and you can simply set the dipole values to the
experimental mu_A, mu_B, mu_C.\
""",
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the pac input file.",
    )
    parser.add_argument(
        "--decimals",
        type=_non_negative_int,
        default=6,
        metavar="N",
        dest="num_of_decimals",
        help="Number of decimal places in the output (default: 6). Must be a non-negative integer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Directory to write output files to (default: same directory as input file).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--theta",
        action="store_true",
        default=False,
        dest="theta",
        help="Calculate and include theta values in the output.",
    )

    return parser


def _set_output_dir(output_dir: Path) -> None:
    """Set a custom output directory for generated files."""
    raise NotImplementedError("Custom output directory is not yet supported.")


def read_args() -> tuple[Path, int, bool]:
    """Parse command-line arguments using argparse.

    Returns a tuple of (input_file_path, num_of_decimals, theta).
    """
    parser = build_parser()
    args = parser.parse_args()
    if args.output_dir is not None:
        _set_output_dir(args.output_dir)
    return args.input_file, args.num_of_decimals, args.theta


def main():
    input_file_path, num_of_decimals, theta = read_args()

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
        COM_values,
    ) = get_principal_axes(
        isotopologue_names,
        isotopologue_dict,
        n_atoms,
        atom_symbols,
        mol_coordinates,
        mol_dipole,
    )

    if theta:
        theta_data = get_theta_values(
            isotopologue_names,
            com_coordinates,
            COM_values,
            pa_coordinates,
            eigenvalues,
            eigenvectors,
        )
    else:
        theta_data = None

    (
        atom_masses_df,
        rotational_constants_df,
        dipole_components_df,
        com_coordinates_df_dict,
        com_inertias_df_dict,
        eigenvectors_df_dict,
        pa_inertias_df_dict,
        pa_coordinates_df_dict,
        com_values_df,
        theta_df_dict,
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
        COM_values,
        theta_data=theta_data,
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
        com_values_df,
        text_output_path,
        theta_df_dict=theta_df_dict,
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
