#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #
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


def header_creator(some_text: str):
    if not isinstance(some_text, str):
        try:
            some_text = str(some_text)
        except (Exception,):
            raise TypeError("TypeError: Bad input passed to header_creator")
    str_len = len(some_text)
    border_len = str_len + 2
    border_str = "# {} #".format("=" * border_len)
    header_str = "#  {}  #".format(some_text)
    out_str = "{}".format("\n".join([border_str, header_str, border_str]))
    return out_str


def df_text_export(dataframe: pd.DataFrame, n_decimals=6):
    def do_format(some_number):
        nice_number = "{:.{n}f}".format(some_number, n=n_decimals)
        return nice_number

    return dataframe.map(do_format).to_string()


def generate_output_file(
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
):
    # TEXT OUTPUT
    #
    # All numbers are "friendly", that is, not in scientific notation.
    # Full numbers are provided in the .csv output.

    preamble = """The numbers in this output have been limited to {} decimal places.
    The numbers in the corresponding {} file have not.
    Rotational constants are in MHz.
    Dipole moments are in the same units provided in the raw input.""".format(
        num_of_decimals, csv_output_name
    )

    input_section = [header_creator("Raw Input"), input_file.strip()]

    nice_atomic_masses = df_text_export(atom_masses_df, n_decimals=num_of_decimals)
    am_width = max([len(i) for i in nice_atomic_masses.split("\n")])
    nice_atomic_masses = nice_atomic_masses.replace(
        "\nTotal", "\n{}\nTotal".format("-" * am_width)
    )
    atom_mass_section = [header_creator("Atomic Masses"), nice_atomic_masses]

    rotational_constants_section = [
        header_creator("Rotational Constants"),
        df_text_export(rotational_constants_df, n_decimals=num_of_decimals),
    ]

    dipole_components_section = [
        header_creator("Dipole Components"),
        df_text_export(dipole_components_df, n_decimals=num_of_decimals),
    ]

    iso_com_coordinates_entries = []
    iso_com_inertias_entries = []
    iso_eigens_entries = []
    iso_pa_inertias_entries = []
    iso_results_entries = []
    for iso in isotopologue_names:
        iso_com_df = com_coordinates_df_dict[iso].copy(deep=True)
        iso_com_df.index = atom_symbols
        iso_com_coordinate = "{}\n{}".format(
            iso,
            df_text_export(com_coordinates_df_dict[iso], n_decimals=num_of_decimals),
        )
        iso_com_coordinates_entries.append(iso_com_coordinate)

        iso_com_inertia = "{}\n{}".format(
            iso, df_text_export(com_inertias_df_dict[iso], n_decimals=num_of_decimals)
        )
        iso_com_inertias_entries.append(iso_com_inertia)

        iso_eigen_vec = df_text_export(
            eigenvectors_df_dict[iso], n_decimals=num_of_decimals
        )
        formatted_eigen_val = []
        for eigen_val in eigenvalues[iso]:
            formatted_eigen_val.append("{:.{n}}".format(eigen_val, n=num_of_decimals))
        iso_eigen_val = "   ".join(formatted_eigen_val)
        iso_eigen = "{}\n\nEigenvectors\n{}\n\nEigenvalues\n{}".format(
            iso, iso_eigen_vec, iso_eigen_val
        )
        iso_eigens_entries.append(iso_eigen)

        iso_pa_inertia = "{}\n{}".format(
            iso, df_text_export(pa_inertias_df_dict[iso], n_decimals=num_of_decimals)
        )
        iso_pa_inertias_entries.append(iso_pa_inertia)

        iso_pa_df = pa_coordinates_df_dict[iso].copy(deep=True)
        iso_pa_df.index = atom_symbols
        iso_pa_df.loc["Dipole"] = list(dipole_components_df[iso])
        iso_pa_df.loc["Rot. Con."] = list(rotational_constants_df[iso])
        iso_result = "{}\n{}".format(
            iso, df_text_export(iso_pa_df, n_decimals=num_of_decimals)
        )
        width = max([len(i) for i in iso_result.split("\n")])
        iso_result = iso_result.replace("\nDip", "\n{}\nDip".format("-" * width))
        iso_results_entries.append(iso_result)

    iso_name_max_width = max([len(i) for i in isotopologue_names])
    iso_delimiter = "\n\n{}\n".format("=" * min([iso_name_max_width, 10]))

    com_coordinates_section = [
        header_creator("COM Coordinates"),
        iso_delimiter.join(iso_com_coordinates_entries),
    ]

    com_inertias_section = [
        header_creator("COM Inertia Matrix"),
        iso_delimiter.join(iso_com_inertias_entries),
    ]

    eigens_section = [
        header_creator("Eigenvectors & Eigenvalues"),
        iso_delimiter.join(iso_eigens_entries),
    ]

    pa_inertias_section = [
        header_creator("Principal Axes Inertia Matrix"),
        "(All entries should be diagonal)\n",
        iso_delimiter.join(iso_pa_inertias_entries),
    ]

    results_section = [
        header_creator("Principal Axes Coordinates"),
        "(Includes dipole moments and rotational constants, for easy reference.)\n",
        iso_delimiter.join(iso_results_entries),
    ]

    sections_list = [
        preamble,
        "\n".join(input_section),
        "\n".join(atom_mass_section),
        "\n".join(com_coordinates_section),
        "\n".join(com_inertias_section),
        "\n".join(eigens_section),
        "\n".join(pa_inertias_section),
        "\n".join(rotational_constants_section),
        "\n".join(dipole_components_section),
        "\n".join(results_section),
    ]

    sections_delimiter = "\n\n"
    file_string = "{}\n\n".format(sections_delimiter.join(sections_list))

    with open(text_output_path, "w") as outfile:
        outfile.write(file_string)


def generate_csv_output(
    pa_coordinates_df_dict,
    rotational_constants_df,
    dipole_components_df,
    atom_masses_df,
    csv_output_path,
):
    # .csv file
    # Outputs all data without formatting; scientific notation may be used in the values.

    all_pa_coordinates = pd.concat(pa_coordinates_df_dict, axis="columns")

    csv_file_string = "\n".join(
        [
            "Rotational Constants",
            rotational_constants_df.to_csv(),
            "Dipole Components",
            dipole_components_df.to_csv(),
            "Principal Axes Coordinates",
            all_pa_coordinates.to_csv(),
            "Atomic Masses",
            atom_masses_df.to_csv(),
        ]
    )

    with open(csv_output_path, "w") as outfile:
        outfile.write(csv_file_string)


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
