#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #

import re
from sys import argv
from mendeleev import element
import numpy as np
import pandas as pd
from pathlib import Path


def read_args(num_of_decimals):
    """
    Manually parsing arguments...
    """
    # TODO: replace with argparse

    # reading arguments from shell
    if len(argv) < 2:
        print(
            """
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
        """
        )
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


def inertia_matrix(coordinates_array, masses_array):
    matrix = np.zeros((3, 3))
    for axis1 in [0, 1, 2]:
        [axis2, axis3] = [x for x in [0, 1, 2] if x != axis1]
        diagonal = sum(
            [
                (
                    masses_array[i]
                    * (
                        (coordinates_array[i][axis2]) ** 2
                        + (coordinates_array[i][axis3]) ** 2
                    )
                )
                for i in range(len(masses_array))
            ]
        )
        off_diagonal = (-1) * sum(
            [
                masses_array[i]
                * coordinates_array[i][axis2]
                * coordinates_array[i][axis3]
                for i in range(len(masses_array))
            ]
        )
        matrix[axis1, axis1] = diagonal
        matrix[axis2, axis3] = off_diagonal
        matrix[axis3, axis2] = off_diagonal
    return matrix


def inertia_to_rot_const(inertia):
    rot_constant = 505379.0046 / inertia
    return rot_constant


def coordinates_error_message(*args):
    message = """
    There was an error reading in the atomic coordinates.
    
    Proper format of the coordinate section is:
        Coordinates     # comments
        Atom1   x_coor1 y_coor1 z_coor1     # more comments
        Atom2   x_coor2 y_coor2 z_coor2
        ...
        AtomZ   x_coorZ y_coorZ z_coorZ
        (blank line)
    where Atom# is the atomic symbol, and x_coor#, y_coor#, and z_coor# are numeric values.
    
    """
    if len(args) == 0:
        return message
    else:
        return "{}\n\t{}".format(message, "\n\t".join([str(x) for x in args]))


def dipole_error_message(*args):
    message = """
    There was an error reading in the dipole.

    Proper format of the dipole section is:
        Dipole      # comments
        muX muY muZ # more comments
        (blank line)
    where muX, muY, and muZ are numeric values.
    
    """
    if len(args) == 0:
        return message
    else:
        return "{}\n\t{}".format(message, "\n\t".join([str(x) for x in args]))


def isotopologue_error_message(*args):
    message = """
    There was an error reading in the isotopologue masses.
    
    Proper format of the isotopologue section is:
        Isotopologues   # comment
        mass1 mass2 mass3 ... massZ iso000  # more comments
        mass1 mass2 mass3 ... massZ iso001  
        ...
        mass1 mass2 mass3 ... massZ isoZZZ
        (blank line)
    where mass# is the atomic mass number of the isotope,
    and iso### is the isotopologue label to use in the output.
    
    The number of atoms (lines) in the Coordinates section 
    MUST MATCH the number of mass numbers in each line of 
    the Isotopologues section!!
    """
    if len(args) == 0:
        return message
    else:
        return "{}\n\t{}".format(message, "\n\t".join([str(x) for x in args]))


def get_coordinate_matches(input_file):
    # TODO: Check to make sure there aren't multiple coordinate sections
    #       in the input file.
    try:
        coordinate_matches = re.split(
            "(?m)^coordinates", input_file, flags=re.IGNORECASE
        )[1]
    except (Exception,):
        raise ValueError(
            coordinates_error_message(
                'Could not find line starting with "coordinates" (case insensitive).'
            )
        )

    return coordinate_matches


def get_coordinate_section(coordinate_matches):
    coordinate_sections: list = re.split(r"\n\s*\n", coordinate_matches)

    if len(coordinate_sections) < 2:
        raise ValueError(
            coordinates_error_message(
                "Could not find end of coordinates section; make sure there is a blank line at the end of the section."
            )
        )

    return coordinate_sections[0]


def get_coordinate_info(coordinate_section):
    try:
        coordinate_lines = [
            i
            for i in coordinate_section.split("\n")[1:]
            if ((not i.isspace()) and (i != ""))
        ]
        coordinate_list = [x.split() for x in coordinate_lines]

        atom_symbols = [x[0] for x in coordinate_list]
        n_atoms = len(atom_symbols)
        atom_numbering = [atom_symbols[x] + str(x + 1) for x in range(0, n_atoms)]

        mol_coordinates = np.array(
            [[float(x[1]), float(x[2]), float(x[3])] for x in coordinate_list]
        )
    except (Exception,) as exc:
        raise ValueError(coordinates_error_message()) from exc

    return n_atoms, atom_symbols, mol_coordinates, atom_numbering


def parse_input_coordinate_section(input_file):
    # reading in atoms, coordinates

    coordinate_matches = get_coordinate_matches(input_file)
    coordinate_section = get_coordinate_section(coordinate_matches)

    return get_coordinate_info(coordinate_section)


def get_dipole_matches(input_file):
    # TODO: Check to make sure there aren't multiple dipole sections
    #       in the input file.
    try:
        dipole_matches = re.split("(?m)^dipole", input_file, flags=re.IGNORECASE)[1]
    except (Exception,):
        raise ValueError(
            dipole_error_message(
                'Could not find line starting with "dipole" (case insensitive).'
            )
        )

    return dipole_matches


def get_dipole_section(dipole_matches):
    dipole_sections: list = re.split(r"\n\s*\n", dipole_matches)

    if len(dipole_sections) < 2:
        raise ValueError(
            dipole_error_message(
                "Could not find end of dipole section; make sure there is a blank line at the end of the section."
            )
        )

    return dipole_sections[0]


def get_dipole_info(dipole_section):
    try:
        dipole_line = dipole_section.split("\n")[1]
        dipole_list = dipole_line.split()
        mol_dipole = np.array(
            [float(dipole_list[0]), float(dipole_list[1]), float(dipole_list[2])]
        )
    except (Exception,) as exc:
        raise ValueError(dipole_error_message()) from exc

    return mol_dipole


def parse_input_dipole_section(input_file):
    # reading in dipole
    dipole_matches = get_dipole_matches(input_file)
    dipole_section = get_dipole_section(dipole_matches)

    return get_dipole_info(dipole_section)


def get_isotopologue_matches(input_file):
    try:
        isotopologue_matches = re.split(
            "isotopologues", input_file, flags=re.IGNORECASE
        )[1]
    except (Exception,):
        raise ValueError(
            isotopologue_error_message(
                'Could not find line starting with "isotopologues" (case insensitive)'
            )
        )

    return isotopologue_matches


def get_isotopologue_section(isotopologue_matches):
    isotopologue_sections: list = re.split(r"\n\s*\n", isotopologue_matches)
    if len(isotopologue_sections) < 2:
        raise ValueError(
            isotopologue_error_message(
                "Could not find end of isotopologues section; make sure there is a blank line at the end of the section."
            )
        )

    return isotopologue_sections[0]


def get_isotopologue_info(isotopologue_section, n_atoms):
    try:
        isotopologue_lines = [
            i
            for i in isotopologue_section.split("\n")[1:]
            if ((not i.isspace()) and (i != ""))
        ]
        isotopologue_list = [x.split() for x in isotopologue_lines]
        isotopologue_names = [x[n_atoms] for x in isotopologue_list]

        isotopologue_dict = {
            x[n_atoms]: [int(y) for y in x[:n_atoms]] for x in isotopologue_list
        }
        # isotopologue_names = [key for key in isotopologue_dict.keys()]
    except (Exception,) as exc:
        raise ValueError(isotopologue_error_message()) from exc

    duplicate_names = [
        i for i in set(isotopologue_names) if isotopologue_names.count(i) > 1
    ]
    if duplicate_names:
        raise ValueError(
            isotopologue_error_message(
                f"Isotopologue section contains duplicate labels: {duplicate_names}"
            )
        )
    return isotopologue_names, isotopologue_dict


def parse_input_isotopologue_section(input_file, n_atoms):
    # reading in isotopologues and masses
    isotopologue_matches = get_isotopologue_matches(input_file)
    isotopologue_section = get_isotopologue_section(isotopologue_matches)

    return get_isotopologue_info(isotopologue_section, n_atoms)


def check_mass_numbers_are_valid(isotopologue_dict, atom_symbols):
    # TODO: Add check that mass number provided in isotopologue section is
    #       indeed a valid mass number for the corresponding atom in the
    #       atom_symbols list. Raise an explanatory error if not.

    pass


def parse_input_file(input_file):
    n_atoms, atom_symbols, mol_coordinates, atom_numbering = (
        parse_input_coordinate_section(input_file)
    )

    mol_dipole = parse_input_dipole_section(input_file)

    isotopologue_names, isotopologue_dict = parse_input_isotopologue_section(
        input_file, n_atoms
    )

    # Raises an explanatory exception if not, silently continues if yes.
    check_mass_numbers_are_valid(isotopologue_dict, atom_symbols)

    return (
        isotopologue_names,
        isotopologue_dict,
        n_atoms,
        atom_symbols,
        mol_coordinates,
        mol_dipole,
        atom_numbering,
    )


def get_principal_axes(
    isotopologue_names,
    isotopologue_dict,
    n_atoms,
    atom_symbols,
    mol_coordinates,
    mol_dipole,
):
    rotational_constants = {}
    atom_masses = {}
    com_coordinates = {}
    com_inertias = {}
    eigenvectors = {}
    eigenvalues = {}
    pa_dipoles = {}
    pa_coordinates = {}
    pa_inertias = {}
    bad_diagonal_warnings = {}

    for iso in isotopologue_names:
        atom_mass_numbers = isotopologue_dict[iso]
        isotopes = []
        for atom in range(0, n_atoms):
            isotopes.append([atom_symbols[atom], atom_mass_numbers[atom]])
        masses = []
        for label, mass_number in isotopes:
            mass = None
            for isotope in element(label).isotopes:
                if isotope.mass_number == mass_number:
                    mass = isotope.mass
            if mass is None:
                raise ValueError(
                    "\n    Isotopic mass not found for {} with mass number {}\n".format(
                        label, mass_number
                    )
                )
            else:
                masses.append(mass)
        mol_masses = np.array(masses)
        atom_masses[iso] = mol_masses
        mol_COM = (1 / mol_masses.sum()) * np.array(
            [mol_masses[x] * mol_coordinates[x] for x in range(0, n_atoms)]
        ).sum(axis=0)
        com_coordinate = np.array([x - mol_COM for x in mol_coordinates])
        com_coordinates[iso] = com_coordinate
        com_inertia = inertia_matrix(com_coordinate, mol_masses)
        com_inertias[iso] = com_inertia
        evals, evecs = np.linalg.eig(com_inertia)
        sort_key = evals.argsort()[::1]
        evals = evals[sort_key]
        evecs = evecs[:, sort_key]
        eigenvectors[iso] = evecs
        eigenvalues[iso] = evals
        pa_coordinate = np.dot(com_coordinate, evecs)
        pa_coordinates[iso] = pa_coordinate
        pa_inertia = inertia_matrix(pa_coordinate, mol_masses)
        pa_inertias[iso] = pa_inertia
        if not np.allclose(pa_inertia, np.diag(evals)):
            bad_diagonal_pas = """WARNING! The inertia matrix calculated using the principal axes system 
    is not diagonal for {}""".format(
                iso
            )
            print(bad_diagonal_pas)
            bad_diagonal_warnings[iso] = bad_diagonal_pas
        pa_dipoles[iso] = abs(np.dot(mol_dipole, evecs))
        rotational_constants[iso] = list(map(inertia_to_rot_const, evals))

    return (
        atom_masses,
        rotational_constants,
        pa_dipoles,
        pa_coordinates,
        pa_inertias,
        com_coordinates,
        com_inertias,
        eigenvectors,
        eigenvalues,
    )


def get_dataframes(
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
):
    # Atomic masses
    atom_masses_df = pd.DataFrame.from_dict(atom_masses)
    atom_masses_df["Atom"] = atom_symbols
    atom_masses_df = atom_masses_df.set_index("Atom")
    atom_masses_df.loc["Total"] = [
        atom_masses_df[i].sum() for i in atom_masses_df.columns
    ]

    # Rotational constants
    rotational_constants_df = pd.DataFrame.from_dict(rotational_constants)
    rotational_constants_df["Axis"] = ["A", "B", "C"]
    rotational_constants_df = rotational_constants_df.set_index("Axis")

    # Dipole moments
    dipole_components_df = pd.DataFrame.from_dict(pa_dipoles)
    dipole_components_df.index = ["mu_A", "mu_B", "mu_C"]

    com_coordinates_df_dict = {}
    com_inertias_df_dict = {}
    eigenvectors_df_dict = {}
    pa_inertias_df_dict = {}
    pa_coordinates_df_dict = {}
    for iso in isotopologue_names:
        com_coordinates_df = pd.DataFrame(
            columns=["x", "y", "z"], data=com_coordinates[iso]
        )
        com_coordinates_df["Atom"] = atom_numbering
        com_coordinates_df = com_coordinates_df.set_index("Atom")
        com_coordinates_df_dict[iso] = com_coordinates_df

        com_inertias_df = pd.DataFrame(columns=["x", "y", "z"], data=com_inertias[iso])
        com_inertias_df["Axis"] = ["x", "y", "z"]
        com_inertias_df = com_inertias_df.set_index("Axis")
        com_inertias_df_dict[iso] = com_inertias_df

        eigenvector_df = pd.DataFrame(columns=["1", "2", "3"], data=eigenvectors[iso])
        eigenvector_df["Axis"] = ["x", "y", "z"]
        eigenvector_df = eigenvector_df.set_index("Axis")
        eigenvectors_df_dict[iso] = eigenvector_df

        pa_inertias_df = pd.DataFrame(columns=["a", "b", "c"], data=pa_inertias[iso])
        pa_inertias_df["Axis"] = ["a", "b", "c"]
        pa_inertias_df = pa_inertias_df.set_index("Axis")
        pa_inertias_df_dict[iso] = pa_inertias_df

        pa_coordinates_df = pd.DataFrame(
            columns=["a", "b", "c"], data=pa_coordinates[iso]
        )
        pa_coordinates_df["Atom"] = atom_numbering
        pa_coordinates_df = pa_coordinates_df.set_index("Atom")
        pa_coordinates_df_dict[iso] = pa_coordinates_df

    return (
        atom_masses_df,
        rotational_constants_df,
        dipole_components_df,
        com_coordinates_df_dict,
        com_inertias_df_dict,
        eigenvectors_df_dict,
        pa_inertias_df_dict,
        pa_coordinates_df_dict,
    )


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
