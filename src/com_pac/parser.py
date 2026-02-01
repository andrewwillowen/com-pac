#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #

import re
import numpy as np


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
