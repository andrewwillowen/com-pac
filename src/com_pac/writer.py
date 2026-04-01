#!/usr/bin/env python3

# ========= #
#  Imports  #
# ========= #
import pandas as pd


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


def _format_isotopologue_entry(
    iso_name: str, dataframe: pd.DataFrame, decimals: int
) -> str:
    return f"{iso_name}\n{df_text_export(dataframe, n_decimals=decimals)}"


def _insert_separator_before_marker(
    text: str, marker: str, separator_char: str = "-"
) -> str:
    width = max(len(line) for line in text.split("\n"))
    return text.replace(marker, f"\n{separator_char * width}\n{marker}")


def _reindex_df_with_atoms(dataframe: pd.DataFrame, atom_symbols: list) -> pd.DataFrame:
    df = dataframe.copy(deep=True)
    df.index = atom_symbols
    return df


def _format_eigenvalues(eigenvalues: list, decimals: int) -> str:
    formatted = [f"{val:.{decimals}f}" for val in eigenvalues]
    return "   ".join(formatted)


def _build_preamble_section(num_of_decimals, csv_output_name):
    preamble = """The numbers in this output have been limited to {} decimal places.
    The numbers in the corresponding {} file have not.
    Rotational constants are in MHz.
    Dipole moments are in the same units provided in the raw input.""".format(
        num_of_decimals, csv_output_name
    )
    return preamble


def _build_input_section(input_file):
    return "\n".join([header_creator("Raw Input"), input_file.strip()])


def _build_atomic_masses_section(atom_masses_df, num_of_decimals):
    nice_atomic_masses = df_text_export(atom_masses_df, n_decimals=num_of_decimals)
    am_width = max([len(i) for i in nice_atomic_masses.split("\n")])
    nice_atomic_masses = nice_atomic_masses.replace(
        "\nTotal", "\n{}\nTotal".format("-" * am_width)
    )
    return "\n".join([header_creator("Atomic Masses"), nice_atomic_masses])


def _build_rotational_constants_section(rotational_constants_df, num_of_decimals):
    return "\n".join(
        [
            header_creator("Rotational Constants"),
            df_text_export(rotational_constants_df, n_decimals=num_of_decimals),
        ]
    )


def _build_dipole_components_section(dipole_components_df, num_of_decimals):
    return "\n".join(
        [
            header_creator("Dipole Components"),
            df_text_export(dipole_components_df, n_decimals=num_of_decimals),
        ]
    )


def _build_com_values_section(com_values_df, num_of_decimals):
    return "\n".join(
        [
            header_creator("COM Values"),
            df_text_export(com_values_df, n_decimals=num_of_decimals),
        ]
    )


def _build_com_coordinates_section(
    isotopologue_names, com_coordinates_df_dict, atom_symbols, num_of_decimals
):
    iso_com_coordinates_entries = []
    for iso in isotopologue_names:
        iso_com_df = _reindex_df_with_atoms(com_coordinates_df_dict[iso], atom_symbols)
        iso_com_coordinates_entries.append(
            _format_isotopologue_entry(iso, iso_com_df, num_of_decimals)
        )
    iso_name_max_width = max([len(i) for i in isotopologue_names])
    iso_delimiter = "\n\n{}\n".format("=" * min([iso_name_max_width, 10]))
    return "\n".join(
        [
            header_creator("COM Coordinates"),
            iso_delimiter.join(iso_com_coordinates_entries),
        ]
    )


def _build_com_inertias_section(
    isotopologue_names, com_inertias_df_dict, num_of_decimals
):
    iso_com_inertias_entries = []
    for iso in isotopologue_names:
        iso_com_inertias_entries.append(
            _format_isotopologue_entry(iso, com_inertias_df_dict[iso], num_of_decimals)
        )
    iso_name_max_width = max([len(i) for i in isotopologue_names])
    iso_delimiter = "\n\n{}\n".format("=" * min([iso_name_max_width, 10]))
    return "\n".join(
        [
            header_creator("COM Inertia Matrix"),
            iso_delimiter.join(iso_com_inertias_entries),
        ]
    )


def _build_eigens_section(
    isotopologue_names, eigenvectors_df_dict, eigenvalues, num_of_decimals
):
    iso_eigens_entries = []
    for iso in isotopologue_names:
        iso_eigen_vec = df_text_export(
            eigenvectors_df_dict[iso], n_decimals=num_of_decimals
        )
        iso_eigen_val = _format_eigenvalues(eigenvalues[iso], num_of_decimals)
        iso_eigen = "{}\n\nEigenvectors\n{}\n\nEigenvalues\n{}".format(
            iso, iso_eigen_vec, iso_eigen_val
        )
        iso_eigens_entries.append(iso_eigen)
    iso_name_max_width = max([len(i) for i in isotopologue_names])
    iso_delimiter = "\n\n{}\n".format("=" * min([iso_name_max_width, 10]))
    return "\n".join(
        [
            header_creator("Eigenvectors & Eigenvalues"),
            iso_delimiter.join(iso_eigens_entries),
        ]
    )


def _build_pa_inertias_section(
    isotopologue_names, pa_inertias_df_dict, num_of_decimals
):
    iso_pa_inertias_entries = []
    for iso in isotopologue_names:
        iso_pa_inertias_entries.append(
            _format_isotopologue_entry(iso, pa_inertias_df_dict[iso], num_of_decimals)
        )
    iso_name_max_width = max([len(i) for i in isotopologue_names])
    iso_delimiter = "\n\n{}\n".format("=" * min([iso_name_max_width, 10]))
    return "\n".join(
        [
            header_creator("Principal Axes Inertia Matrix"),
            "(All entries should be diagonal)\n",
            iso_delimiter.join(iso_pa_inertias_entries),
        ]
    )


def _build_results_section(
    isotopologue_names,
    pa_coordinates_df_dict,
    dipole_components_df,
    rotational_constants_df,
    atom_symbols,
    num_of_decimals,
    pa_rotation_df_dict=None,
):
    iso_results_entries = []
    for iso in isotopologue_names:
        iso_pa_df = _reindex_df_with_atoms(pa_coordinates_df_dict[iso], atom_symbols)
        iso_pa_df.loc["Dipole"] = list(dipole_components_df[iso])
        iso_pa_df.loc["Rot. Con."] = list(rotational_constants_df[iso])
        iso_result = _format_isotopologue_entry(iso, iso_pa_df, num_of_decimals)
        iso_result = _insert_separator_before_marker(iso_result, "\nDip")
        if pa_rotation_df_dict is not None:
            rotation_angle = pa_rotation_df_dict[iso].iloc[0]["RotationAngle"]
            iso_result += (
                f"\n\nRotation Angle: {rotation_angle:.{num_of_decimals}f} degrees"
            )
        iso_results_entries.append(iso_result)
    iso_name_max_width = max([len(i) for i in isotopologue_names])
    iso_delimiter = "\n\n{}\n".format("=" * min([iso_name_max_width, 10]))
    return "\n".join(
        [
            header_creator("Principal Axes Coordinates"),
            "(Includes dipole moments and rotational constants, for easy reference.)\n",
            iso_delimiter.join(iso_results_entries),
        ]
    )


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
    com_values_df,
    pa_rotation_df_dict,
    text_output_path,
):
    # TEXT OUTPUT
    #
    # All numbers are "friendly", that is, not in scientific notation.
    # Full numbers are provided in the .csv output.

    sections_list = [
        _build_preamble_section(num_of_decimals, csv_output_name),
        _build_input_section(input_file),
        _build_atomic_masses_section(atom_masses_df, num_of_decimals),
        _build_com_values_section(com_values_df, num_of_decimals),
        _build_com_coordinates_section(
            isotopologue_names, com_coordinates_df_dict, atom_symbols, num_of_decimals
        ),
        _build_com_inertias_section(
            isotopologue_names, com_inertias_df_dict, num_of_decimals
        ),
        _build_eigens_section(
            isotopologue_names, eigenvectors_df_dict, eigenvalues, num_of_decimals
        ),
        _build_pa_inertias_section(
            isotopologue_names, pa_inertias_df_dict, num_of_decimals
        ),
        _build_rotational_constants_section(rotational_constants_df, num_of_decimals),
        _build_dipole_components_section(dipole_components_df, num_of_decimals),
        _build_results_section(
            isotopologue_names,
            pa_coordinates_df_dict,
            dipole_components_df,
            rotational_constants_df,
            atom_symbols,
            num_of_decimals,
            pa_rotation_df_dict,
        ),
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
