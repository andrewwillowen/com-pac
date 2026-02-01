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
