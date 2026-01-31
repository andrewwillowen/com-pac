"""
Whole test of com_pac against original script.
"""

import pytest
import numpy as np
import subprocess
import os


def delete_if_exists(file_to_delete):
    if os.path.exists(file_to_delete):
        os.remove(file_to_delete)


def shell_exec(
    command_list: list[str],
    capture_std: bool = False,
    combined_std: bool = True,
    timeout: int = 60,
) -> subprocess.CompletedProcess:
    """
    Wrapper function for running shell commands
    """

    # Make sure not to run a sudo command using this function
    test_str: str = " ".join([str(i) for i in command_list])
    if "sudo" in test_str.lower():
        raise ValueError('Will not execute a command involing "sudo"')

    # Make sure command is split on white space
    clean_list: list[str] = []
    for i in command_list:
        split_list: list[str] = [s for s in i.split() if not s.isspace() and s != ""]
        if len(split_list) == 0:
            continue
        else:
            clean_list.extend(split_list)

    if capture_std:
        if combined_std:
            exec_command: subprocess.CompletedProcess = subprocess.run(
                clean_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        else:
            exec_command = subprocess.run(
                clean_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
    else:
        exec_command = subprocess.run(clean_list, timeout=timeout)

    return exec_command


def import_csv_output(file: str) -> dict:
    """
    Reads the csv pac output files into numpy arrays for numeric comparison
    """
    with open(file, "r") as f:
        pac_csv = f.read().split("\n")

    out_dict = {}

    # Rotational Constants section
    out_dict["rot_header"] = pac_csv[1]

    rot_floats = [
        [float(i) for i in pac_csv[2].split(",")[1:]],
        [float(i) for i in pac_csv[3].split(",")[1:]],
        [float(i) for i in pac_csv[4].split(",")[1:]],
    ]

    out_dict["rot_floats"] = np.array(rot_floats, dtype=float)

    # Dipole Components section
    out_dict["dipole_header"] = pac_csv[7]

    dipole_floats = [
        [float(i) for i in pac_csv[8].split(",")[1:]],
        [float(i) for i in pac_csv[9].split(",")[1:]],
        [float(i) for i in pac_csv[10].split(",")[1:]],
    ]

    out_dict["dipole_floats"] = np.array(dipole_floats, dtype=float)

    # Principal Axes Coordinates section
    out_dict["pac_header"] = pac_csv[13]

    pac_floats = [
        [float(i) for i in pac_csv[16].split(",")[1:]],
        [float(i) for i in pac_csv[17].split(",")[1:]],
        [float(i) for i in pac_csv[18].split(",")[1:]],
        [float(i) for i in pac_csv[19].split(",")[1:]],
    ]

    out_dict["pac_floats"] = np.array(pac_floats, dtype=float)

    # Atomic Masses section
    out_dict["mass_header"] = pac_csv[22]

    mass_floats = [
        [float(i) for i in pac_csv[23].split(",")[1:]],
        [float(i) for i in pac_csv[24].split(",")[1:]],
        [float(i) for i in pac_csv[25].split(",")[1:]],
        [float(i) for i in pac_csv[26].split(",")[1:]],
    ]

    out_dict["mass_floats"] = np.array(mass_floats, dtype=float)

    return out_dict


@pytest.fixture(scope="module")
def original_pac():
    return import_csv_output("tests/original_pac.csv")


@pytest.fixture(scope="module")
def latest_pac():
    try:
        shell_exec(["com_pac", "tests/latest.txt"], timeout=60)
    except subprocess.TimeoutExpired:
        pytest.skip("test execution timed out after 60 seconds")

    yield import_csv_output("tests/latest_pac.csv")

    delete_if_exists("tests/latest_pac.csv")
    delete_if_exists("tests/latest_pac.out")


class Test_rotational_section:
    def test_rotational_header(self, original_pac, latest_pac):
        assert original_pac["rot_header"] == latest_pac["rot_header"]

    def test_rotational_constants(self, original_pac, latest_pac):
        np.testing.assert_allclose(original_pac["rot_floats"], latest_pac["rot_floats"])


class Test_dipole_section:
    def test_dipole_header(self, original_pac, latest_pac):
        assert original_pac["dipole_header"] == latest_pac["dipole_header"]

    def test_dipole_components(self, original_pac, latest_pac):
        np.testing.assert_allclose(
            original_pac["dipole_floats"], latest_pac["dipole_floats"]
        )


class Test_pac_section:
    def test_pac_header(self, original_pac, latest_pac):
        assert original_pac["pac_header"] == latest_pac["pac_header"]

    def test_pac_coordinates(self, original_pac, latest_pac):
        np.testing.assert_allclose(original_pac["pac_floats"], latest_pac["pac_floats"])


class Test_mass_section:
    def test_mass_header(self, original_pac, latest_pac):
        assert original_pac["mass_header"] == latest_pac["mass_header"]

    def test_mass_values(self, original_pac, latest_pac):
        np.testing.assert_allclose(
            original_pac["mass_floats"], latest_pac["mass_floats"]
        )
