"""
Unit tests for functions in core.py
"""

from mendeleev.fetch import fetch_table
from mendeleev.mendeleev import element

import pytest
import numpy as np
from com_pac.core import (
    inertia_matrix,
    inertia_to_rot_const,
    coordinates_error_message,
    dipole_error_message,
    isotopologue_error_message,
    get_coordinate_matches,
    get_coordinate_section,
    get_coordinate_info,
    parse_input_coordinate_section,
    get_dipole_matches,
    get_dipole_section,
    get_dipole_info,
    parse_input_dipole_section,
    get_isotopologue_matches,
    get_isotopologue_section,
    get_isotopologue_info,
    parse_input_isotopologue_section,
    parse_input_file,
)


class Test_read_args:
    """
    Test core.read_args()

    Not sure how to test this yet.
    Will wait until implementing ~argparse.
    """

    pass


class Test_inertia_matrix:
    """
    Test core.inertia_matrix()
    """

    # The following parameterize block was generated as follows:
    # >>> import numpy as np
    # >>> from com_pac.core import inertia_matrix
    # >>> from random import randint
    # >>> n_atoms = set()
    # >>> while len(n_atoms) < 6:
    # ...     n_atoms.add(randint(2, 30))
    # ...
    # >>> rng = np.random.default_rng()
    # >>> params = []
    # >>> for n in n_atoms:
    # ...     coordinates = 10*rng.random((n, 3)) - 5
    # ...     masses = 100*rng.random(n)
    # ...     expected = inertia_matrix(coordinates, masses)
    # ...     params.append((coordinates, masses, expected))
    # ...
    # >>> params
    #
    # and then replacing `array` with `np.array`.
    @pytest.mark.parametrize(
        "coordinates,masses,expected",
        [
            (
                np.array(
                    [
                        [-3.99787168, -3.64611009, -0.79857223],
                        [3.82041845, 2.59015573, -1.72846941],
                        [-4.14737334, 2.22363115, 0.53194212],
                        [2.27960531, -1.5805131, 4.57701207],
                        [2.26206522, 1.89713985, -2.23942121],
                        [-1.39925724, -3.85145304, -2.41527489],
                    ]
                ),
                np.array(
                    [
                        81.56014294,
                        70.45185325,
                        38.77692425,
                        23.499187,
                        59.64661262,
                        7.02675499,
                    ]
                ),
                np.array(
                    [
                        [3232.1414003, -1737.59407925, 323.60723383],
                        [-1737.59407925, 4545.80479967, 390.10606476],
                        [323.60723383, 390.10606476, 5566.20277886],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [4.97560498, -1.52711508, -4.6439251],
                        [-2.24354171, 1.46618232, 3.7042745],
                        [-3.36868809, 3.44637206, 2.35270223],
                        [-2.74559909, -1.83363707, 4.32503685],
                        [3.97839912, 3.50814032, 1.35067896],
                        [2.83399728, 4.29457292, -2.56178475],
                        [-3.47524305, -1.79860119, -2.91935792],
                        [-3.1576719, 1.45821783, 3.66951168],
                        [2.75273824, -2.46957419, -2.42323765],
                        [1.23559648, -4.10342235, 0.0342625],
                        [0.55863261, 4.49406731, -3.45222319],
                        [3.20436179, 2.7530986, 3.71186733],
                        [-2.5281156, -3.02525142, -2.32624522],
                    ]
                ),
                np.array(
                    [
                        66.39567115,
                        25.42758243,
                        12.10387171,
                        56.5553822,
                        99.34941121,
                        1.49050769,
                        3.3605794,
                        7.62411296,
                        57.64534152,
                        39.65313212,
                        93.76464895,
                        28.58071292,
                        59.6945278,
                    ]
                ),
                np.array(
                    [
                        [10897.5136067, -1297.86596882, 1916.87183564],
                        [-1297.86596882, 10638.94516674, -368.24168553],
                        [1916.87183564, -368.24168553, 10734.69310443],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [-4.49293915, 4.03885847, 1.65023071],
                        [2.73773913, -1.38385831, -0.29478332],
                        [-3.6569788, -3.04826657, 2.38729837],
                        [3.56138384, 4.21194603, -0.93437524],
                        [-0.28509864, 1.59014592, 3.20042278],
                        [-2.71822367, 1.56476573, -4.26303531],
                        [0.22374025, 0.02508884, -1.59555182],
                        [-4.73572977, -4.8476358, -0.34839849],
                        [-1.86067362, -3.61194145, -2.33933639],
                        [-4.08346961, -3.40129955, 1.2458658],
                        [-3.59163869, 4.36481191, 1.93637673],
                        [3.67824905, 4.30448273, 3.7315571],
                        [-0.45674827, 2.45193433, -1.06657915],
                        [0.83681692, -3.26198859, 2.3793701],
                    ]
                ),
                np.array(
                    [
                        24.44258036,
                        51.31191312,
                        27.02860376,
                        42.04959691,
                        75.99514217,
                        26.94976933,
                        39.17214241,
                        13.39067499,
                        34.01789416,
                        73.33494646,
                        15.09268609,
                        9.23540266,
                        15.97082643,
                        49.11549009,
                    ]
                ),
                np.array(
                    [
                        [6849.15179067, -1457.50492045, 445.12063352],
                        [-1457.50492045, 6391.06906719, 120.70623906],
                        [445.12063352, 120.70623906, 8414.81403036],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [3.96390746, -0.67009673, -3.78638909],
                        [-0.97508314, 0.71709878, -2.18910578],
                        [-1.63892115, -3.77498069, 3.35093019],
                        [-0.34838011, 1.30907539, 4.39468057],
                        [-0.87651819, -0.94048715, -0.75813082],
                        [2.35792482, -1.75711675, 0.46681507],
                        [-2.20548831, 3.3729211, -1.59562694],
                        [-3.15865731, -4.77142873, -0.41150384],
                        [-1.51268111, 1.61447757, 1.8560678],
                        [-3.23860143, -2.81544905, -4.53942349],
                        [-0.60865633, 1.67477126, 0.50844679],
                        [1.09688226, -3.70537924, 3.40619787],
                        [-0.6062438, 1.28309456, 2.62646272],
                        [3.42870482, 0.19124565, -4.14686582],
                        [0.24266555, 4.56107419, 4.40228434],
                    ]
                ),
                np.array(
                    [
                        46.68630678,
                        99.49174265,
                        57.49908941,
                        55.73556217,
                        54.77612185,
                        23.69534488,
                        6.80554803,
                        80.89043075,
                        96.47992045,
                        54.00339023,
                        36.67616018,
                        45.93546425,
                        58.63812302,
                        68.69535095,
                        0.45328784,
                    ]
                ),
                np.array(
                    [
                        [11067.32871999, -1284.86304906, 1084.18125054],
                        [-1284.86304906, 10206.63369192, -280.38973705],
                        [1084.18125054, -280.38973705, 8237.79128261],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [3.68298295, -1.37269109, 2.4642938],
                        [0.6112734, -4.43929693, -2.7431966],
                        [-0.96091245, 0.21938073, 2.78211631],
                        [2.31280884, -0.78181226, 0.81243993],
                        [-2.19369393, -1.90879107, 2.35045596],
                        [-3.62836528, -2.17413976, 4.53140054],
                        [3.6825248, -1.07700952, 4.145048],
                        [-3.33631858, 3.85992344, -4.35136979],
                        [-3.4395121, -1.90364688, -2.8220233],
                        [2.57566903, 1.61392781, 2.01030047],
                        [-0.03451005, 1.05264577, 2.53813792],
                        [3.93591591, -2.69984733, -0.65188013],
                        [-0.58850111, 4.68941346, -1.45594462],
                        [-0.21489326, 4.74768585, 0.72561243],
                        [2.39220695, 4.66874347, -3.90243994],
                        [4.96800812, -4.77728687, 0.91040739],
                        [1.57130354, -3.22044076, -3.5916896],
                        [-0.29918607, 2.1079678, -4.30237633],
                        [-0.8362887, -4.41980221, 3.7140988],
                        [4.74230899, 4.11001456, -1.28491623],
                    ]
                ),
                np.array(
                    [
                        59.42932656,
                        72.27199283,
                        79.22695787,
                        30.10260956,
                        81.22668949,
                        99.73264266,
                        32.3263043,
                        45.94425458,
                        77.42637648,
                        77.90308513,
                        32.81125,
                        26.18119917,
                        45.76544488,
                        83.60650457,
                        31.96763481,
                        0.90581103,
                        51.71909441,
                        35.89846196,
                        62.79934048,
                        36.11486306,
                    ]
                ),
                np.array(
                    [
                        [19399.69694079, -1166.30038708, 479.30043135],
                        [-1166.30038708, 16271.3761811, 2311.98305113],
                        [479.30043135, 2311.98305113, 16671.13308675],
                    ]
                ),
            ),
            (
                np.array(
                    [
                        [-1.93734895, -2.63265954, 4.47319237],
                        [2.52541157, -4.63926297, 1.5031159],
                        [0.61809639, -0.47564114, -3.10892042],
                        [4.6517968, 0.22819689, -0.69415857],
                        [1.88146161, -0.31312459, -4.62432865],
                        [-1.84766667, -3.27297568, 4.79377011],
                        [2.78530933, 1.71524513, 1.2197056],
                        [-2.66193302, 3.31612525, -4.82718247],
                        [1.58219188, -1.12735441, -4.96497088],
                        [0.81543365, -3.00270805, -0.42590554],
                        [-0.25497613, -3.4886161, -0.34980678],
                        [-3.86653908, 0.95481952, 1.60967454],
                        [2.79850643, -3.33354572, 1.48022283],
                        [-2.24949682, 0.39455737, 1.93785538],
                        [-4.34266128, 3.97990183, 3.00491918],
                        [1.01248152, 1.76677232, 1.97426829],
                        [-4.87744645, -4.11757653, 1.56527998],
                        [0.23006163, 3.88122187, 0.48648778],
                        [-4.84379227, 0.98660212, 1.74231152],
                        [2.09221312, 2.30311194, -4.15832087],
                        [3.21641853, -3.35653076, -4.15893557],
                        [-2.25505907, -4.22427013, 4.49797067],
                        [4.92876124, -2.42490202, -3.01027028],
                        [3.25837938, -4.02027853, -4.30231197],
                        [-2.88829288, 4.18723646, 4.19084859],
                    ]
                ),
                np.array(
                    [
                        64.22493155,
                        1.45673541,
                        5.28084363,
                        74.48745809,
                        67.8077349,
                        53.46849713,
                        56.19106865,
                        22.55876519,
                        70.60279363,
                        67.79123668,
                        0.78301868,
                        37.320597,
                        65.57367055,
                        31.4189891,
                        8.03536251,
                        42.13122716,
                        65.52250683,
                        96.88313465,
                        96.12195345,
                        56.75508311,
                        3.04330345,
                        76.10181566,
                        60.29418345,
                        82.17395582,
                        59.67869168,
                    ]
                ),
                np.array(
                    [
                        [23473.29957642, 1001.59416876, 7456.76700785],
                        [1001.59416876, 25026.47335396, 383.56115821],
                        [7456.76700785, 383.56115821, 22148.19342006],
                    ]
                ),
            ),
        ],
    )
    def test_inertia_matrix_rand(self, coordinates, masses, expected):
        value = inertia_matrix(coordinates, masses)
        np.testing.assert_allclose(value, expected)


class Test_inertia_to_rot_const:
    @pytest.mark.parametrize(
        "inertia, expected",
        [
            (np.float64(884870.6315160872), np.float64(0.5711332104379058)),
            (np.float64(559120.6814956776), np.float64(0.903881793905538)),
            (np.float64(50143.26041790018), np.float64(10.078702509332429)),
            (np.float64(877851.8579169718), np.float64(0.5756996468620555)),
            (np.float64(356444.735124563), np.float64(1.417832709531789)),
            (np.float64(598210.5345886707), np.float64(0.8448179618693916)),
        ],
    )
    def test_inertia_to_rot_const_rand(self, inertia, expected):
        value = inertia_to_rot_const(inertia)
        np.testing.assert_allclose(value, expected)


class Test_coordinates_error_message:
    def test_exact_message(self):
        expected_message: str = (
            "\n    There was an error reading in the atomic coordinates.\n    \n    Proper format of the coordinate section is:\n        Coordinates     # comments\n        Atom1   x_coor1 y_coor1 z_coor1     # more comments\n        Atom2   x_coor2 y_coor2 z_coor2\n        ...\n        AtomZ   x_coorZ y_coorZ z_coorZ\n        (blank line)\n    where Atom# is the atomic symbol, and x_coor#, y_coor#, and z_coor# are numeric values.\n    \n    "
        )
        actual_message = coordinates_error_message()
        assert expected_message == actual_message

    def test_exact_message_with_args(self):
        arg1: str = "This is the first line appended to the message"
        arg2: str = "This is the second line appended to the message"
        arg3: str = "This is the last line appended to the message"
        expected_message: str = (
            "\n    There was an error reading in the atomic coordinates.\n    \n    Proper format of the coordinate section is:\n        Coordinates     # comments\n        Atom1   x_coor1 y_coor1 z_coor1     # more comments\n        Atom2   x_coor2 y_coor2 z_coor2\n        ...\n        AtomZ   x_coorZ y_coorZ z_coorZ\n        (blank line)\n    where Atom# is the atomic symbol, and x_coor#, y_coor#, and z_coor# are numeric values.\n    \n    \n\tThis is the first line appended to the message\n\tThis is the second line appended to the message\n\tThis is the last line appended to the message"
        )
        actual_message = coordinates_error_message(arg1, arg2, arg3)
        assert expected_message == actual_message


class Test_dipole_error_message:
    def test_exact_message(self):
        expected_message: str = (
            "\n    There was an error reading in the dipole.\n\n    Proper format of the dipole section is:\n        Dipole      # comments\n        muX muY muZ # more comments\n        (blank line)\n    where muX, muY, and muZ are numeric values.\n    \n    "
        )
        actual_message = dipole_error_message()
        assert expected_message == actual_message

    def test_exact_message_with_args(self):
        arg1: str = "This is the first line appended to the message"
        arg2: str = "This is the second line appended to the message"
        arg3: str = "This is the last line appended to the message"
        expected_message: str = (
            "\n    There was an error reading in the dipole.\n\n    Proper format of the dipole section is:\n        Dipole      # comments\n        muX muY muZ # more comments\n        (blank line)\n    where muX, muY, and muZ are numeric values.\n    \n    \n\tThis is the first line appended to the message\n\tThis is the second line appended to the message\n\tThis is the last line appended to the message"
        )
        actual_message = dipole_error_message(arg1, arg2, arg3)
        assert expected_message == actual_message


class Test_isotopologue_error_message:
    def test_exact_message(self):
        expected_message: str = (
            "\n    There was an error reading in the isotopologue masses.\n    \n    Proper format of the isotopologue section is:\n        Isotopologues   # comment\n        mass1 mass2 mass3 ... massZ iso000  # more comments\n        mass1 mass2 mass3 ... massZ iso001  \n        ...\n        mass1 mass2 mass3 ... massZ isoZZZ\n        (blank line)\n    where mass# is the atomic mass number of the isotope,\n    and iso### is the isotopologue label to use in the output.\n    \n    The number of atoms (lines) in the Coordinates section \n    MUST MATCH the number of mass numbers in each line of \n    the Isotopologues section!!\n    "
        )
        actual_message = isotopologue_error_message()
        assert expected_message == actual_message

    def test_exact_message_with_args(self):
        arg1: str = "This is the first line appended to the message"
        arg2: str = "This is the second line appended to the message"
        arg3: str = "This is the last line appended to the message"
        expected_message: str = (
            "\n    There was an error reading in the isotopologue masses.\n    \n    Proper format of the isotopologue section is:\n        Isotopologues   # comment\n        mass1 mass2 mass3 ... massZ iso000  # more comments\n        mass1 mass2 mass3 ... massZ iso001  \n        ...\n        mass1 mass2 mass3 ... massZ isoZZZ\n        (blank line)\n    where mass# is the atomic mass number of the isotope,\n    and iso### is the isotopologue label to use in the output.\n    \n    The number of atoms (lines) in the Coordinates section \n    MUST MATCH the number of mass numbers in each line of \n    the Isotopologues section!!\n    \n\tThis is the first line appended to the message\n\tThis is the second line appended to the message\n\tThis is the last line appended to the message"
        )
        actual_message = isotopologue_error_message(arg1, arg2, arg3)
        assert expected_message == actual_message


# Tests for coordinate-section functions
# --------------------------------------
# Proper format of the coordinate section is:
#     Coordinates     # comments
#     Atom1   x_coor1 y_coor1 z_coor1     # more comments
#     Atom2   x_coor2 y_coor2 z_coor2
#     ...
#     AtomZ   x_coorZ y_coorZ z_coorZ
#     (blank line)
# where Atom# is the atomic symbol, and x_coor#, y_coor#, and z_coor# are numeric values.

# Want to utilize hypothesis to help with this.
element_df = fetch_table("elements")
atom_symbols = tuple(element_df.symbol.values)
atom_names = tuple(element_df.name.values)


# fixture? Actually, this should probably be it's own function in the core
# to help with exporting data to a file! Then use Hypothesis strategies to
# fill in the values.
def atom_line(atom, x_coor, y_coor, z_coor, comment=None, delimiter: str = " "):
    if comment is None:
        return delimiter.join(str(i) for i in (atom, x_coor, y_coor, z_coor))

    return delimiter.join(str(i) for i in (atom, x_coor, y_coor, z_coor, comment))


simple_coordinate_section = """Simple coordinates
Coordinates
H 0.0 0.0 0.0

Other stuff
"""


class Test_simple_coordinates_section:
    @pytest.mark.dependency(name="simple_coord_match")
    def test_simple_get_coordinate_matches(self):
        # NOTE: Currently, the "blank line" must be truly blank! No white space!!
        result = get_coordinate_matches(simple_coordinate_section)
        assert result == "\nH 0.0 0.0 0.0\n\nOther stuff\n"

    @pytest.mark.dependency(name="simple_coord_sect")
    def test_simple_get_coordinate_section(self):
        result = get_coordinate_section("\nH 0.0 0.0 0.0\n\nOther stuff\n")
        assert result == "\nH 0.0 0.0 0.0"

    @pytest.mark.dependency(name="simple_coord_info")
    def test_simple_get_coordinate_info(self):
        n_atoms, atom_symbols, mol_coordinates, atom_numbering = get_coordinate_info(
            "\nH 0.0 0.0 0.0"
        )
        assert all(
            (
                n_atoms == 1,
                atom_symbols == ["H"],
                np.allclose(mol_coordinates, np.array([[0.0, 0.0, 0.0]], dtype=float)),
                atom_numbering == ["H1"],
            )
        )


@pytest.fixture
def multiple_atoms_input():
    atom1 = atom_line("H", 0.0, 0.0, 0.0)
    atom2 = atom_line("H", 1.0, -1.0, 0.0, comment="# Second atom")
    input_text = f"Coordinates\n{atom1}\n{atom2}\n\nOther stuff"
    return input_text


@pytest.fixture
def multiple_atoms_matched():
    return "\nH 0.0 0.0 0.0\nH 1.0 -1.0 0.0 # Second atom\n\nOther stuff"


@pytest.fixture
def multiple_coords(multiple_atoms_input):
    input_text = f"{multiple_atoms_input}\n\n{multiple_atoms_input}"
    return input_text


@pytest.fixture
def multiple_coords_matched():
    return "\nH 0.0 0.0 0.0\nH 1.0 -1.0 0.0 # Second atom\n\nOther stuff\n\n"


@pytest.mark.dependency(name="coord_matches", depends=["simple_coord_match"])
class Test_get_coordinate_matches:
    def test_multiple_atoms(self, multiple_atoms_input, multiple_atoms_matched):
        result = get_coordinate_matches(multiple_atoms_input)
        assert result == multiple_atoms_matched

    def test_multiple_coords(self, multiple_coords, multiple_coords_matched):
        result = get_coordinate_matches(multiple_coords)
        assert result == multiple_coords_matched

    def test_no_coordinates(self):
        atom1 = atom_line("H", 0.0, 0.0, 0.0)
        atom2 = atom_line("H", 1.0, -1.0, 0.0, comment="# Second atom")
        input_text = f"BadCoordination\n{atom1}\n{atom2}\n\nOther stuff"
        with pytest.raises(ValueError) as excinfo:
            result = get_coordinate_matches(input_text)
        assert (excinfo.type is ValueError) and (
            'Could not find line starting with "coordinates" (case insensitive)'
            in str(excinfo.value)
        )

    def test_case_sensitivity(self):
        atom1 = atom_line("H", 0.0, 0.0, 0.0)
        atom2 = atom_line("H", 1.0, -1.0, 0.0, comment="# Second atom")
        result1 = get_coordinate_matches(
            f"COORDINATES\n{atom1}\n{atom2}\n\nOther stuff"
        )
        result2 = get_coordinate_matches(
            f"coordinates\n{atom1}\n{atom2}\n\nOther stuff"
        )
        assert result1.lower() == result2.lower()

    def test_leading_space(self):
        # Perhaps the core behavior should change?
        atom1 = atom_line("H", 0.0, 0.0, 0.0)
        atom2 = atom_line("H", 1.0, -1.0, 0.0, comment="# Second atom")
        input_text = f" Coordinates\n{atom1}\n{atom2}\n\nOther stuff"
        with pytest.raises(ValueError) as excinfo:
            result = get_coordinate_matches(input_text)
        assert (excinfo.type is ValueError) and (
            'Could not find line starting with "coordinates" (case insensitive)'
            in str(excinfo.value)
        )


@pytest.fixture
def multiple_coords_section():
    return "\nH 0.0 0.0 0.0\nH 1.0 -1.0 0.0 # Second atom"


@pytest.mark.dependency(
    name="coord_section", depends=["coord_matches", "simple_coord_match"]
)
class Test_get_coordinate_section:
    def test_multiple_atoms(self, multiple_atoms_matched, multiple_coords_section):
        result = get_coordinate_section(multiple_atoms_matched)
        assert result == multiple_coords_section

    def test_multiple_coords(self, multiple_coords_matched, multiple_coords_section):
        result = get_coordinate_section(multiple_coords_matched)
        assert result == multiple_coords_section

    def test_no_double_line_break(self, multiple_coords_section):
        with pytest.raises(ValueError) as exc:
            result = get_coordinate_section(multiple_coords_section)
        assert (exc.type is ValueError) and (
            "Could not find end of coordinates section" in str(exc.value)
        )

    def test_double_line_with_whitespace(self, multiple_coords_section):
        input_text: str = f"{multiple_coords_section}\n \t  \nOther stuff"
        result = get_coordinate_section(input_text)
        assert result == multiple_coords_section


@pytest.fixture
def multiple_atoms_info():
    return (
        2,
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]], dtype=float),
        ["H1", "H2"],
    )


@pytest.mark.dependency(
    name="coord_info", depends=["coord_section", "simple_coord_sect"]
)
class Test_get_coordinate_info:
    def test_multiple_atoms(self, multiple_coords_section, multiple_atoms_info):
        result = get_coordinate_info(multiple_coords_section)
        assert all(
            (
                result[0] == multiple_atoms_info[0],
                result[1] == multiple_atoms_info[1],
                np.allclose(result[2], multiple_atoms_info[2]),
                result[3] == multiple_atoms_info[3],
            )
        )

    def test_bad_coordinate(self):
        coord_section = "\nH a b c"
        with pytest.raises(ValueError) as exc:
            result = get_coordinate_info(coord_section)
        assert (exc.type is ValueError) and (
            "There was an error reading in the atomic coordinates" in str(exc.value)
        )

    def test_int_coordinate(self):
        coord_section = "\nH 0 1 2"
        result = get_coordinate_info(coord_section)
        assert all(
            (
                result[0] == 1,
                result[1] == ["H"],
                np.allclose(result[2], np.array([[0.0, 1.0, 2.0]])),
                result[3] == ["H1"],
            )
        )


@pytest.fixture
def example_A_input_coordinates():
    return """Coordinates  # Example A
H 0.0 0.0 0.0  # Example A atom 1
H 1.0 -1.0 0.0  # Example A atom 2"""


@pytest.fixture
def example_A_parsed_coordinates():
    return (
        2,
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]], dtype=float),
        ["H1", "H2"],
    )


@pytest.fixture
def example_B_input_coordinates():
    return """Coordinates
H 0.0 0.0 0.0
N 1.0 1.0 -1.0
N -1.0 -1.0 1.0
N 0.0 2.0 -2.0"""


@pytest.fixture
def example_B_parsed_coordinates():
    return (
        4,
        ["H", "N", "N", "N"],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [0.0, 2.0, -2.0],
            ],
            dtype=float,
        ),
        ["H1", "N2", "N3", "N4"],
    )


@pytest.fixture
def example_C_input_coordinates():
    return """Coordinates  # Example C
H A B C  # Example C atom 1
H 1.0  # Example C atom 2"""


@pytest.mark.dependency(
    name="parse_coord",
    depends=[
        "coord_info",
        "simple_coord_match",
        "simple_coord_sect",
        "simple_coord_info",
    ],
)
class Test_parse_input_coordinate_section:
    def test_simple_coordinate_section(self):
        n_atoms, atom_symbols, mol_coordinates, atom_numbering = (
            parse_input_coordinate_section("Coordinates\nH 0.0 0.0 0.0\n\n")
        )
        assert all(
            (
                n_atoms == 1,
                atom_symbols == ["H"],
                np.allclose(mol_coordinates, np.array([[0.0, 0.0, 0.0]], dtype=float)),
                atom_numbering == ["H1"],
            )
        )

    def test_multiple_atoms_input(self, multiple_atoms_input):
        result = parse_input_coordinate_section(multiple_atoms_input)
        assert all(
            (
                result[0] == 2,
                result[1] == ["H", "H"],
                np.allclose(result[2], np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])),
                result[3] == ["H1", "H2"],
            )
        )

    def test_multiple_coords_input(self, multiple_coords):
        result = parse_input_coordinate_section(multiple_coords)
        assert all(
            (
                result[0] == 2,
                result[1] == ["H", "H"],
                np.allclose(result[2], np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]])),
                result[3] == ["H1", "H2"],
            )
        )

    def test_example_A(self, example_A_input_coordinates, example_A_parsed_coordinates):
        input_text = f"A comment\n{example_A_input_coordinates}\n\nOther stuff\n"
        result = parse_input_coordinate_section(input_text)
        assert all(
            (
                result[0] == example_A_parsed_coordinates[0],
                result[1] == example_A_parsed_coordinates[1],
                np.allclose(result[2], example_A_parsed_coordinates[2]),
                result[3] == example_A_parsed_coordinates[3],
            )
        )

    def test_example_B(self, example_B_input_coordinates, example_B_parsed_coordinates):
        input_text = f"B comment\n{example_B_input_coordinates}\n\nOther stuff\n"
        result = parse_input_coordinate_section(input_text)
        assert all(
            (
                result[0] == example_B_parsed_coordinates[0],
                result[1] == example_B_parsed_coordinates[1],
                np.allclose(result[2], example_B_parsed_coordinates[2]),
                result[3] == example_B_parsed_coordinates[3],
            )
        )

    def test_example_C(self, example_C_input_coordinates):
        input_text = f"C comment\n{example_C_input_coordinates}\n\nOther stuff\n"
        with pytest.raises(ValueError) as exc:
            result = parse_input_coordinate_section(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the atomic coordinates." in str(exc.value)
        )


# Tests for dipole-section functions
# ----------------------------------
#
# Proper format of the dipole section is:
#     Dipole      # comments
#     muX muY muZ # more comments
#     (blank line)
# where muX, muY, and muZ are numeric values.
#


@pytest.fixture
def simple_dipole_section():
    return """Simple dipole
Dipole
0.1 0.2 0.3

"""


class Test_simple_dipole_section:
    @pytest.mark.dependency(name="simple_dipole_match")
    def test_simple_get_dipole_matches(self, simple_dipole_section):
        result = get_dipole_matches(simple_dipole_section)
        assert result == "\n0.1 0.2 0.3\n\n"

    @pytest.mark.dependency(name="simple_dipole_sect")
    def test_simple_get_dipole_section(self):
        result = get_dipole_section("\n0.1 0.2 0.3\n\n")
        assert result == "\n0.1 0.2 0.3"

    @pytest.mark.dependency(name="simple_dipole_info")
    def test_simple_get_dipole_info(self):
        result = get_dipole_info("\n0.1 0.2 0.3")
        assert np.allclose(result, np.array([0.1, 0.2, 0.3]))


@pytest.fixture
def multiple_dipoles():
    input_text = """Multiple dipole
Dipole
0.1 0.2 0.3

Other stuff
Dipole
0.4 0.5 0.6
"""
    return input_text


@pytest.fixture
def multiple_dipoles_matched():
    return "\n0.1 0.2 0.3\n\nOther stuff\n"


@pytest.mark.dependency(name="dipole_matches", depends=["simple_dipole_match"])
class Test_get_dipole_matches:
    def test_mutiple_dipoles(self, multiple_dipoles, multiple_dipoles_matched):
        result = get_dipole_matches(multiple_dipoles)
        assert result == multiple_dipoles_matched

    def test_no_dipole(self):
        input_text = f"BadDipole\n0.1 0.2 0.3\n\nOther stuff"
        with pytest.raises(ValueError) as excinfo:
            result = get_dipole_matches(input_text)
        assert (excinfo.type is ValueError) and (
            'Could not find line starting with "dipole" (case insensitive)'
            in str(excinfo.value)
        )

    def test_case_sensitivity(self):
        dipole1 = "Comment\nDIPOLE\n0.1 0.2 0.3\n\nOther stuff"
        dipole2 = "Comment\ndipole\n0.1 0.2 0.3\n\nOther stuff"
        result1 = get_dipole_matches(dipole1)
        result2 = get_dipole_matches(dipole2)
        assert result1.lower() == result2.lower()

    def test_leading_space(self):
        dipole = " Dipole\n0.1 0.2 0.3\n\nOther stuff"
        with pytest.raises(ValueError) as excinfo:
            result = get_dipole_matches(dipole)
        assert (excinfo.type is ValueError) and (
            'Could not find line starting with "dipole" (case insensitive)'
            in str(excinfo.value)
        )


@pytest.fixture
def dipole_section():
    return "\n0.1 0.2 0.3"


@pytest.mark.dependency(
    name="dipole_section", depends=["dipole_matches", "simple_dipole_match"]
)
class Test_get_dipole_section:
    def test_dipole_match(self, multiple_dipoles_matched, dipole_section):
        result = get_dipole_section(multiple_dipoles_matched)
        assert result == dipole_section

    def test_no_double_line_break(self, dipole_section):
        with pytest.raises(ValueError) as exc:
            result = get_dipole_section(dipole_section)
        assert (exc.type is ValueError) and (
            "Could not find end of dipole section" in str(exc.value)
        )

    def test_double_line_with_whitespace(self, dipole_section):
        input_text: str = f"{dipole_section}\n \t  \nOther stuff"
        result = get_dipole_section(input_text)
        assert result == dipole_section


@pytest.fixture
def dipole_info():
    return np.array([0.1, 0.2, 0.3])


@pytest.mark.dependency(
    name="dipole_info", depends=["dipole_section", "simple_dipole_sect"]
)
class Test_get_dipole_info:
    def test_dipole_section(self, dipole_section, dipole_info):
        result = get_dipole_info(dipole_section)
        assert np.allclose(result, dipole_info)

    def test_bad_dipole(self):
        dipole_section = "\n x y z"
        with pytest.raises(ValueError) as exc:
            result = get_dipole_info(dipole_section)
        assert (exc.type is ValueError) and (
            "There was an error reading in the dipole." in str(exc.value)
        )

    def test_int_dipole(self):
        dipole_section = "\n0 1 2"
        result = get_dipole_info(dipole_section)
        assert np.allclose(result, np.array([0.0, 1.0, 2.0]))

    def test_leading_space(self, dipole_info):
        dipole_section = "\n 0.1 0.2 0.3"
        result = get_dipole_info(dipole_section)
        assert np.allclose(result, dipole_info)

    def test_negative_dipole(self):
        dipole_section = "\n-0.1 -0.2 -0.3"
        result = get_dipole_info(dipole_section)
        assert np.allclose(result, np.array([-0.1, -0.2, -0.3]))

    def test_zero_dipole(self):
        dipole_section = "\n0 0 0"
        result = get_dipole_info(dipole_section)
        assert np.allclose(result, np.array([0.0, 0.0, 0.0]))


@pytest.fixture
def example_A_input_dipole():
    return """Dipole  # Example A
1.0 -1.0 0.0  # Example A dipole"""


@pytest.fixture
def example_A_parsed_dipole():
    return np.array([1.0, -1.0, 0.0])


@pytest.fixture
def example_B_input_dipole():
    return """Dipole
0 2.0 -2.0"""


@pytest.fixture
def example_B_parsed_dipole():
    return np.array([0.0, 2.0, -2.0])


@pytest.fixture
def example_C_input_dipole():
    return """Dipole  # Example C
H 1.0  # Example C dipole"""


@pytest.mark.dependency(
    name="parse_dipole",
    depends=[
        "dipole_info",
        "simple_dipole_match",
        "simple_dipole_sect",
        "simple_dipole_info",
    ],
)
class Test_parse_input_dipole_section:
    def test_simple_dipole_section(self, simple_dipole_section):
        result = parse_input_dipole_section(simple_dipole_section)
        assert np.allclose(result, np.array([0.1, 0.2, 0.3]))

    def test_multiple_dipoles_input(self, multiple_dipoles):
        result = parse_input_dipole_section(multiple_dipoles)
        assert np.allclose(result, np.array([0.1, 0.2, 0.3]))

    def test_example_A(self, example_A_input_dipole, example_A_parsed_dipole):
        input_text = f"A comment\n{example_A_input_dipole}\n\nOther stuff"
        result = parse_input_dipole_section(input_text)
        assert np.allclose(result, example_A_parsed_dipole)

    def test_example_B(self, example_B_input_dipole, example_B_parsed_dipole):
        input_text = f"B comment\n{example_B_input_dipole}\n\nOther stuff"
        result = parse_input_dipole_section(input_text)
        assert np.allclose(result, example_B_parsed_dipole)

    def test_example_C(self, example_C_input_dipole):
        input_text = f"C comment\n{example_C_input_dipole}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_dipole_section(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the dipole." in str(exc.value)
        )


# Tests for isotopologue-section functions
# ----------------------------------------
#
# Proper format of the isotopologue section is:
#     Isotopologues   # comment
#     mass1 mass2 mass3 ... massZ iso000  # more comments
#     mass1 mass2 mass3 ... massZ iso001
#     ...
#     mass1 mass2 mass3 ... massZ isoZZZ
#     (blank line)
# where mass# is the atomic mass number of the isotope,
# and iso### is the isotopologue label to use in the output.


@pytest.fixture
def simple_iso_input():
    return """Simple isotopologue
Isotopologues  # comment
1 2 3 iso000  # comment

Other stuff
"""


@pytest.fixture
def simple_iso_match():
    return "  # comment\n1 2 3 iso000  # comment\n\nOther stuff\n"


@pytest.fixture
def simple_iso_section():
    return "  # comment\n1 2 3 iso000  # comment"


@pytest.fixture
def simple_iso_info():
    return (
        ["iso000"],
        {"iso000": [1, 2, 3]},
    )


class Test_simple_isotopologue_section:
    @pytest.mark.dependency(name="simple_iso_match")
    def test_simple_get_coordinate_matches(self, simple_iso_input, simple_iso_match):
        result = get_isotopologue_matches(simple_iso_input)
        assert result == simple_iso_match

    @pytest.mark.dependency(name="simple_iso_sect")
    def test_simple_get_isotopologue_section(
        self, simple_iso_match, simple_iso_section
    ):
        result = get_isotopologue_section(simple_iso_match)
        assert result == simple_iso_section

    @pytest.mark.dependency(name="simple_iso_info")
    def test_simple_get_isotopologue_info(self, simple_iso_section, simple_iso_info):
        result = get_isotopologue_info(simple_iso_section, 3)
        assert result == simple_iso_info


@pytest.fixture
def multiple_isos_input():
    return """
Isotopologues  # comment
1 2 3 iso000  # comment2
4 5 6 iso001  # comment3

Other stuff
"""


@pytest.fixture
def multiple_isos_matched():
    return "  # comment\n1 2 3 iso000  # comment2\n4 5 6 iso001  # comment3\n\nOther stuff\n"


@pytest.fixture
def multiple_isos_section():
    return "  # comment\n1 2 3 iso000  # comment2\n4 5 6 iso001  # comment3"


@pytest.fixture
def multiple_isos_info():
    return (
        ["iso000", "iso001"],
        {
            "iso000": [1, 2, 3],
            "iso001": [4, 5, 6],
        },
    )


@pytest.fixture
def duplicate_isos_input(multiple_isos_input):
    return f"{multiple_isos_input}\n\n{multiple_isos_input}"


@pytest.fixture
def duplicate_isos_matched():
    return "  # comment\n1 2 3 iso000  # comment2\n4 5 6 iso001  # comment3\n\nOther stuff\n\n\n\n"


@pytest.mark.dependency(name="iso_matches", depends=["simple_iso_match"])
class Test_get_isotopologue_matches:
    def test_multiple_isotopologues(self, multiple_isos_input, multiple_isos_matched):
        result = get_isotopologue_matches(multiple_isos_input)
        assert result == multiple_isos_matched

    def test_duplicate_input(self, duplicate_isos_input, duplicate_isos_matched):
        result = get_isotopologue_matches(duplicate_isos_input)
        assert result == duplicate_isos_matched

    def test_no_header(self):
        input_text = """
1 2 3 iso000  # comment2
4 5 6 iso001  # comment3

Other stuff
"""
        with pytest.raises(ValueError) as excinfo:
            result = get_isotopologue_matches(input_text)
        assert (excinfo.type is ValueError) and (
            'Could not find line starting with "isotopologues" (case insensitive)'
            in str(excinfo.value)
        )

    def test_case_sensitivity(self):
        input1 = "ISOTOPOLOGUES  # comment\n1 2 3 iso000  # comment2\n4 5 6 iso001  # comment3\n\nOther stuff\n"
        input2 = "isotopologues  # comment\n1 2 3 iso000  # comment2\n4 5 6 iso001  # comment3\n\nOther stuff\n"
        result1 = get_isotopologue_matches(input1)
        result2 = get_isotopologue_matches(input2)
        assert result1.lower() == result2.lower()

    def test_leading_space(self):
        input_text = " Isotopologue  # comment\n1 2 3 iso000  # comment2\n4 5 6 iso001  # comment3\n\nOther stuff\n"
        with pytest.raises(ValueError) as excinfo:
            result = get_isotopologue_matches(input_text)
        assert (excinfo.type is ValueError) and (
            'Could not find line starting with "isotopologues" (case insensitive)'
            in str(excinfo.value)
        )


@pytest.mark.dependency(name="iso_section", depends=["iso_matches", "simple_iso_match"])
class Test_get_isotopologue_section:
    def test_multiple_isotopologues(self, multiple_isos_matched, multiple_isos_section):
        result = get_isotopologue_section(multiple_isos_matched)
        assert result == multiple_isos_section

    def test_duplicate_input(self, duplicate_isos_matched, multiple_isos_section):
        result = get_isotopologue_section(duplicate_isos_matched)
        assert result == multiple_isos_section

    def test_no_double_line_break(self, multiple_isos_section):
        with pytest.raises(ValueError) as excinfo:
            result = get_isotopologue_section(multiple_isos_section)
        assert (excinfo.type is ValueError) and (
            "Could not find end of isotopologues section" in str(excinfo.value)
        )

    def test_double_line_with_whitespace(self, multiple_isos_section):
        input_text = f"{multiple_isos_section}\n \t  \nOther stuff"
        result = get_isotopologue_section(input_text)
        assert result == multiple_isos_section


@pytest.mark.dependency(name="iso_info", depends=["iso_section", "simple_iso_sect"])
class Test_get_isotopologue_info:
    def test_multiple_isotopologues(self, multiple_isos_section, multiple_isos_info):
        result = get_isotopologue_info(multiple_isos_section, 3)
        assert result == multiple_isos_info

    def test_bad_isotopologue(self):
        input_text = "\nA B C 1234"
        with pytest.raises(ValueError) as exc:
            result = get_isotopologue_info(input_text, 3)
        assert (exc.type is ValueError) and (
            "There was an error reading in the isotopologue masses" in str(exc.value)
        )

    def test_float_masses(self):
        input_text = "\n1.2 3.4 5.6 badiso"
        with pytest.raises(ValueError) as exc:
            result = get_isotopologue_info(input_text, 3)
        assert (exc.type is ValueError) and (
            "There was an error reading in the isotopologue masses" in str(exc.value)
        )

    def test_wrong_number_of_atoms(self, multiple_isos_section):
        with pytest.raises(ValueError) as exc:
            result = get_isotopologue_info(multiple_isos_section, 4)
        assert (exc.type is ValueError) and (
            "There was an error reading in the isotopologue masses" in str(exc.value)
        )

    def test_leading_space(self):
        input_text = "\n  1 2 3 iso000 #comment"
        result = get_isotopologue_info(input_text, 3)
        assert result == (["iso000"], {"iso000": [1, 2, 3]})

    def test_duplicate_isolabel(self):
        input_text = "\n1 2 3 iso000 #first one\n2 3 4 iso000 #second one"
        with pytest.raises(ValueError) as exc:
            result = get_isotopologue_info(input_text, 3)
        assert (exc.type is ValueError) and (
            "Isotopologue section contains duplicate labels: ['iso000']"
            in str(exc.value)
        )


@pytest.fixture
def example_A_input_isotopologues():
    return """Isotopologues  # Example A
1 1 iso000  # normal isotopologue
1 2 iso001  # one heavy substitution
2 2 iso002  # two heavy substitutions"""


@pytest.fixture
def example_A_parsed_isotopologues():
    return (
        ["iso000", "iso001", "iso002"],
        {
            "iso000": [1, 1],
            "iso001": [1, 2],
            "iso002": [2, 2],
        },
    )


@pytest.fixture
def example_B_input_isotopologues():
    return """Isotopologues
1 14 14 14 iso000
2 14 14 14 iso001"""


@pytest.fixture
def example_B_parsed_isotopologues():
    return (
        ["iso000", "iso001"],
        {
            "iso000": [1, 14, 14, 14],
            "iso001": [2, 14, 14, 14],
        },
    )


@pytest.fixture
def example_C_input_isotopologues():
    return """Isotopologues  # Example C
H A B C
0 1.0"""


@pytest.mark.dependency(
    name="parse_iso",
    depends=[
        "iso_info",
        "simple_iso_match",
        "simple_iso_sect",
        "simple_iso_info",
    ],
)
class Test_parse_input_isotopologue_section:
    def test_simple_isotopologue_section(self, simple_iso_input, simple_iso_info):
        result = parse_input_isotopologue_section(simple_iso_input, 3)
        assert result == simple_iso_info

    def test_multiple_isos_input(self, multiple_isos_input, multiple_isos_info):
        result = parse_input_isotopologue_section(multiple_isos_input, 3)
        assert result == multiple_isos_info

    def test_duplicate_isos_input(self, duplicate_isos_input, multiple_isos_info):
        result = parse_input_isotopologue_section(duplicate_isos_input, 3)
        assert result == multiple_isos_info

    def test_example_A(
        self, example_A_input_isotopologues, example_A_parsed_isotopologues
    ):
        input_text = f"A comment\n{example_A_input_isotopologues}\n\nOther stuff"
        result = parse_input_isotopologue_section(input_text, 2)
        assert result == example_A_parsed_isotopologues

    def test_example_B(
        self, example_B_input_isotopologues, example_B_parsed_isotopologues
    ):
        input_text = f"B comment\n{example_B_input_isotopologues}\n\nOther stuff"
        result = parse_input_isotopologue_section(input_text, 4)
        assert result == example_B_parsed_isotopologues

    def test_example_C(self, example_C_input_isotopologues):
        input_text = f"C comment\n{example_C_input_isotopologues}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_isotopologue_section(input_text, 3)
        assert (exc.type is ValueError) and (
            "There was an error reading in the isotopologue masses." in str(exc.value)
        )


# ---------------------------------
# Tests for the whole input_parser!
# ---------------------------------


# Example A: valid, pseudo-H2
# Example B: valid, pseudo-HN3
# Example C: invalid, pseudo-H2


@pytest.fixture
def example_A_inputs(
    example_A_input_coordinates, example_A_input_dipole, example_A_input_isotopologues
):
    return (
        example_A_input_coordinates,
        example_A_input_dipole,
        example_A_input_isotopologues,
    )


@pytest.fixture
def example_A_parsed(
    example_A_parsed_coordinates,
    example_A_parsed_dipole,
    example_A_parsed_isotopologues,
):
    return (
        example_A_parsed_coordinates,
        example_A_parsed_dipole,
        example_A_parsed_isotopologues,
    )


@pytest.fixture
def example_B_inputs(
    example_B_input_coordinates, example_B_input_dipole, example_B_input_isotopologues
):
    return (
        example_B_input_coordinates,
        example_B_input_dipole,
        example_B_input_isotopologues,
    )


@pytest.fixture
def example_B_parsed(
    example_B_parsed_coordinates,
    example_B_parsed_dipole,
    example_B_parsed_isotopologues,
):
    return (
        example_B_parsed_coordinates,
        example_B_parsed_dipole,
        example_B_parsed_isotopologues,
    )


class Test_parse_input_file:
    def test_example_A(
        self,
        example_A_inputs,
        example_A_parsed,
    ):
        coord, dip, iso = example_A_inputs
        input_text = f"Example A input file\n\n{coord}\n\n{dip}\n\n{iso}\n\nOther stuff"

        p_coord, p_dip, p_iso = example_A_parsed

        n_atoms, atom_symbols, mol_coord, atom_numbering = p_coord
        mol_dipole = p_dip
        iso_names, iso_dict = p_iso

        result = parse_input_file(input_text)

        assert all(
            (
                result[0] == iso_names,
                result[1] == iso_dict,
                result[2] == n_atoms,
                result[3] == atom_symbols,
                np.allclose(result[4], mol_coord),
                np.allclose(result[5], mol_dipole),
                result[6] == atom_numbering,
            )
        )

    def test_example_B(
        self,
        example_B_inputs,
        example_B_parsed,
    ):
        coord, dip, iso = example_B_inputs
        input_text = f"Example B input file\n\n{coord}\n\n{dip}\n\n{iso}\n\nOther stuff"

        p_coord, p_dip, p_iso = example_B_parsed

        n_atoms, atom_symbols, mol_coord, atom_numbering = p_coord
        mol_dipole = p_dip
        iso_names, iso_dict = p_iso

        result = parse_input_file(input_text)

        assert all(
            (
                result[0] == iso_names,
                result[1] == iso_dict,
                result[2] == n_atoms,
                result[3] == atom_symbols,
                np.allclose(result[4], mol_coord),
                np.allclose(result[5], mol_dipole),
                result[6] == atom_numbering,
            )
        )

    def test_rearranged_A(self, example_A_inputs, example_A_parsed):
        coord, dip, iso = example_A_inputs
        input_text = f"Example A input file\n\n{iso}\n\n{dip}\n\n{coord}\n\nOther stuff"

        p_coord, p_dip, p_iso = example_A_parsed

        n_atoms, atom_symbols, mol_coord, atom_numbering = p_coord
        mol_dipole = p_dip
        iso_names, iso_dict = p_iso

        result = parse_input_file(input_text)

        assert all(
            (
                result[0] == iso_names,
                result[1] == iso_dict,
                result[2] == n_atoms,
                result[3] == atom_symbols,
                np.allclose(result[4], mol_coord),
                np.allclose(result[5], mol_dipole),
                result[6] == atom_numbering,
            )
        )

    def test_rearranged_B(
        self,
        example_B_inputs,
        example_B_parsed,
    ):
        coord, dip, iso = example_B_inputs
        input_text = f"Example B input file\n\n{iso}\n\n{coord}\n\n{dip}\n\nOther stuff"

        p_coord, p_dip, p_iso = example_B_parsed

        n_atoms, atom_symbols, mol_coord, atom_numbering = p_coord
        mol_dipole = p_dip
        iso_names, iso_dict = p_iso

        result = parse_input_file(input_text)

        assert all(
            (
                result[0] == iso_names,
                result[1] == iso_dict,
                result[2] == n_atoms,
                result[3] == atom_symbols,
                np.allclose(result[4], mol_coord),
                np.allclose(result[5], mol_dipole),
                result[6] == atom_numbering,
            )
        )

    def test_coord_A_rest_B(self, example_A_inputs, example_B_inputs):
        coordA, dipA, isoA = example_A_inputs
        coordB, dipB, isoB = example_B_inputs
        input_text = (
            f"Mismatched input file\n\n{coordA}\n\n{dipB}\n\n{isoB}\n\nOther stuff"
        )
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert exc.type is ValueError

    def test_coord_B_rest_A(self, example_A_inputs, example_B_inputs):
        coordA, dipA, isoA = example_A_inputs
        coordB, dipB, isoB = example_B_inputs
        input_text = (
            f"Mismatched input file\n\n{coordB}\n\n{dipA}\n\n{isoA}\n\nOther stuff"
        )
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert exc.type is ValueError

    def test_AAC(self, example_A_inputs, example_C_input_isotopologues):
        coordA, dipA, isoA = example_A_inputs
        input_text = f"Bad input file\n\n{coordA}\n\n{dipA}\n\n{example_C_input_isotopologues}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the isotopologue masses" in str(exc.value)
        )

    def test_ACA(self, example_A_inputs, example_C_input_dipole):
        coordA, dipA, isoA = example_A_inputs
        input_text = f"Bad input file\n\n{coordA}\n\n{example_C_input_dipole}\n\n{isoA}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the dipole" in str(exc.value)
        )

    def test_CAA(self, example_A_inputs, example_C_input_coordinates):
        coordA, dipA, isoA = example_A_inputs
        input_text = f"Bad input file\n\n{example_C_input_coordinates}\n\n{dipA}\n\n{isoA}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the atomic coordinates" in str(exc.value)
        )

    def test_BBC(self, example_B_inputs, example_C_input_isotopologues):
        coordB, dipB, isoB = example_B_inputs
        input_text = f"Bad input file\n\n{coordB}\n\n{dipB}\n\n{example_C_input_isotopologues}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the isotopologue masses" in str(exc.value)
        )

    def test_BCB(self, example_B_inputs, example_C_input_dipole):
        coordB, dipB, isoB = example_B_inputs
        input_text = f"Bad input file\n\n{coordB}\n\n{example_C_input_dipole}\n\n{isoB}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the dipole" in str(exc.value)
        )

    def test_CBB(self, example_B_inputs, example_C_input_coordinates):
        coordB, dipB, isoB = example_B_inputs
        input_text = f"Bad input file\n\n{example_C_input_coordinates}\n\n{dipB}\n\n{isoB}\n\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the atomic coordinates" in str(exc.value)
        )

    def test_white_space_double_lines_A(self, example_A_inputs, example_A_parsed):
        coord, dip, iso = example_A_inputs
        input_text = f"Example A input file\n  \t \n{coord}\n  \t \n{dip}\n  \t \n{iso}\n  \t \nOther stuff"

        p_coord, p_dip, p_iso = example_A_parsed

        n_atoms, atom_symbols, mol_coord, atom_numbering = p_coord
        mol_dipole = p_dip
        iso_names, iso_dict = p_iso

        result = parse_input_file(input_text)

        assert all(
            (
                result[0] == iso_names,
                result[1] == iso_dict,
                result[2] == n_atoms,
                result[3] == atom_symbols,
                np.allclose(result[4], mol_coord),
                np.allclose(result[5], mol_dipole),
                result[6] == atom_numbering,
            )
        )

    def test_white_space_double_lines_B(self, example_B_inputs, example_B_parsed):
        coord, dip, iso = example_B_inputs
        input_text = f"Example B input file\n  \t \n{coord}\n  \t \n{dip}\n  \t \n{iso}\n  \t \nOther stuff"

        p_coord, p_dip, p_iso = example_B_parsed

        n_atoms, atom_symbols, mol_coord, atom_numbering = p_coord
        mol_dipole = p_dip
        iso_names, iso_dict = p_iso

        result = parse_input_file(input_text)

        assert all(
            (
                result[0] == iso_names,
                result[1] == iso_dict,
                result[2] == n_atoms,
                result[3] == atom_symbols,
                np.allclose(result[4], mol_coord),
                np.allclose(result[5], mol_dipole),
                result[6] == atom_numbering,
            )
        )

    def test_no_double_lines_A(self, example_A_inputs, example_A_parsed):
        coord, dip, iso = example_A_inputs
        input_text = f"Example A input file\n{coord}\n{dip}\n{iso}\nOther stuff"
        with pytest.raises(ValueError) as exc:
            result = parse_input_file(input_text)
        assert (exc.type is ValueError) and (
            "There was an error reading in the atomic coordinates" in str(exc.value)
        )


# class Test_get_principal_axes:
# class Test_get_dataframes:
# class Test_header_creator:
# class Test_df_text_export:
# class Test_generate_output_file:
# class Test_generate_csv_output:
