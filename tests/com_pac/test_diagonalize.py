"""
Unit tests for functions in diagonalize.py
"""

from decimal import DivisionByZero

from mendeleev.fetch import fetch_table
from mendeleev.mendeleev import element

import pytest
import numpy as np
from com_pac.diagonalize import (
    inertia_matrix,
    inertia_to_rot_const,
    get_mol_masses,
    get_isotopes_dict,
    get_unique_isotopes,
    get_isotopes_mass,
    get_unique_isotopes_mass_dict,
    get_COM_coordinates,
)


@pytest.fixture
def random_coords1():
    return np.array(
        [
            [-3.99787168, -3.64611009, -0.79857223],
            [3.82041845, 2.59015573, -1.72846941],
            [-4.14737334, 2.22363115, 0.53194212],
            [2.27960531, -1.5805131, 4.57701207],
            [2.26206522, 1.89713985, -2.23942121],
            [-1.39925724, -3.85145304, -2.41527489],
        ]
    )


@pytest.fixture
def random_masses1():
    return np.array(
        [
            81.56014294,
            70.45185325,
            38.77692425,
            23.499187,
            59.64661262,
            7.02675499,
        ]
    )


@pytest.fixture
def random_inertias1():
    return np.array(
        [
            [3232.1414003, -1737.59407925, 323.60723383],
            [-1737.59407925, 4545.80479967, 390.10606476],
            [323.60723383, 390.10606476, 5566.20277886],
        ]
    )


@pytest.fixture
def random_COM_coords1():
    return np.array(
        [
            [-3.85880405, -3.71830349, -0.05374474],
            [3.95948608, 2.51796233, -0.98364192],
            [-4.00830571, 2.15143775, 1.27676961],
            [2.41867294, -1.6527065, 5.32183956],
            [2.40113285, 1.82494645, -1.49459372],
            [-1.26018961, -3.92364644, -1.6704474],
        ]
    )


@pytest.fixture
def random_COM_value1():
    return np.array([-0.13906763, 0.0721934, -0.74482749])


@pytest.fixture
def random_coords2():
    return np.array(
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
    )


@pytest.fixture
def random_masses2():
    return np.array(
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
    )


@pytest.fixture
def random_inertias2():
    return np.array(
        [
            [10897.5136067, -1297.86596882, 1916.87183564],
            [-1297.86596882, 10638.94516674, -368.24168553],
            [1916.87183564, -368.24168553, 10734.69310443],
        ]
    )


@pytest.fixture
def random_COM_coords2():
    return np.array(
        [
            [3.81234572, -1.97724529, -4.1230642],
            [-3.40680097, 1.01605211, 4.2251354],
            [-4.53194735, 2.99624185, 2.87356313],
            [-3.90885835, -2.28376728, 4.84589775],
            [2.81513986, 3.05801011, 1.87153986],
            [1.67073802, 3.84444271, -2.04092385],
            [-4.63850231, -2.2487314, -2.39849702],
            [-4.32093116, 1.00808762, 4.19037258],
            [1.58947898, -2.9197044, -1.90237675],
            [0.07233722, -4.55355256, 0.5551234],
            [-0.60462665, 4.0439371, -2.93136229],
            [2.04110253, 2.30296839, 4.23272823],
            [-3.69137486, -3.47538163, -1.80538432],
        ]
    )


@pytest.fixture
def random_COM_value2():
    return np.array([1.16325926, 0.45013021, -0.5208609])


@pytest.fixture
def random_coords3():
    return np.array(
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
    )


@pytest.fixture
def random_masses3():
    return np.array(
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
    )


@pytest.fixture
def random_inertias3():
    return np.array(
        [
            [6849.15179067, -1457.50492045, 445.12063352],
            [-1457.50492045, 6391.06906719, 120.70623906],
            [445.12063352, 120.70623906, 8414.81403036],
        ]
    )


@pytest.fixture
def random_COM_coords3():
    return np.array(
        [
            [-3.65370632, 4.37322082, 1.07306988],
            [3.57697196, -1.04949596, -0.87194415],
            [-2.81774597, -2.71390422, 1.81013754],
            [4.40061667, 4.54630838, -1.51153607],
            [0.55413419, 1.92450827, 2.62326195],
            [-1.87899084, 1.89912808, -4.84019614],
            [1.06297308, 0.35945119, -2.17271265],
            [-3.89649694, -4.51327345, -0.92555932],
            [-1.02144079, -3.2775791, -2.91649722],
            [-3.24423678, -3.0669372, 0.66870497],
            [-2.75240586, 4.69917426, 1.3592159],
            [4.51748188, 4.63884508, 3.15439627],
            [0.38248456, 2.78629668, -1.64373998],
            [1.67604975, -2.92762624, 1.80220927],
        ]
    )


@pytest.fixture
def random_COM_value3():
    return np.array([-0.83923283, -0.33436235, 0.57716083])


@pytest.fixture
def random_coords4():
    return np.array(
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
    )


@pytest.fixture
def random_masses4():
    return np.array(
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
    )


@pytest.fixture
def random_inertias4():
    return np.array(
        [
            [11067.32871999, -1284.86304906, 1084.18125054],
            [-1284.86304906, 10206.63369192, -280.38973705],
            [1084.18125054, -280.38973705, 8237.79128261],
        ]
    )


@pytest.fixture
def random_COM_coords4():
    return np.array(
        [
            [4.44827985, 0.06079028, -3.72104722],
            [-0.49071075, 1.44798579, -2.12376391],
            [-1.15454876, -3.04409368, 3.41627206],
            [0.13599228, 2.0399624, 4.46002244],
            [-0.3921458, -0.20960014, -0.69278895],
            [2.84229721, -1.02622974, 0.53215694],
            [-1.72111592, 4.10380811, -1.53028507],
            [-2.67428492, -4.04054172, -0.34616197],
            [-1.02830872, 2.34536458, 1.92140967],
            [-2.75422904, -2.08456204, -4.47408162],
            [-0.12428394, 2.40565827, 0.57378866],
            [1.58125465, -2.97449223, 3.47153974],
            [-0.12187141, 2.01398157, 2.69180459],
            [3.91307721, 0.92213266, -4.08152395],
            [0.72703794, 5.2919612, 4.46762621],
        ]
    )


@pytest.fixture
def random_COM_value4():
    return np.array([-0.48437239, -0.73088701, -0.06534187])


@pytest.fixture
def random_coords5():
    return np.array(
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
    )


@pytest.fixture
def random_masses5():
    return np.array(
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
    )


@pytest.fixture
def random_inertias5():
    return np.array(
        [
            [19399.69694079, -1166.30038708, 479.30043135],
            [-1166.30038708, 16271.3761811, 2311.98305113],
            [479.30043135, 2311.98305113, 16671.13308675],
        ]
    )


@pytest.fixture
def random_COM_coords5():
    return np.array(
        [
            [3.73577885, -1.22290984, 2.00250111],
            [0.6640693, -4.28951568, -3.20498929],
            [-0.90811655, 0.36916198, 2.32032362],
            [2.36560474, -0.63203101, 0.35064724],
            [-2.14089803, -1.75900982, 1.88866327],
            [-3.57556938, -2.02435851, 4.06960785],
            [3.7353207, -0.92722827, 3.68325531],
            [-3.28352268, 4.00970469, -4.81316248],
            [-3.3867162, -1.75386563, -3.28381599],
            [2.62846493, 1.76370906, 1.54850778],
            [0.01828585, 1.20242702, 2.07634523],
            [3.98871181, -2.55006608, -1.11367282],
            [-0.53570521, 4.83919471, -1.91773731],
            [-0.16209736, 4.8974671, 0.26381974],
            [2.44500285, 4.81852472, -4.36423263],
            [5.02080402, -4.62750562, 0.4486147],
            [1.62409944, -3.07065951, -4.05348229],
            [-0.24639017, 2.25774905, -4.76416902],
            [-0.7834928, -4.27002096, 3.25230611],
            [4.79510489, 4.25979581, -1.74670892],
        ]
    )


@pytest.fixture
def random_COM_value5():
    return np.array([-0.0527959, -0.14978125, 0.46179269])


@pytest.fixture
def random_coords6():
    return np.array(
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
    )


@pytest.fixture
def random_masses6():
    return np.array(
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
    )


@pytest.fixture
def random_inertias6():
    return np.array(
        [
            [23473.29957642, 1001.59416876, 7456.76700785],
            [1001.59416876, 25026.47335396, 383.56115821],
            [7456.76700785, 383.56115821, 22148.19342006],
        ]
    )


@pytest.fixture
def random_COM_coords6():
    return np.array(
        [
            [-2.00366086, -2.02981915, 4.3189062],
            [2.45909966, -4.03642258, 1.34882973],
            [0.55178448, 0.12719925, -3.26320659],
            [4.58548489, 0.83103728, -0.84844474],
            [1.8151497, 0.2897158, -4.77861482],
            [-1.91397858, -2.67013529, 4.63948394],
            [2.71899742, 2.31808552, 1.06541943],
            [-2.72824493, 3.91896564, -4.98146864],
            [1.51587997, -0.52451402, -5.11925705],
            [0.74912174, -2.39986766, -0.58019171],
            [-0.32128804, -2.88577571, -0.50409295],
            [-3.93285099, 1.55765991, 1.45538837],
            [2.73219452, -2.73070533, 1.32593666],
            [-2.31580873, 0.99739776, 1.78356921],
            [-4.40897319, 4.58274222, 2.85063301],
            [0.94616961, 2.36961271, 1.81998212],
            [-4.94375836, -3.51473614, 1.41099381],
            [0.16374972, 4.48406226, 0.33220161],
            [-4.91010418, 1.58944251, 1.58802535],
            [2.02590121, 2.90595233, -4.31260704],
            [3.15010662, -2.75369037, -4.31322174],
            [-2.32137098, -3.62142974, 4.3436845],
            [4.86244933, -1.82206163, -3.16455645],
            [3.19206747, -3.41743814, -4.45659814],
            [-2.95460479, 4.79007685, 4.03656242],
        ]
    )


@pytest.fixture
def random_COM_value6():
    return np.array([0.06631191, -0.60284039, 0.15428617])


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
        "f_coordinates,f_masses,f_inertias",
        [
            ("random_coords1", "random_masses1", "random_inertias1"),
            ("random_coords2", "random_masses2", "random_inertias2"),
            ("random_coords3", "random_masses3", "random_inertias3"),
            ("random_coords4", "random_masses4", "random_inertias4"),
            ("random_coords5", "random_masses5", "random_inertias5"),
            ("random_coords6", "random_masses6", "random_inertias6"),
        ],
    )
    def test_inertia_matrix_rand(self, f_coordinates, f_masses, f_inertias, request):
        coordinates = request.getfixturevalue(f_coordinates)
        masses = request.getfixturevalue(f_masses)
        inertias = request.getfixturevalue(f_inertias)
        value = inertia_matrix(coordinates, masses)
        np.testing.assert_allclose(value, inertias)


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


# HN3 fixtures
@pytest.fixture
def hn3_symbols():
    return ["H", "N", "N", "N"]


@pytest.fixture
def hn3_mass_numbers():
    return [1, 14, 14, 14]


@pytest.fixture
def hn3_n_atoms():
    return 4


@pytest.fixture
def hn3_inputs(hn3_symbols, hn3_mass_numbers, hn3_n_atoms):
    return hn3_symbols, hn3_mass_numbers, hn3_n_atoms


# DN3 fixtures
@pytest.fixture
def dn3_symbols():
    return ["H", "N", "N", "N"]


@pytest.fixture
def dn3_mass_numbers():
    return [2, 14, 14, 14]


@pytest.fixture
def dn3_n_atoms():
    return 4


@pytest.fixture
def dn3_inputs(dn3_symbols, dn3_mass_numbers, dn3_n_atoms):
    return dn3_symbols, dn3_mass_numbers, dn3_n_atoms


# Pyridazine fixtures
@pytest.fixture
def pyridazine_symbols():
    return ["N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]


@pytest.fixture
def pyridazine_mass_numbers():
    return [14, 14, 12, 12, 12, 12, 1, 1, 1, 1]


@pytest.fixture
def pyridazine_n_atoms():
    return 10


@pytest.fixture
def pyridazine_inputs(pyridazine_symbols, pyridazine_mass_numbers, pyridazine_n_atoms):
    return pyridazine_symbols, pyridazine_mass_numbers, pyridazine_n_atoms


# Pyridazine 4-D, 4-C13 fixtures
@pytest.fixture
def pyridazine_heavy_symbols():
    return ["N", "N", "C", "C", "C", "C", "H", "H", "H", "H"]


@pytest.fixture
def pyridazine_heavy_mass_numbers():
    return [14, 14, 12, 13, 12, 12, 1, 2, 1, 1]


@pytest.fixture
def pyridazine_heavy_n_atoms():
    return 10


@pytest.fixture
def pyridazine_heavy_inputs(
    pyridazine_heavy_symbols, pyridazine_heavy_mass_numbers, pyridazine_heavy_n_atoms
):
    return (
        pyridazine_heavy_symbols,
        pyridazine_heavy_mass_numbers,
        pyridazine_heavy_n_atoms,
    )


@pytest.fixture
def hn3_isotopes_dict():
    return {
        0: {"symbol": "H", "mass_number": 1},
        1: {"symbol": "N", "mass_number": 14},
        2: {"symbol": "N", "mass_number": 14},
        3: {"symbol": "N", "mass_number": 14},
    }


class Test_get_isotopes_dict:
    def test_hn3_is_correct(self, hn3_inputs, hn3_isotopes_dict):
        symbols, mass_numbers, n_atoms = hn3_inputs
        result = get_isotopes_dict(symbols, mass_numbers, n_atoms)
        assert result == hn3_isotopes_dict

    def test_hn3_switched_arguments(self, hn3_inputs, hn3_isotopes_dict):
        # TODO: Update once there is a proper data structure, this should raise
        #       an exception about improper types.
        symbols, mass_numbers, n_atoms = hn3_inputs
        result = get_isotopes_dict(mass_numbers, symbols, n_atoms)
        assert all(
            (
                result[0]["symbol"] == hn3_isotopes_dict[0]["mass_number"],
                result[1]["symbol"] == hn3_isotopes_dict[1]["mass_number"],
                result[2]["symbol"] == hn3_isotopes_dict[2]["mass_number"],
                result[3]["symbol"] == hn3_isotopes_dict[3]["mass_number"],
                result[0]["mass_number"] == hn3_isotopes_dict[0]["symbol"],
                result[1]["mass_number"] == hn3_isotopes_dict[1]["symbol"],
                result[2]["mass_number"] == hn3_isotopes_dict[2]["symbol"],
                result[3]["mass_number"] == hn3_isotopes_dict[3]["symbol"],
            )
        )

    @pytest.mark.parametrize(
        "inputs",
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pyridazine_heavy_inputs"],
    )
    def test_wrong_n_atoms(self, inputs, request):
        symbols, mass_numbers, n_atoms = request.getfixturevalue(inputs)
        n_atoms = 30
        with pytest.raises(IndexError) as exc:
            result = get_isotopes_dict(symbols, mass_numbers, n_atoms)
        assert exc.type is IndexError

    @pytest.mark.parametrize(
        "inputs",
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pyridazine_heavy_inputs"],
    )
    def test_wrong_mass_number_length(self, inputs, request):
        # TODO: Data validation should be done earlier and this test can be dropped.
        symbols, mass_numbers, n_atoms = request.getfixturevalue(inputs)
        mass_numbers = mass_numbers[:-1]

        with pytest.raises(IndexError) as exc:
            result = get_isotopes_dict(symbols, mass_numbers, n_atoms)
        assert exc.type is IndexError

    @pytest.mark.parametrize(
        "inputs",
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pyridazine_heavy_inputs"],
    )
    def test_correct_length(self, inputs, request):
        symbols, mass_numbers, n_atoms = request.getfixturevalue(inputs)
        result = get_isotopes_dict(symbols, mass_numbers, n_atoms)

        assert len(result) == n_atoms


@pytest.fixture
def hn3_unique_isotopes():
    return set([("H", 1), ("N", 14)])


@pytest.fixture
def dn3_unique_isotopes():
    return set([("H", 2), ("N", 14)])


@pytest.fixture
def pyridazine_unique_isotopes():
    return set([("H", 1), ("C", 12), ("N", 14)])


@pytest.fixture
def pyridazine_heavy_unique_isotopes():
    return set([("H", 1), ("H", 2), ("C", 12), ("C", 13), ("N", 14)])


class Test_get_unique_isotopes:
    def test_hn3_is_correct(self, hn3_isotopes_dict, hn3_unique_isotopes):
        result = get_unique_isotopes(hn3_isotopes_dict)
        assert result == hn3_unique_isotopes

    def test_missing_keys(self):
        test_dict = {0: {"a": 1}, 1: {"a": 2}}
        with pytest.raises(KeyError) as exc:
            result = get_unique_isotopes(test_dict)
        assert exc.type is KeyError

    def test_different_types(self):
        test_dict = {
            0: {"symbol": 123, "mass_number": "A"},
            1: {"symbol": (0, 1), "mass_number": "B"},
        }
        result = get_unique_isotopes(test_dict)
        assert isinstance(result, set)

    def test_unhashable_values(self):
        test_dict = {
            0: {"symbol": [1, 2, 3], "mass_number": [1, 2, 3]},
            1: {"symbol": [1, 2, 3], "mass_number": [1, 2, 3]},
        }
        with pytest.raises(TypeError) as exc:
            result = get_unique_isotopes(test_dict)
        assert exc.type is TypeError


class Test_get_isotopes_mass:
    """
    Only simple testing, since this function is effectively a wrapper for the Mendeleev method.
    """

    @pytest.mark.parametrize(
        "symbol,mass_number,mass",
        [
            ("H", 1, 1.007825031898),
            ("H", 2, 2.014101777844),
            ("C", 12, 12.0),
            ("C", 13, 13.00335483534),
            ("N", 14, 14.00307400425),
            ("N", 15, 15.00010889827),
        ],
    )
    def test_masses(self, symbol, mass_number, mass):
        assert np.allclose(get_isotopes_mass(symbol, mass_number), mass)

    def test_bad_mass_number(self):
        with pytest.raises(ValueError) as exc:
            result = get_isotopes_mass("H", 123123)
        assert exc.type is ValueError and (
            "Isotopic mass not found for H with mass number 123123." in str(exc.value)
        )

    def test_bad_symbol(self):
        with pytest.raises(ValueError) as exc:
            result = get_isotopes_mass("AF", 123)
        assert exc.type is ValueError and (
            "Isotopic mass not found for AF with mass number 123." in str(exc.value)
        )


@pytest.fixture
def hn3_unique_isotopes_dict():
    return {
        ("H", 1): 1.007825031898,
        ("N", 14): 14.00307400425,
    }


@pytest.fixture
def dn3_unique_isotopes_dict():
    return {
        ("H", 2): 2.014101777844,
        ("N", 14): 14.00307400425,
    }


@pytest.fixture
def pyridazine_unique_isotopes_dict():
    return {
        ("H", 1): 1.007825031898,
        ("C", 12): 12.0,
        ("N", 14): 14.00307400425,
    }


@pytest.fixture
def pyridazine_heavy_unique_isotopes_dict():
    return {
        ("H", 1): 1.007825031898,
        ("H", 2): 2.014101777844,
        ("C", 12): 12.0,
        ("C", 13): 13.00335483534,
        ("N", 14): 14.00307400425,
    }


class Test_get_unique_isotopes_mass_dict:
    @pytest.mark.parametrize(
        "unique_isotopes,unique_isotopes_dict",
        [
            ("hn3_unique_isotopes", "hn3_unique_isotopes_dict"),
            ("dn3_unique_isotopes_dict", "dn3_unique_isotopes_dict"),
            ("pyridazine_unique_isotopes", "pyridazine_unique_isotopes_dict"),
            (
                "pyridazine_heavy_unique_isotopes",
                "pyridazine_heavy_unique_isotopes_dict",
            ),
        ],
    )
    def test_expected_results(self, unique_isotopes, unique_isotopes_dict, request):
        # TODO: Investigate why this test adds a full second to testing..
        unique_isos = request.getfixturevalue(unique_isotopes)
        unique_isos_dict = request.getfixturevalue(unique_isotopes_dict)
        result = get_unique_isotopes_mass_dict(unique_isos)
        assert result == unique_isos_dict

    def test_incorrect_type(self):
        # Normally, "unique_isotopes" should be a set, which inherently requires the entries
        # to be hashable. Let's check what happens if that is not the case.
        with pytest.raises(TypeError) as exc:
            result = get_unique_isotopes_mass_dict([[1, 2], [3, 4]])
        assert exc.type is TypeError


# get_mol_masses results
@pytest.fixture
def hn3_get_mol_masses():
    return np.array([1.00782503, 14.003074, 14.003074, 14.003074])


@pytest.fixture
def dn3_get_mol_masses():
    return np.array([2.01410178, 14.003074, 14.003074, 14.003074])


@pytest.fixture
def pyridazine_get_mol_masses():
    return np.array(
        [
            14.003074,
            14.003074,
            12.0,
            12.0,
            12.0,
            12.0,
            1.00782503,
            1.00782503,
            1.00782503,
            1.00782503,
        ]
    )


@pytest.fixture
def pyridazine_heavy_get_mol_masses():
    return np.array(
        [
            14.003074,
            14.003074,
            12.0,
            13.00335484,
            12.0,
            12.0,
            1.00782503,
            2.01410178,
            1.00782503,
            1.00782503,
        ]
    )


class Test_get_mol_masses:
    @pytest.mark.parametrize(
        "inputs",
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pyridazine_heavy_inputs"],
    )
    def test_switched_arguments(self, inputs, request):
        # Can't use a fixture directly in the list of parameterized values
        # Instead, use the name of the fixture combined with the built-in "request"
        # to fixture to look-up the fixture & its value.
        fixture_atom_symbols, fixture_atom_mass_numbers, fixture_n_atoms = (
            request.getfixturevalue(inputs)
        )

        with pytest.raises(ValueError) as exc:
            result = get_mol_masses(
                fixture_atom_mass_numbers, fixture_atom_symbols, fixture_n_atoms
            )
        assert (
            (exc.type is ValueError)
            and ("Isotopic mass not found for" in str(exc.value))
            and ("with mass number" in str(exc.value))
        )

    @pytest.mark.parametrize(
        "inputs,result",
        [
            ("hn3_inputs", "hn3_get_mol_masses"),
            ("dn3_inputs", "dn3_get_mol_masses"),
            ("pyridazine_inputs", "pyridazine_get_mol_masses"),
            ("pyridazine_heavy_inputs", "pyridazine_heavy_get_mol_masses"),
        ],
    )
    def test_expected_output(self, inputs, result, request):
        fixture_atom_symbols, fixture_atom_mass_numbers, fixture_n_atoms = (
            request.getfixturevalue(inputs)
        )
        fixture_result = request.getfixturevalue(result)

        inputs_result = get_mol_masses(
            fixture_atom_symbols, fixture_atom_mass_numbers, fixture_n_atoms
        )

        assert np.allclose(inputs_result, fixture_result)


class Test_get_COM_coordinates:
    @pytest.mark.parametrize(
        "f_masses,f_coordinates,f_COM_coordinates,f_COM_value",
        [
            (
                "random_masses1",
                "random_coords1",
                "random_COM_coords1",
                "random_COM_value1",
            ),
            (
                "random_masses2",
                "random_coords2",
                "random_COM_coords2",
                "random_COM_value2",
            ),
            (
                "random_masses3",
                "random_coords3",
                "random_COM_coords3",
                "random_COM_value3",
            ),
            (
                "random_masses4",
                "random_coords4",
                "random_COM_coords4",
                "random_COM_value4",
            ),
            (
                "random_masses5",
                "random_coords5",
                "random_COM_coords5",
                "random_COM_value5",
            ),
            (
                "random_masses6",
                "random_coords6",
                "random_COM_coords6",
                "random_COM_value6",
            ),
        ],
    )
    def test_expected_output(
        self, f_masses, f_coordinates, f_COM_coordinates, f_COM_value, request
    ):
        masses = request.getfixturevalue(f_masses)
        coordinates = request.getfixturevalue(f_coordinates)
        COM_coordinates = request.getfixturevalue(f_COM_coordinates)
        COM_value = request.getfixturevalue(f_COM_value)

        result_COM_coordinates, result_COM_value = get_COM_coordinates(
            masses, coordinates
        )

        print(f"{result_COM_value=}")
        assert np.allclose(result_COM_coordinates, COM_coordinates) and np.allclose(
            result_COM_value, COM_value
        )

    @pytest.mark.parametrize(
        "f_masses,f_coordinates",
        [
            ("random_masses1", "random_coords3"),
            ("random_masses2", "random_coords4"),
            ("random_masses3", "random_coords5"),
        ],
    )
    def test_mismatched_lengths(self, f_masses, f_coordinates, request):
        masses = request.getfixturevalue(f_masses)
        coordinates = request.getfixturevalue(f_coordinates)

        with pytest.raises(ValueError) as exc:
            result = get_COM_coordinates(masses, coordinates)

        assert (exc.type is ValueError) and (
            'Length of "masses" array must match length of "coordinates" array.'
            in str(exc.value)
        )

    @pytest.mark.parametrize(
        "f_masses,f_coordinates",
        [
            ("random_masses1", "random_coords1"),
            ("random_masses2", "random_coords2"),
            ("random_masses3", "random_coords3"),
            ("random_masses4", "random_coords4"),
            ("random_masses5", "random_coords5"),
            ("random_masses6", "random_coords6"),
        ],
    )
    def test_zero_masses(self, f_masses, f_coordinates, request):
        masses = request.getfixturevalue(f_masses)
        masses = np.zeros(len(masses))
        coordinates = request.getfixturevalue(f_coordinates)

        with pytest.raises(ValueError) as exc:
            result = get_COM_coordinates(masses, coordinates)

        assert (exc.type is ValueError) and (
            'Sum of "masses" array is zero!' in str(exc.value)
        )
