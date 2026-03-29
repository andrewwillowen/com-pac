"""
Unit tests for functions in diagonalize.py
"""

from decimal import DivisionByZero

from mendeleev.fetch import fetch_table
from mendeleev.mendeleev import element

import pytest
import numpy as np
import com_pac.diagonalize as diagonalize
from com_pac.diagonalize import (
    get_inertia_matrix,
    inertia_to_rot_const,
    get_mol_masses,
    get_isotopes_dict,
    get_unique_isotopes,
    get_isotopes_mass,
    get_unique_isotopes_mass_dict,
    get_COM_coordinates,
    get_eigens,
    rotate_coordinates,
    get_principal_axes,
    get_isotopologue_principal_axes,
    check_for_length_mismatch,
    check_for_bad_diagonal,
    transform_dipole,
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
def random_COM_inertias1():
    return np.array(
        [
            [3074.80862736, -1740.41486936, 352.70961294],
            [-1740.41486936, 4384.50262567, 374.99830783],
            [352.70961294, 374.99830783, 5559.30470071],
        ]
    )


@pytest.fixture
def random_evals1():
    return np.array([1802.59995719, 5499.72702001, 5716.28897654])


@pytest.fixture
def random_evecs1():
    return np.array(
        [
            [-0.81278719, -0.5085473, -0.28417712],
            [-0.56719216, 0.57949809, 0.58521365],
            [0.13292872, -0.63683719, 0.75945266],
        ]
    )


@pytest.fixture
def random_pa_coords1():
    return np.array(
        [
            [5.23823485, -0.15813875, -1.12023472],
            [-4.7771423, 0.07198818, -0.39867891],
            [2.20734025, 2.47207272, 3.36776561],
            [-0.32103888, -5.57689522, 2.38716729],
            [-3.18537976, 0.78827623, -0.74943663],
            [3.02767703, -0.56907656, -3.20668013],
        ]
    )


@pytest.fixture
def random_pa_inertias1():
    return np.array(
        [
            [1.80259996e03, -1.72128978e-12, -1.49213975e-12],
            [-1.72128978e-12, 5.49972702e03, 2.67696976e-12],
            [-1.49213975e-12, 2.67696976e-12, 5.71628898e03],
        ]
    )


@pytest.fixture
def random_rot_consts1():
    return [
        np.float64(280.3611542226121),
        np.float64(91.89165257858001),
        np.float64(88.41033171591842),
    ]


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
def random_COM_inertias2():
    return np.array(
        [
            [10636.08147787, -1009.01438769, 1582.63189538],
            [-1009.01438769, 9742.81462155, -497.57785028],
            [1582.63189538, -497.57785028, 9876.44895467],
        ]
    )


@pytest.fixture
def random_evals2():
    return np.array([8561.66542187, 9342.88468794, 12350.79494428])


@pytest.fixture
def random_evecs2():
    return np.array(
        [
            [-0.6640655, 0.12063266, 0.73787857],
            [-0.27427654, 0.87879164, -0.39050945],
            [0.69554971, 0.46170663, 0.5504885],
        ]
    )


@pytest.fixture
def random_pa_coords2():
    return np.array(
        [
            [-4.85713137, -3.18133931, 1.31548175],
            [4.92245143, 2.43269966, -0.58469493],
            [4.18641704, 3.41311457, -2.93222413],
            [6.59268451, -0.24110846, 0.67518089],
            [-1.40642865, 3.89105389, 1.91331067],
            [-3.58348389, 2.63770161, -1.39199456],
            [2.02876969, -3.64312321, -3.86484559],
            [5.50749895, 2.29937636, -1.27523833],
            [-1.57790934, -3.25240871, 1.26577812],
            [1.58701189, -3.73659355, 2.13717046],
            [-2.74655355, 2.12741099, -3.63901795],
            [0.95699694, 4.22433168, 2.93682308],
            [2.14879577, -4.33299461, -2.36046031],
        ]
    )


@pytest.fixture
def random_pa_inertias2():
    return np.array(
        [
            [8.56166542e03, 1.36424205e-11, -2.61479727e-12],
            [1.36424205e-11, 9.34288469e03, -1.25055521e-12],
            [-2.61479727e-12, -1.25055521e-12, 1.23507949e04],
        ]
    )


@pytest.fixture
def random_rot_consts2():
    return [
        np.float64(59.02811891121509),
        np.float64(54.09239453126498),
        np.float64(40.918743034766095),
    ]


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
def random_COM_inertias3():
    return np.array(
        [
            [6627.98222727, -1318.01260243, 204.33544295],
            [-1318.01260243, 5875.35646453, 24.77398136],
            [204.33544295, 24.77398136, 8009.11952539],
        ]
    )


@pytest.fixture
def random_evals3():
    return np.array([4874.4618115, 7572.86642365, 8065.12998205])


@pytest.fixture
def random_evecs3():
    return np.array(
        [
            [-0.60360442, -0.74696963, 0.27874375],
            [-0.79597672, 0.58459783, -0.15705552],
            [0.04563729, 0.31667294, 0.94743627],
        ]
    )


@pytest.fixture
def random_pa_coords3():
    return np.array(
        [
            [-1.2266167, 5.62559524, -0.68862095],
            [-1.36349489, -3.5615436, 0.33577622],
            [3.94361828, 1.09144973, 1.3557945],
            [-6.34396973, -1.10802758, -0.91946249],
            [-1.74662306, 1.54185803, 2.33758032],
            [-0.59838802, 0.98101607, -5.40780287],
            [-1.02688675, -1.27191354, -1.81866345],
            [5.90216336, -0.02098458, -1.25419815],
            [3.09232181, -2.07665614, -2.5331536],
            [4.42996416, 0.84218229, 0.21092402],
            [-2.01703806, 5.23351755, -0.21747674],
            [-6.27522663, 0.33634895, 3.51925308],
            [-2.52371252, 0.82263065, -1.88832695],
            [1.40089926, -2.39273128, 2.63446669],
        ]
    )


@pytest.fixture
def random_pa_inertias3():
    return np.array(
        [
            [4.87446181e03, 1.98951966e-13, 1.70530257e-13],
            [1.98951966e-13, 7.57286642e03, -9.09494702e-13],
            [1.70530257e-13, -9.09494702e-13, 8.06512998e03],
        ]
    )


@pytest.fixture
def random_rot_consts3():
    return [
        np.float64(103.67893403276554),
        np.float64(66.73549701363824),
        np.float64(62.66222686113838),
    ]


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
def random_COM_inertias4():
    return np.array(
        [
            [10643.84624275, -1006.43865782, 1109.07261218],
            [-1006.43865782, 10018.75883211, -242.83026478],
            [1109.07261218, -242.83026478, 7633.14963987],
        ]
    )


@pytest.fixture
def random_evals4():
    return np.array([7265.8834173, 9371.47285745, 11658.39843998])


@pytest.fixture
def random_evecs4():
    return np.array(
        [
            [-0.32084209, 0.49589771, -0.80693606],
            [-0.03380536, 0.84543917, 0.53300076],
            [0.94652921, 0.19828784, -0.25448847],
        ]
    )


@pytest.fixture
def random_pa_coords4():
    return np.array(
        [
            [-4.95133033, 1.51944788, -2.61011256],
            [-1.9017136, 0.55972501, 1.70822315],
            [3.70693583, -2.46872892, -1.56025904],
            [4.10894783, 2.67647059, -0.15745985],
            [-0.52284249, -0.50904, 0.38102636],
            [-0.37353441, 0.64739412, -2.97596115],
            [-1.0349838, 2.31258576, 3.96560324],
            [0.66696271, -4.81084372, 0.09245939],
            [2.06930921, 1.85391932, 1.59088389],
            [-3.28070699, -4.01534226, 2.25001578],
            [0.50165911, 2.08598093, 1.23648428],
            [2.87913452, -1.04224756, -3.74484484],
            [2.51888979, 2.17601527, 0.48676291],
            [-5.14993454, 1.91077655, -1.62740493],
            [3.81657771, 5.72044369, 1.09698684],
        ]
    )


@pytest.fixture
def random_pa_inertias4():
    return np.array(
        [
            [7.26588342e03, -1.76214598e-12, 5.33795230e-13],
            [-1.76214598e-12, 9.37147286e03, 1.78168591e-12],
            [5.33795230e-13, 1.78168591e-12, 1.16583984e04],
        ]
    )


@pytest.fixture
def random_rot_consts4():
    return [
        np.float64(69.55506654515754),
        np.float64(53.92738284444609),
        np.float64(43.348921998318616),
    ]


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
def random_COM_inertias5():
    return np.array(
        [
            [19149.07694514, -1157.89150809, 453.37490896],
            [-1157.89150809, 16041.64803492, 2238.43270098],
            [453.37490896, 2238.43270098, 16644.31320216],
        ]
    )


@pytest.fixture
def random_evals5():
    return np.array([13825.3778147, 18430.26331592, 19579.39705161])


@pytest.fixture
def random_evecs5():
    return np.array(
        [
            [0.21606256, 0.37512542, 0.90144211],
            [0.74741099, 0.53051595, -0.39991204],
            [-0.62824659, 0.76015376, -0.16574823],
        ]
    )


@pytest.fixture
def random_pa_coords5():
    return np.array(
        [
            [-1.36491879, 2.27482121, 3.52473374],
            [-1.04902705, -4.46283187, 2.8452703],
            [-1.37802967, 1.61899145, -1.35083635],
            [-0.18156123, 0.81864177, 2.32709339],
            [-2.96381744, -0.30061355, -1.53948903],
            [-4.84229174, 0.67828627, -3.08813377],
            [-2.19995022, 3.70914477, 3.12749209],
            [5.31130396, -2.76326413, -3.76566165],
            [0.02044518, -4.69710212, -1.80724994],
            [0.91328368, 3.09878382, 1.40741807],
            [-0.40179875, 2.22310784, -0.80853195],
            [-0.34447496, -0.7031461, 4.79998423],
            [4.70593341, 0.90853809, -2.1002979],
            [3.45965372, 2.73792113, -2.1484051],
            [6.8715062, 0.15599908, 1.00040632],
            [-2.65568142, -0.23051814, 6.30220231],
            [0.60244886, -4.10106267, 3.36388286],
            [4.62730372, -2.51615635, -0.33535492],
            [-5.40399428, -0.08696955, 0.46229541],
            [5.31722479, 2.73088801, 2.90847977],
        ]
    )


@pytest.fixture
def random_pa_inertias5():
    return np.array(
        [
            [1.38253778e04, 1.93267624e-12, -6.25277607e-12],
            [1.93267624e-12, 1.84302633e04, -2.27373675e-13],
            [-6.25277607e-12, -2.27373675e-13, 1.95793971e04],
        ]
    )


@pytest.fixture
def random_rot_consts5():
    return [
        np.float64(36.55444439736383),
        np.float64(27.42114943976391),
        np.float64(25.811775677664073),
    ]


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


@pytest.fixture
def random_COM_inertias6():
    return np.array(
        [
            [22983.19133622, 950.99688089, 7469.71647789],
            [950.99688089, 24990.77851323, 265.83777493],
            [7469.71647789, 265.83777493, 21682.64871461],
        ]
    )


@pytest.fixture
def random_evals6():
    return np.array([14815.24454289, 24854.83579265, 29986.53822854])


@pytest.fixture
def random_evecs6():
    return np.array(
        [
            [-0.67699564, 0.09870115, 0.72933873],
            [0.04407816, -0.98375064, 0.17404538],
            [0.73466592, 0.14997587, 0.66164433],
        ]
    )


@pytest.fixture
def random_pa_coords6():
    return np.array(
        [
            [4.43995219, 2.44680398, 1.04295166],
            [-0.85177857, 4.41584117, 1.98344145],
            [-2.76531567, -0.56007284, -1.7345059],
            [-3.69104612, -0.49218707, 2.92764129],
            [-4.72676377, -0.82252767, -1.78746071],
            [4.58653127, 3.13364606, 1.20902479],
            [-0.95584511, -1.85226273, 3.09145094],
            [-1.63996457, -4.87166598, -4.6036973],
            [-4.81030746, -0.10215495, -2.37282666],
            [-1.03918098, 2.34779576, -0.25520295],
            [-0.28002899, 2.73157042, -1.070114],
            [3.80040599, -1.70225271, -1.63432759],
            [-0.99592775, 3.15486236, 2.3947271],
            [2.92208339, -0.94227132, -0.33532808],
            [5.28111739, -4.51592014, -0.53192464],
            [0.80097431, -1.96476658, 2.30667914],
            [4.22858883, 3.18128431, -3.28382202],
            [0.33084856, -4.34521461, 1.11965866],
            [4.56084692, -1.81008253, -2.25378607],
            [-4.4117627, -3.30556069, -0.87007618],
            [-5.42276308, 2.37298461, -1.03559103],
            [4.60308907, 3.98490971, 0.55061531],
            [-5.6970619, 1.79777652, 1.13544039],
            [-5.58576093, 3.00858549, -1.21537377],
            [5.17691718, -4.39847708, 1.34955167],
        ]
    )


@pytest.fixture
def random_pa_inertias6():
    return np.array(
        [
            [1.48152445e04, 5.22959454e-12, 9.66338121e-13],
            [5.22959454e-12, 2.48548358e04, -5.11590770e-13],
            [9.66338121e-13, -5.11590770e-13, 2.99865382e04],
        ]
    )


@pytest.fixture
def random_rot_consts6():
    return [
        np.float64(34.112093333121415),
        np.float64(20.33322645203411),
        np.float64(16.853529432052575),
    ]


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
        value = get_inertia_matrix(coordinates, masses)
        np.testing.assert_allclose(value, inertias)

    # TODO: add test for diagonal symmetry


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
# @pytest.fixture
# def hn3_symbols():
#     return ["H", "N", "N", "N"]


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
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pheavy_inputs"],
    )
    def test_wrong_n_atoms(self, inputs, request):
        symbols, mass_numbers, n_atoms = request.getfixturevalue(inputs)
        n_atoms = 30
        with pytest.raises(IndexError) as exc:
            result = get_isotopes_dict(symbols, mass_numbers, n_atoms)
        assert exc.type is IndexError

    @pytest.mark.parametrize(
        "inputs",
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pheavy_inputs"],
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
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pheavy_inputs"],
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
def pheavy_unique_isotopes():
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


class Test_get_isotopes_mass_cache:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        diagonalize.clear_isotope_mass_cache()
        yield
        diagonalize.clear_isotope_mass_cache()

    def test_repeated_lookup_hits_cache(self, monkeypatch):
        calls = []

        class _MockIsotope:
            def __init__(self, mass):
                self.mass = mass

        def mock_isotope(symbol, mass_number):
            calls.append((symbol, mass_number))
            return _MockIsotope(1.2345)

        monkeypatch.setattr(diagonalize, "isotope", mock_isotope)

        result1 = get_isotopes_mass("H", 1)
        result2 = get_isotopes_mass("H", 1)

        assert np.allclose(result1, 1.2345)
        assert np.allclose(result2, 1.2345)
        assert calls == [("H", 1)]

    def test_distinct_keys_each_lookup_once(self, monkeypatch):
        calls = []
        mass_map = {
            ("H", 1): 1.0,
            ("H", 2): 2.0,
        }

        class _MockIsotope:
            def __init__(self, mass):
                self.mass = mass

        def mock_isotope(symbol, mass_number):
            calls.append((symbol, mass_number))
            return _MockIsotope(mass_map[(symbol, mass_number)])

        monkeypatch.setattr(diagonalize, "isotope", mock_isotope)

        assert np.allclose(get_isotopes_mass("H", 1), 1.0)
        assert np.allclose(get_isotopes_mass("H", 2), 2.0)
        assert np.allclose(get_isotopes_mass("H", 1), 1.0)

        assert calls.count(("H", 1)) == 1
        assert calls.count(("H", 2)) == 1


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
def pheavy_unique_isotopes_dict():
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
                "pheavy_unique_isotopes",
                "pheavy_unique_isotopes_dict",
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


class Test_get_mol_masses:
    @pytest.mark.parametrize(
        "inputs",
        ["hn3_inputs", "dn3_inputs", "pyridazine_inputs", "pheavy_inputs"],
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
            ("hn3_inputs", "hn3_mol_masses"),
            ("dn3_inputs", "dn3_mol_masses"),
            ("pyridazine_inputs", "pyridazine_mol_masses"),
            ("pheavy_inputs", "pheavy_mol_masses"),
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


class Test_get_eigens:
    """
    This should be fairly minimal - mostly to make sure that the matching
    of eigenvalues to eigenvectors is correct.
    """

    @pytest.mark.parametrize(
        "f_COM_inertia, f_evals, f_evecs",
        [
            ("random_COM_inertias1", "random_evals1", "random_evecs1"),
            ("random_COM_inertias2", "random_evals2", "random_evecs2"),
            ("random_COM_inertias3", "random_evals3", "random_evecs3"),
            ("random_COM_inertias4", "random_evals4", "random_evecs4"),
            ("random_COM_inertias5", "random_evals5", "random_evecs5"),
            ("random_COM_inertias6", "random_evals6", "random_evecs6"),
        ],
    )
    def test_expected_results(self, f_COM_inertia, f_evals, f_evecs, request):
        COM_inertia = request.getfixturevalue(f_COM_inertia)
        evals = request.getfixturevalue(f_evals)
        evecs = request.getfixturevalue(f_evecs)

        r_evals, r_evecs = get_eigens(COM_inertia)
        assert np.allclose(r_evals, evals) and np.allclose(r_evecs, evecs)

    # TODO: Test negative conditions


class Test_rotate_coordinates:
    @pytest.mark.parametrize(
        "f_COM_coordinate,f_evecs,f_pa_coordinate",
        [
            ("random_COM_coords1", "random_evecs1", "random_pa_coords1"),
            ("random_COM_coords2", "random_evecs2", "random_pa_coords2"),
            ("random_COM_coords3", "random_evecs3", "random_pa_coords3"),
            ("random_COM_coords4", "random_evecs4", "random_pa_coords4"),
            ("random_COM_coords5", "random_evecs5", "random_pa_coords5"),
            ("random_COM_coords6", "random_evecs6", "random_pa_coords6"),
        ],
    )
    def test_expected_results(
        self, f_COM_coordinate, f_evecs, f_pa_coordinate, request
    ):
        COM_coordinates = request.getfixturevalue(f_COM_coordinate)
        evecs = request.getfixturevalue(f_evecs)
        pa_coordinates = request.getfixturevalue(f_pa_coordinate)

        result = rotate_coordinates(COM_coordinates, evecs)

        assert np.allclose(result, pa_coordinates)

    # TODO: Test negative conditions


class Test_get_isotopologue_principal_axes:
    @pytest.mark.parametrize(
        "f_coords,f_mass_nums,f_atom_symbols,f_n_atoms,f_mol_masses,f_COM_coords,f_COM_inertia,f_evecs,f_evals,f_pa_coords,f_pa_inertia,f_COM_value",
        [
            (
                "hn3_coords",
                "hn3_mass_numbers",
                "hn3_symbols",
                "hn3_n_atoms",
                "hn3_mol_masses",
                "hn3_COM_coords",
                "hn3_COM_inertia",
                "hn3_evecs",
                "hn3_evals",
                "hn3_pa_coords",
                "hn3_pa_inertia",
                "hn3_COM_value",
            ),
            (
                "dn3_coords",
                "dn3_mass_numbers",
                "dn3_symbols",
                "dn3_n_atoms",
                "dn3_mol_masses",
                "dn3_COM_coords",
                "dn3_COM_inertia",
                "dn3_evecs",
                "dn3_evals",
                "dn3_pa_coords",
                "dn3_pa_inertia",
                "dn3_COM_value",
            ),
            (
                "pyridazine_coords",
                "pyridazine_mass_numbers",
                "pyridazine_symbols",
                "pyridazine_n_atoms",
                "pyridazine_mol_masses",
                "pyridazine_COM_coords",
                "pyridazine_COM_inertia",
                "pyridazine_evecs",
                "pyridazine_evals",
                "pyridazine_pa_coords",
                "pyridazine_pa_inertia",
                "pyridazine_COM_value",
            ),
            (
                "pheavy_coords",
                "pheavy_mass_numbers",
                "pheavy_symbols",
                "pheavy_n_atoms",
                "pheavy_mol_masses",
                "pheavy_COM_coords",
                "pheavy_COM_inertia",
                "pheavy_evecs",
                "pheavy_evals",
                "pheavy_pa_coords",
                "pheavy_pa_inertia",
                "pheavy_COM_value",
            ),
        ],
    )
    def test_expected_results(
        self,
        f_coords,
        f_mass_nums,
        f_atom_symbols,
        f_n_atoms,
        f_mol_masses,
        f_COM_coords,
        f_COM_inertia,
        f_evecs,
        f_evals,
        f_pa_coords,
        f_pa_inertia,
        f_COM_value,
        request,
    ):
        coords = request.getfixturevalue(f_coords)
        mass_nums = request.getfixturevalue(f_mass_nums)
        atom_symbols = request.getfixturevalue(f_atom_symbols)
        n_atoms = request.getfixturevalue(f_n_atoms)

        mol_masses = request.getfixturevalue(f_mol_masses)
        COM_coords = request.getfixturevalue(f_COM_coords)
        COM_inertia = request.getfixturevalue(f_COM_inertia)
        evecs = request.getfixturevalue(f_evecs)
        evals = request.getfixturevalue(f_evals)
        pa_coords = request.getfixturevalue(f_pa_coords)
        pa_inertia = request.getfixturevalue(f_pa_inertia)
        COM_value = request.getfixturevalue(f_COM_value)

        (
            r_mol_masses,
            r_COM_coords,
            r_COM_inertia,
            r_evecs,
            r_evals,
            r_pa_coords,
            r_pa_inertia,
            r_COM_value,
        ) = get_isotopologue_principal_axes(coords, mass_nums, atom_symbols, n_atoms)

        assert all(
            [
                np.allclose(i, j)
                for i, j in (
                    [mol_masses, r_mol_masses],
                    [COM_coords, r_COM_coords],
                    [COM_inertia, r_COM_inertia],
                    [evecs, r_evecs],
                    [evals, r_evals],
                    [pa_coords, r_pa_coords],
                    [pa_inertia, r_pa_inertia],
                    [COM_value, r_COM_value],
                )
            ]
        )


class Test_check_for_length_mismatch:
    @pytest.mark.parametrize(
        "listlike,expected",
        [
            ([1, 2, 3], 3),
            ((1, 2, 3, 4), 4),
            ("ab", 2),
        ],
    )
    def test_matching_lengths(self, listlike, expected, capsys):
        result = check_for_length_mismatch(
            listlike, expected, "This string should not be printed"
        )
        captured = capsys.readouterr()
        assert (result is None) and (captured.out == "") and (captured.err == "")

    @pytest.mark.parametrize(
        "listlike,expected",
        [
            ([1, 2, 3], 1),
            ((1, 2, 3, 4), 1),
            ("ab", 1),
        ],
    )
    def test_mismatched_lengths(self, listlike, expected):
        with pytest.raises(ValueError) as exc:
            result = check_for_length_mismatch(
                listlike, expected, "This string should be printed"
            )
        expected_message = f"actual_length={len(listlike)} vs expected_length={expected}: This string should be printed"
        assert exc.type is ValueError and (expected_message in str(exc.value))

    @pytest.mark.parametrize(
        "listlike,expected",
        [
            (1, 1),
            ((1, 2, 3, 4), "asdf"),
        ],
    )
    def test_bad_types(self, listlike, expected):
        with pytest.raises(TypeError) as exc:
            result = check_for_length_mismatch(
                listlike, expected, "This string should not be printed"
            )
        assert exc.type is TypeError


class Test_check_for_bad_diagonal:
    @pytest.mark.parametrize(
        "matrix,eigenvalues",
        [
            (np.array([[1, 0], [0, 2]]), np.array([1, 2])),
            (
                np.array([[-1, 0, 0], [0, 34, 0], [0, 0, -200]]),
                np.array([-1, 34, -200]),
            ),
        ],
    )
    def test_good_diagonal(self, matrix, eigenvalues, capsys):
        result = check_for_bad_diagonal(
            matrix, eigenvalues, "This string should not be printed"
        )
        captured = capsys.readouterr()
        assert (result is None) and (captured.out == "") and (captured.err == "")

    @pytest.mark.parametrize(
        "matrix, eigenvalues",
        [
            (np.array([[1, 0], [0, 2]]), np.array([0, 0])),
            (
                np.array([[-1, 0, 0], [0, 34, 0], [0, 0, -200]]),
                np.array([0, 0, 0]),
            ),
        ],
    )
    def test_bad_diagonal(self, matrix, eigenvalues, capsys):
        result = check_for_bad_diagonal(
            matrix, eigenvalues, "This string should be printed"
        )
        captured = capsys.readouterr()
        assert (
            (result is None)
            and ("This string should be printed" in captured.out)
            and (captured.err == "")
        )


class Test_transform_dipole:
    @pytest.mark.parametrize(
        "f_dipole,f_evecs,f_pa_dipole",
        [
            ("hn3_dipole", "hn3_evecs", "hn3_pa_dipole"),
            ("dn3_dipole", "dn3_evecs", "dn3_pa_dipole"),
            ("pyridazine_dipole", "pyridazine_evecs", "pyridazine_pa_dipole"),
            ("pheavy_dipole", "pheavy_evecs", "pheavy_pa_dipole"),
        ],
    )
    def test_expected(self, f_dipole, f_evecs, f_pa_dipole, request):
        dipole = request.getfixturevalue(f_dipole)
        evecs = request.getfixturevalue(f_evecs)
        pa_dipole = request.getfixturevalue(f_pa_dipole)

        result = transform_dipole(dipole, evecs)

        assert np.allclose(result, pa_dipole)


class Test_get_principal_axes:
    def test_hn3_dn3(
        self,
        hn3_mass_numbers,
        dn3_mass_numbers,
        hn3_n_atoms,
        hn3_symbols,
        hn3_coords,
        hn3_dipole,
        hn3_mol_masses,
        dn3_mol_masses,
        hn3_rot_consts,
        dn3_rot_consts,
        hn3_pa_dipole,
        dn3_pa_dipole,
        hn3_pa_coords,
        dn3_pa_coords,
        hn3_pa_inertia,
        dn3_pa_inertia,
        hn3_COM_coords,
        dn3_COM_coords,
        hn3_COM_inertia,
        dn3_COM_inertia,
        hn3_evecs,
        dn3_evecs,
        hn3_evals,
        dn3_evals,
        hn3_COM_value,
        dn3_COM_value,
    ):
        isotopologue_names = ["iso001", "iso002"]
        isotopologue_dict = {
            "iso001": hn3_mass_numbers,
            "iso002": dn3_mass_numbers,
        }
        n_atoms = hn3_n_atoms
        atom_symbols = hn3_symbols
        mol_coordinates = hn3_coords
        mol_dipole = hn3_dipole

        atom_masses = {
            "iso001": hn3_mol_masses,
            "iso002": dn3_mol_masses,
        }

        rot_consts = {
            "iso001": hn3_rot_consts,
            "iso002": dn3_rot_consts,
        }

        pa_dipoles = {
            "iso001": hn3_pa_dipole,
            "iso002": dn3_pa_dipole,
        }

        pa_coordinates = {
            "iso001": hn3_pa_coords,
            "iso002": dn3_pa_coords,
        }

        pa_inertias = {
            "iso001": hn3_pa_inertia,
            "iso002": dn3_pa_inertia,
        }

        COM_coords = {
            "iso001": hn3_COM_coords,
            "iso002": dn3_COM_coords,
        }

        COM_inertias = {
            "iso001": hn3_COM_inertia,
            "iso002": dn3_COM_inertia,
        }

        evecs = {
            "iso001": hn3_evecs,
            "iso002": dn3_evecs,
        }

        evals = {
            "iso001": hn3_evals,
            "iso002": dn3_evals,
        }

        COM_values = {
            "iso001": hn3_COM_value,
            "iso002": dn3_COM_value,
        }

        (
            r_atom_masses,
            r_rot_consts,
            r_pa_dipoles,
            r_pa_coordinates,
            r_pa_inertias,
            r_COM_coords,
            r_COM_inertias,
            r_evecs,
            r_evals,
            r_COM_values,
        ) = get_principal_axes(
            isotopologue_names,
            isotopologue_dict,
            n_atoms,
            atom_symbols,
            mol_coordinates,
            mol_dipole,
        )

        assert all(
            [
                all(
                    [
                        (k in dict2.keys() and np.allclose(v, dict2[k]))
                        for k, v in dict1.items()
                    ]
                )
                for dict1, dict2 in (
                    (r_atom_masses, atom_masses),
                    (r_rot_consts, rot_consts),
                    (r_pa_dipoles, pa_dipoles),
                    (r_pa_inertias, pa_inertias),
                    (r_COM_coords, COM_coords),
                    (r_COM_inertias, COM_inertias),
                    (r_evecs, evecs),
                    (r_evals, evals),
                    (r_COM_values, COM_values),
                )
            ]
        )

    def test_pyridazine_and_heavy(
        self,
        pyridazine_mass_numbers,
        pheavy_mass_numbers,
        pyridazine_n_atoms,
        pyridazine_symbols,
        pyridazine_coords,
        pyridazine_dipole,
        pyridazine_mol_masses,
        pheavy_mol_masses,
        pyridazine_rot_consts,
        pheavy_rot_consts,
        pyridazine_pa_dipole,
        pheavy_pa_dipole,
        pyridazine_pa_coords,
        pheavy_pa_coords,
        pyridazine_pa_inertia,
        pheavy_pa_inertia,
        pyridazine_COM_coords,
        pheavy_COM_coords,
        pyridazine_COM_inertia,
        pheavy_COM_inertia,
        pyridazine_evecs,
        pheavy_evecs,
        pyridazine_evals,
        pheavy_evals,
        pyridazine_COM_value,
        pheavy_COM_value,
    ):
        isotopologue_names = ["iso001", "iso002"]
        isotopologue_dict = {
            "iso001": pyridazine_mass_numbers,
            "iso002": pheavy_mass_numbers,
        }
        n_atoms = pyridazine_n_atoms
        atom_symbols = pyridazine_symbols
        mol_coordinates = pyridazine_coords
        mol_dipole = pyridazine_dipole

        atom_masses = {
            "iso001": pyridazine_mol_masses,
            "iso002": pheavy_mol_masses,
        }

        rot_consts = {
            "iso001": pyridazine_rot_consts,
            "iso002": pheavy_rot_consts,
        }

        pa_dipoles = {
            "iso001": pyridazine_pa_dipole,
            "iso002": pheavy_pa_dipole,
        }

        pa_coordinates = {
            "iso001": pyridazine_pa_coords,
            "iso002": pheavy_pa_coords,
        }

        pa_inertias = {
            "iso001": pyridazine_pa_inertia,
            "iso002": pheavy_pa_inertia,
        }

        COM_coords = {
            "iso001": pyridazine_COM_coords,
            "iso002": pheavy_COM_coords,
        }

        COM_inertias = {
            "iso001": pyridazine_COM_inertia,
            "iso002": pheavy_COM_inertia,
        }

        evecs = {
            "iso001": pyridazine_evecs,
            "iso002": pheavy_evecs,
        }

        evals = {
            "iso001": pyridazine_evals,
            "iso002": pheavy_evals,
        }

        COM_values = {
            "iso001": pyridazine_COM_value,
            "iso002": pheavy_COM_value,
        }

        (
            r_atom_masses,
            r_rot_consts,
            r_pa_dipoles,
            r_pa_coordinates,
            r_pa_inertias,
            r_COM_coords,
            r_COM_inertias,
            r_evecs,
            r_evals,
            r_COM_values,
        ) = get_principal_axes(
            isotopologue_names,
            isotopologue_dict,
            n_atoms,
            atom_symbols,
            mol_coordinates,
            mol_dipole,
        )

        assert all(
            [
                all(
                    [
                        (k in dict2.keys() and np.allclose(v, dict2[k]))
                        for k, v in dict1.items()
                    ]
                )
                for dict1, dict2 in (
                    (r_atom_masses, atom_masses),
                    (r_rot_consts, rot_consts),
                    (r_pa_dipoles, pa_dipoles),
                    (r_pa_inertias, pa_inertias),
                    (r_COM_coords, COM_coords),
                    (r_COM_inertias, COM_inertias),
                    (r_evecs, evecs),
                    (r_evals, evals),
                    (r_COM_values, COM_values),
                )
            ]
        )
