from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .vessel_jm import Vessel_jm
from .vessel_jun import Vessel_jun
from .vessel_min import Vessel_min
from .veri import VeRi
from .vehicleid import VehicleID
from .Market1501 import Market1501

__imgreid_factory = {
    'veri': VeRi,
    'vehicleID': VehicleID,
    'vessel_jun': Vessel_jun,
    'vessel_min': Vessel_min,
    'market1501': Market1501,
    'vessel_jm': Vessel_jm,
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)

