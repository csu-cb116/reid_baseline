from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .vessel_min import Vessel_min
from .veri import VeRi
from .vehicleid import VehicleID

__imgreid_factory = {
    'veri': VeRi,
    'vehicleID': VehicleID,
    'vessel_min': Vessel_min
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)

