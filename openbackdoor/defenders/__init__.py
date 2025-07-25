from .defender import Defender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender
from .onion_defender import ONIONDefender
from .bki_defender import BKIDefender
from .cube_defender import CUBEDefender
from .dan_defender import DANDefender
from .badacts_defender import BadActsDefender
from .masf_defender import MASFDefender
from .cluster_defender import ClusterDefender
from .md_defender import MDDefender
from .ms_defender import MSDefender
from .msmlp_defender import MSMLPDefender



DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
    'onion': ONIONDefender,
    'bki':  BKIDefender,
    'cube': CUBEDefender,
    'dan': DANDefender,
    'badacts': BadActsDefender,
    'masf': MASFDefender,
    'cluster': ClusterDefender,
    'md': MDDefender,
    'ms': MSDefender,
    'msmlp':MSMLPDefender

}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
