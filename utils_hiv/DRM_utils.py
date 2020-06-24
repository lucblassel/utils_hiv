import os
import pandas as pd

HERE = os.path.dirname(__file__)
MUTATION_FILE = "data/mutation_characteristics.tsv"

MUTATIONS = pd.read_csv(os.path.join(HERE, MUTATION_FILE), sep="\t")


def get_SDRMs():
    """returns SDRMs only"""
    return MUTATIONS[MUTATIONS["SDRM"] == 1]["feature"].values.tolist()


def get_DRMs_only():
    """returns DRMs that are not SDRMs"""
    return MUTATIONS[MUTATIONS["SDRM"] == 0]["feature"].values.tolist()


def get_all_DRMs():
    """returns all known DRMs"""
    return MUTATIONS["feature"].values.tolist()


def get_accessory():
    """
    returns mutations that are mentioned as 'accessory' in the
    Standford HIVdb mutation comments.
    """
    return MUTATIONS[MUTATIONS["accessory"] == 1]["feature"].values.tolist()


def get_standalone():
    """returns non accessory mutations"""
    return MUTATIONS[MUTATIONS["accessory"] == 0]["feature"].values.tolist()


def get_by_score(score=60):
    """
    returns mutations that have an HIVdb mutation score lower 
    than a given value.
    """
    return MUTATIONS[MUTATIONS["HIVDB Score"] >= score]["feature"].values.tolist()


def get_NRTI():
    """returns mutations caused by NRTIs"""
    return MUTATIONS[MUTATIONS["type"] == "NRTI"]["feature"].values.tolist()


def get_NNRTI():
    """returns mutations caused by NNRTIs"""
    return MUTATIONS[MUTATIONS["type"] == "NNRTI"]["feature"].values.tolist()


def get_Other():
    """returns mutations caused by neither NRTIs nor NNRTIs"""
    return MUTATIONS[MUTATIONS["type"] == "Other"]["feature"].values.tolist()
