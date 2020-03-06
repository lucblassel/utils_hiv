import os
import pickle
import re

import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from .DRM_utils import *

HERE = os.path.dirname(__file__)
CONSENSUS_FASTA = "data/consensus.fa"


def _IdsFromSubtypes(subtypes):
    """
    Get id for consensus sequence from subtype in dataset
    """
    name = "CONSENSUS_"
    names = []
    recombinants = re.compile(r'CRF\d{2}_\S+')
    for subtype in subtypes:
        if subtype in ["B", "C", "D", "G", "H"]:
            names += [name + subtype]
        elif subtype in ["A", "F"]:
            names += [name + subtype + str(i) for i in range(1, 3)]
        elif recombinants.match(subtype):
            end = subtype[3:].split('_')
            names += [name + end[0] + "_" + end[1].upper()]

    return names


def _readSeqs(subtypes):
    """
    Get consensus sequences for given subtypes as a list
    """
    ids = _IdsFromSubtypes(subtypes)
    records = list(SeqIO.parse(os.path.join(HERE, CONSENSUS_FASTA), 'fasta'))
    if ids:
        return [rec.seq for rec in records if rec.id in ids]
    else:
        return [rec.seq for rec in records]


def _getConsensusDict(subtypes=None):
    """
    Get consensus sequences for given subtypes as a dict with sequence IDs
    """
    ids = _IdsFromSubtypes(subtypes)
    records = list(SeqIO.parse(os.path.join(HERE, CONSENSUS_FASTA), 'fasta'))
    if ids:
        return {rec.id: rec.seq for rec in records if rec.id in ids}
    else:
        return {rec.id: rec.seq for rec in records}


def getConsensusAAs(subtypes):
    """
    Get features (pos_AA) corresponding to consensus features of given subtypes
    """
    seqs = _readSeqs(subtypes)
    vars = [["{}_{}".format(i + 1, AA) for (i, AA) in enumerate(seq)]
            for seq in seqs]

    return list(set([var for varSet in vars for var in varSet]))


def split_data(dataset, target):
    X = dataset.filter(regex=r'\d+_\S', axis=1)
    y = dataset[target]

    return X, y


def join_data(X, y):
    return pd.concat([X, y], axis=1)


def _read_dataset(dataset_path, target):
    dataset = pd.read_csv(dataset_path, sep='\t', header=0, index_col=0)
    drm_status = dataset.get('hasDRM', None)
    X, y = split_data(dataset, target)

    return X, y, drm_status


def deep_clean_features(dataset):
    """
    remove features corresponding to gaps, stop codons and unknown AAs
    """
    features_to_clean = dataset.filter(regex=r'\d+_(nan|-|\*|X)',
                                       axis=1).columns
    return dataset.drop(features_to_clean, axis=1)


def remove_consensus(dataset, subtypes):
    """
    remove features corresponding to consensus AA of given subtypes from dataset
    """
    consensus = getConsensusAAs(subtypes)
    return dataset.drop(columns=consensus, errors='ignore')


def removal_wrapper(dataset, choice):
    """
    Remove features corresponding to DRM subclass of choice
    """
    choice_dict = {
        'SDRM': get_SDRMs,
        'DRM': get_DRMs_only,
        'ALL': get_all_DRMs,
        'ACCESSORY': get_accessory,
        'STANDALONE': get_standalone,
        'NRTI': get_NRTI,
        'NNRTI': get_NNRTI,
        'OTHER': get_Other
    }

    cols = choice_dict.get(choice.upper(), lambda: [])()

    return dataset.drop(columns=cols, errors='ignore')


def remove_naive_drms(train, test, drm_status, target):
    """
    Move naive sequences with DRMs from training set to testing set
    """
    naive_drms_index = train[(train[target] == 0) & (drm_status == 1)].index
    new_test = pd.concat([test, train.loc[naive_drms_index]], axis=0)
    new_train = train.drop(naive_drms_index, axis=0)

    return new_train, new_test


def sequence_removal_wrapper(dataset, choice):
    """
    remove sequences depending on their mutated status
    """
    if choice.upper() == "DRM":
        return dataset[dataset['hasDRM'] == 0]
    elif choice.upper() == "NO DRM":
        return dataset[dataset['hasDRM'] == 1]
    else:
        return dataset


def subsampling_balance(data, target):
    """
    class balance binary task dataset
    """
    values = data[target].value_counts()
    _, min_n, maj_l, min_l = *values, *values.index
    maj_df = data[data[target] == maj_l]
    min_df = data[data[target] == min_l]

    balanced = pd.concat([min_df, maj_df.sample(n=min_n, replace=False)]).\
        sample(frac=1) # shuffling data

    return balanced


def homogenize_datasets(complete_path, external_path):
    """
    Give union of features of all datasets to all datasets
    """
    complete = pickle.load(open(complete_path, 'rb'))
    external = pickle.load(open(external_path, 'rb'))

    total = pd.concat([complete, external], axis=0, sort=False)
    total.fillna(0, inplace=True)
    total['hasDRM'] = total.filter(DRMS, axis=1).any(axis=1).apply(int)
    total['is_resistant'] = ((total['hasDRM'] == 1) |
                             (total['encoded_label'] == 1)).apply(int)

    types = {
        col: 'int8' if col != 'label' else 'object'
        for col in total.columns
    }
    total = total.astype(dtype=types)

    complete_same_features = total.loc[complete.index]
    external_same_features = total.loc[external.index]

    return complete_same_features, external_same_features


def choose_subtype(dataset_path, metadata_path, subtype):
    """
    select samples of given subtype from dataset
    """
    dataset = pd.read_csv(dataset_path, sep='\t', header=0, index_col=0)
    metadata = pd.read_csv(metadata_path, sep='\t').set_index('id')

    index = metadata[metadata.filter(regex=r'.*(s|S)ubtype.*',
                                     axis=1).iloc[:, 0] == subtype].index

    return dataset.filter(index, axis=0)


def split_dataset(dataset_path, target, test_proportion=0.2):
    """
    split dataset in training and testing set
    """
    X, y, drm_status = _read_dataset(dataset_path, target)

    split = train_test_split(X, y, test_size=test_proportion, stratify=y)

    train = pd.concat((split[0], split[2]), axis=1)
    test = pd.concat((split[1], split[3]), axis=1)

    return train, test, drm_status


def get_parameter_cv_folds(dataset_path,
                           num_folds,
                           num_repeats,
                           target,
                           balance=False):
    def fold_generator(splitter, X, y, balance):
        for split_train, validation in splitter.split(X, y):
            train = X.iloc[split_train].join(y.iloc[split_train])
            test = X.iloc[validation].join(y.iloc[validation])
            if balance:
                train = subsampling_balance(train, target)
                test = subsampling_balance(test, target)

            yield train, test

    X, y, _ = _read_dataset(dataset_path, target)

    splitter = RepeatedStratifiedKFold(n_splits=num_folds,
                                       n_repeats=num_repeats)

    return fold_generator(splitter, X, y, balance)


def resample_binary_task(data, final_pos_freq=0.5, target='encoded_label'):
    """
    resample binary task dataset with a given frequency of positive samples from a given dataset
    """
    neg, pos = data[data[target] == 0], data[data[target] == 1]

    if final_pos_freq >= (len(pos) / len(data)):
        final_N = round(len(pos) / final_pos_freq)
    else:
        final_N = round(len(neg) / (1 - final_pos_freq))
    neg_count, pos_count = round(
        (1 - final_pos_freq) * final_N), round(final_pos_freq * final_N)
    final_neg, final_pos = neg.sample(n=neg_count), pos.sample(n=pos_count)
    return pd.concat([final_neg, final_pos], axis=0).sample(frac=1)
