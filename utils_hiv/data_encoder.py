import os
import pickle
import re

import pandas as pd
from Bio import SeqIO
from category_encoders import OneHotEncoder

from .DRM_utils import get_all_DRMs

HERE = os.path.dirname(__file__)
SEQUENCE_ID_COL = "inputSequence"
CONSENSUS_FASTA = "data/consensus.fa"

NRTI = ["ABC", "AZT", "FTC", "3TC", "TDF"]
NNRTI = ["DOR", "EFV", "ETR", "NVP", "RPV"]

SUBTYPES = [
    "A",
    "A2",
    "B",
    "C",
    "CRF01_AE",
    "CRF02_AG",
    "CRF04_cpx",
    "CRF05_DF",
    "CRF06_cpx",
    "CRF09_cpx",
    "CRF10_CD",
    "CRF11_cpx",
    "CRF13_cpx",
    "CRF18_cpx",
    "CRF22_01A1",
    "CRF25_cpx",
    "CRF37_cpx",
    "CRF45_cpx",
    "CRF49_cpx",
    "D",
    "F",
    "F2",
    "G",
    "H",
    "J",
    "K",
]


def _IdsFromSubtypes(subtypes):
    """
    Transforms subtypes for which you want the consensus AAs, to the sequence Ids in the consensus sequences fasta file.
    """
    name = "CONSENSUS_"
    names = []
    recombinants = re.compile(r"CRF\d{2}_\S+")
    for subtype in subtypes:
        if subtype in ["B", "C", "D", "G", "H"]:
            names += [name + subtype]
        elif subtype in ["A", "F"]:
            names += [name + subtype + str(i) for i in range(1, 3)]
        elif recombinants.match(subtype):
            end = subtype[3:].split("_")
            names += [name + end[0] + "_" + end[1].upper()]
        else:
            return None

    return names


def _readSeqs(subtypes):
    """
    reads consensus sequences from chosen subtypes
    """
    ids = _IdsFromSubtypes(subtypes)
    records = list(SeqIO.parse(os.path.join(HERE, CONSENSUS_FASTA), "fasta"))
    if ids:
        return [list(rec.seq) for rec in records if rec.id in ids]
    return [list(rec.seq) for rec in records]


def getConsensusAAs(subtypes):
    """
    get OneHot encoded features corresponding to chosen consensus sequences
    """
    seqs = _readSeqs(subtypes)
    vars_ = [
        ["{}_{}".format(i + 1, AA) for (i, AA) in zip(range(len(seq)), seq)]
        for seq in seqs
    ]

    return list(set([var for varSet in vars_ for var in varSet]))


def reader(naive_filename, treated_filename, truncate):
    """reads the 2 .tsv files and returns the concatenated DataFrame

    Arguments:
        naive_filename {String} -- path to file for naive sequences
        treated_filename {String} -- path to file for treated sequences
        truncate {iterable of int} -- wanted start and end positions under
        the form: [start, end]

    Returns:
        pandas.DataFrame -- concatenation of both files
        pandas.Series -- consensus sequence
    """

    truncate_filter = [str(pos) for pos in range(truncate[0], truncate[1] + 1)]
    all_sequences = []

    filenames = {"naive": naive_filename, "treated": treated_filename}

    for label, filename in filenames.items():
        sequences = pd.read_csv(filename, sep="\t").set_index("Sequence Names")
        truncated = sequences.filter(truncate_filter, axis=1)
        truncated["label"] = label
        truncated.index.names = [SEQUENCE_ID_COL]
        all_sequences.append(truncated.drop("Consensus"))

    consensus = truncated.loc["Consensus"].drop("label")
    sequences = pd.concat(all_sequences, axis=0)
    sequences.index = sequences.index.astype(str)

    return (sequences, consensus)


def read_metadata(filename):
    """reads metadata file

    Arguments:
        filenames -- string

    Returns:
        pandas.DataFrame -- metadata Dataframe
    """

    metadata = pd.read_csv(filename, sep="\t").set_index("id")
    metadata.index.names = [SEQUENCE_ID_COL]
    metadata.index = metadata.index.astype(str)

    return metadata


def get_resistance_scores(resistance_files):
    """
    gets the resistance score of each sequence for different ARTs
    from the HIVDB resistance summary files 
    """
    resistances = pd.concat(
        pd.read_csv(name, sep="\t") for name in resistance_files
    ).set_index("Sequence Name")

    scores = resistances.filter([f"{t} Score" for t in NRTI + NNRTI], axis=1)

    scores.columns = scores.columns.map(lambda x: x.replace(" ", "_"))
    scores["mean_score"] = scores.mean(axis=1)
    scores.index = scores.index.astype(str)

    return scores


def fill_consensus_AAs(pretty_pairwise_sequences, consensus):
    """replaces "-" symbol by the consensus AA for that position in the
    sequences DataFrame

    Arguments:
        pretty_pairwise_sequences {pandas.DataFrame} -- Aligned sequences with
        "-' denoting consensus AA and a letter denoting a mutation
        consensus {pandas.Series} -- Consensus AA for HXB2 at each position

    Returns:
        pandas.DataFrame
    """

    def get_AA(AA, col_name, consensus):
        return consensus[col_name] if AA.strip() == "-" else AA

    return pretty_pairwise_sequences.apply(
        lambda col: col.apply(lambda AA: get_AA(AA, col.name, consensus))
    ).applymap(lambda AA: AA.replace(".", "-"))


def get_single_AA_freqs(sequences_df):
    """gets frequencies of single amino acids for each position

    Arguments:
        sequences_df {pandas.DataFrame} -- sequences with only AAs and gaps

    Returns:
        pandas.DataFrame
    """

    def get_series_freqs(series):
        freqs = series.value_counts()
        return freqs.loc[[i for i in freqs.index if len(i) == 1]]

    return sequences_df.apply(get_series_freqs)


def get_single_AAs(sequences_df, total_freqs):
    """for ambiguous AAs, replaces with most frequent AA at position in all
    sequences within the possible AAs in ambiguity.

    Arguments:
        sequences_df {pandas.DataFrame} -- sequences with only AAs and gaps
        total_freqs {pandas.DataFrame} -- frequencies of unambiguous AAs at
        every position

    Returns:
        pandas.DataFrame -- DataFrame with ambiguous AAs replaced
    """

    def get_most_frequent_single_AA(AAs, freqs):
        if len(AAs) == 1 or AAs == "del":
            return AAs
        return freqs.loc[list(AAs)].sort_values(ascending=False).index[0]

    return sequences_df.apply(
        lambda col: col.apply(
            lambda AAs: get_most_frequent_single_AA(AAs, total_freqs[col.name])
        )
        if col.name != "label"
        else col
    )


def choose_subtype(sequences, metadata, chosen_subtype):
    """select sequences of a given subtype

    Arguments:
        sequences {pandas.DataFrame} -- DataFrame of the sequences
        metadata {pandas.DataFrame} -- metadata with info on the subtype
        chosen_subtype {String} -- wanted subtype ('All' or 'None' will select
        all subtypes in the dataset)

    Returns:
        pandas.DataFrame -- sequences of the chosen subtype
    """

    subtype_col = metadata.filter(regex=".*(s|S)ubtype.*", axis=1)
    cleaned = subtype_col[subtype_col.columns[0]].apply(
        lambda subtype: subtype.split("(")[0].strip()
    )
    unique_subtypes = cleaned.unique().tolist()

    if chosen_subtype.upper() in ("ALL", "NONE"):
        return sequences, unique_subtypes

    return (sequences[cleaned == chosen_subtype], unique_subtypes)


def get_features_to_remove(dataset_subtypes):
    """gives list of features to remove from sequences dataset

    Arguments:
        dataset_subtypes {list(String)} -- list of subtypes present in
        sequences dataset

    Returns:
        list(String) -- all features corresponding to consensus AAs and, if the
        option is selected, DRMs
    """

    subtypes_to_remove = [
        subtype for subtype in dataset_subtypes if subtype in SUBTYPES
    ]

    features_to_remove = getConsensusAAs(subtypes_to_remove)

    return features_to_remove


def process(
    naive_file,
    treated_file,
    metadata_file,
    resistance_files,
    outfile,
    subtype="All",
    truncate=[41, 235],
):
    print("reading sequences and metadata")
    raw_sequences, consensus = reader(naive_file, treated_file, truncate)

    metadata = read_metadata(metadata_file)

    print(f"choosing {subtype} subtype(s)")
    chosen_sequences, dataset_subtypes = choose_subtype(
        raw_sequences, metadata, subtype
    )

    print("Filling with consensus AAs")
    AA_sequences = fill_consensus_AAs(chosen_sequences, consensus)
    freqs = get_single_AA_freqs(AA_sequences.drop("label", axis=1))
    single_AA_sequences = get_single_AAs(AA_sequences, freqs)

    print("OneHot encoding")
    columns_to_encode = single_AA_sequences.columns.drop("label")
    encoder = OneHotEncoder(
        use_cat_names=True, handle_unknown="ignore", cols=columns_to_encode.tolist()
    )
    encoded_sequences = encoder.fit_transform(single_AA_sequences)

    print("removing consensus features")
    features_to_remove = get_features_to_remove(dataset_subtypes)
    total_sequences = encoded_sequences.drop(
        columns=features_to_remove, errors="ignore"
    )

    total_sequences["encoded_label"] = total_sequences["label"].apply(
        {"treated": 1, "naive": 0}.get
    )

    drms = get_all_DRMs()
    total_sequences["hasDRM"] = (
        total_sequences.filter(drms, axis=1).any(axis=1).astype(int)
    )

    total_sequences["is_resistant"] = (
        total_sequences[["encoded_label", "hasDRM"]].any(axis=1).astype(int)
    )

    print("getting resistance scores")
    resistance_scores = get_resistance_scores(resistance_files)

    print("saving dataset to disk")
    joined = total_sequences.join(resistance_scores)
    joined.to_csv(outfile, sep="\t", index=True, header=True)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="""processes data extracted
        from HIVdB for machine learning"""
    )
    parser.add_argument(
        "--naive",
        help="tab delimited file pairwise mutations for naive samples",
        required=True,
    )
    parser.add_argument(
        "--treated",
        help="tab delimited file pairwise mutations for treated samples",
        required=True,
    )
    parser.add_argument(
        "--metadata",
        help="tab delimited file with metadata on sequences",
        required=True,
    )
    parser.add_argument(
        "--resistance",
        help="tab delimited resistance summary file from the HIVDB program",
        nargs=2,
        required=True,
    )
    parser.add_argument(
        "--outfile", help="filepath for the created dataset", required=True
    )
    parser.add_argument(
        "--subtype",
        help="If you want to restrict the data to a certain subtype",
        required=False,
        nargs="?",
        default="All",
        type=str,
    )
    parser.add_argument(
        "--truncate",
        help="""define custom first and last AA positions for training set
        sequences""",
        required=False,
        nargs=2,
        default=[41, 235],
        type=int,
    )
    params = parser.parse_args()

    print(f"starting data processing with parameters:\n{params}")

    process(
        params.naive,
        params.treated,
        params.metadata,
        params.resistance,
        params.outfile,
        params.subtype,
        params.truncate,
    )
