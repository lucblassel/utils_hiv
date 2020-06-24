# coding: utf-8
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from DRM_utils import get_all_DRMs


def get_pvalue(df, target="encoded_label"):
    """
    Get corrected p-values for fisher exact tests on prevalence of 'cleaned' features in df wrt the target feature
    """
    features = df.filter(regex=r"\d+_[^X*nan-]", axis=1).columns.tolist()
    pvals = {}
    for feature in features:
        table = (
            pd.crosstab(df[feature], df[target])
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            .values
        )
        pvals[feature] = fisher_exact(table, alternative="greater")
    pval_df = pd.DataFrame.from_dict(
        pvals, orient="index", columns=["odds_ratio", "p_value"]
    )
    for method in ["Bonferroni", "fdr_bh", "fdr_by"]:
        _, pval_df[method], _, _ = multipletests(
            pval_df["p_value"], method=method, alpha=0.05
        )
    pval_df["is_DRM"] = pval_df.index.map(lambda x: x in drms)
    return pval_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--dataset", "-d", required=True)  # UK dataset
    parser.add_argument("--metadata", "-m", required=True)
    args = parser.parse_args()

    drms = get_all_DRMs()

    uk = pd.read_csv(args.dataset, sep="\t", index_col=0)
    meta = pd.read_csv(args.metadata, sep="\t", index_col=0)

    uk = uk.join(meta["REGAsubtype"])
    to_drop = uk.filter(regex=r"\d+_[*nanX-]", axis=1).columns.tolist()
    uk.drop(to_drop, axis=1, inplace=True)

    pvals = []
    for target in ["encoded_label", "hasDRM", "is_resistant"]:
        for subtype in ["ALL", "B", "C"]:
            for name_drms in ["None", "ALL"]:
                for name_seqs in ["None", "DRM", "NO DRM"]:
                    if subtype != "ALL":
                        data = uk[uk["REGAsubtype"] == subtype].copy()
                    else:
                        data = uk.copy()
                    if name_drms == "ALL":
                        data.drop(drms, axis=1, errors="ignore", inplace=True)
                    if name_seqs == "DRM":
                        data = data[data["hasDRM"] == 0]
                    if name_seqs == "NO DRM":
                        data = data[data["hasDRM"] == 1]
                    pval = get_pvalue(data, target=target)
                    pval["subtype"], pval["DRMs"], pval["seqs"], pval["target"] = (
                        subtype,
                        name_drms,
                        name_seqs,
                        target,
                    )
                    pvals.append(pval)

    pval_all = pd.concat(pvals, axis=0)
    subset = pval_all.loc[
        :, ["Bonferroni", "fdr_bh", "fdr_by", "DRMs", "seqs", "subtype", "target"]
    ]

    subset.to_csv(
        args.output, sep="\t", index=True, header=True,
    )
