import json
import pandas as pd
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform, randint

ALLOWED_DISTRIBS = {"uniform": uniform, "discrete uniform": randint}


def _read_param_distribs(distrib_path, model_type):
    distrib = json.load(open(distrib_path, "r"))
    return distrib.get(model_type, dict())


def get_parameter_grid(distrib_path, model_type, num_iter):
    params_file = _read_param_distribs(distrib_path, model_type)
    param_distribs = {}

    for param, spec in params_file.items():
        if spec[0]:
            low, high, distrib = spec[1:]
            if distrib not in ALLOWED_DISTRIBS.keys():
                raise ValueError(
                    "Allowed distributions: {}".format(
                        ", ".join(ALLOWED_DISTRIBS.keys())
                    )
                )
            param_distribs[param] = ALLOWED_DISTRIBS[distrib](low, high)
        else:
            param_distribs[param] = spec[1:]

    return (
        list(ParameterSampler(param_distribs, num_iter)) if param_distribs != {} else {}
    )


def join_parameter_grid(param_performance_paths):
    results = []

    for path in param_performance_paths:
        set_results = json.load(open(path, "r"))
        results.append(set_results)

    df = pd.DataFrame(results)

    return df


def get_best_parameters(perf_paths, param_paths, metric):

    results = {}

    best_score = 0
    best_index = 0

    for ind, (perf_path, param_path) in enumerate(zip(perf_paths, param_paths)):
        performance_df = pd.read_json(perf_path)
        score = performance_df.mean()[metric]

        if score > best_score:
            best_score = score
            best_index = ind

        params = json.load(open(param_path, "r"))
        results[ind] = {"params": params, "performance": performance_df}

    return results, best_index
