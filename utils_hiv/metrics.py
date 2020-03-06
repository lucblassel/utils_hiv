import numpy as np
import pandas as pd


class Bezier:
    def __init__(self, threshold=0.2, definition=1000, method='polynomial',
                 order=3, to1=True, mult=0):
        self.df = self.get_bezier(
            threshold, definition, to1, mult).set_index('x')['y']
        self.order = order
        self.method = method

    def get_value(self, x):
        if x in self.df.index:
            return self.df.loc[x]
        else:
            self.df[x] = np.nan
            self.df = self.df.sort_index(axis=0).interpolate(
                axis=0, method=self.method, order=self.order)
            return self.df[x]

    def calc_bezier(self, t, p0, p1, p2, p3):  # cubic bezier curve
        coords = {'x': 0, 'y': 0}
        coefs = [
            (1 - t) ** 3,
            3 * ((1 - t) ** 2) * t,
            3 * (1 - t) * (t ** 2),
            t ** 3
        ]  # binomial expansion for degree 3

        for coord in coords.keys():
            coords[coord] =\
                coefs[0] * p0[coord] +\
                coefs[1] * p1[coord] +\
                coefs[2] * p2[coord] +\
                coefs[3] * p3[coord]

        return coords

    def get_bezier(self, a=0.2, definition=100, to1=True, mult=0):
        res = {'x': [], 'y': []}
        x0, x1, x2, x3 = 0, a * (1 + mult / 100), a * (1 - mult / 100), 1
        y0, y1, y2, y3 = 0, 0, _get_line_value(
            x2, a, to1), _get_line_value(1, a, to1)
        p0, p1, p2, p3 = _point(x0, y0), _point(
            x1, y1), _point(x2, y2), _point(x3, y3)

        for t in np.linspace(0, 1, definition):
            coords = self.calc_bezier(t, p0, p1, p2, p3)
            for k, v in coords.items():
                res[k].append(v)

        return pd.DataFrame(res)


def solve_bezier(x, a=0.2, to1=True, mult=0):

    x0, x1, x2, x3 = 0, a * (1 + mult / 100), a * (1 - mult / 100), 1
    y0, y1, y2, y3 = 0, 0, _get_line_value(
        x2, a, to1), _get_line_value(1, a, to1)

    polynomial_coefs = [
        -1 * x0 + 3 * x1 - 3 * x2 + x3,  # t^3
        3 * x0 - 6 * x1 + 3 * x2,       # t^2
        -3 * x0 + 3 * x1,                # t^1
        1 * x0 - x                      # t^0
    ]

    roots = np.roots(polynomial_coefs)
    t = float(roots[np.isreal(roots)][0])

    coefs_bez = [
        (1 - t) ** 3,
        3 * ((1 - t) ** 2) * t,
        3 * (1 - t) * (t ** 2),
        t ** 3
    ]
    y = coefs_bez[0] * y0 +\
        coefs_bez[1] * y1 +\
        coefs_bez[2] * y2 +\
        coefs_bez[3] * y3

    return y
#                                       #
# General usage function in all metrics #
#                                       #


def _get_status(real_values, pred_values):
    """
    returns TP, FP, TN, FN
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for real, pred in zip(real_values, pred_values):
        if real == pred:
            if real == 0:
                tn += 1
            else:
                tp += 1
        else:
            if real == 0:
                fp += 1
            else:
                fn += 1
    return tp, fp, tn, fn


#                #
# Base functions #
#                #
def _sharp_slope(fp, total_n, threshold=0.2):
    if fp <= total_n * threshold:
        return fp / (total_n * threshold)
    else:
        return 1


def _hinge(threshold, x):
    return 0 if x <= threshold else x - threshold


def _sigmoid(x, x0, k=0.01):
    return 1 / (1 + np.exp(-k * (x - x0)))


def _tunable_sigmoid(fp, total_n, threshold=0.2, K=1, A=0, C=1, Q=1, B_mult=70,
                     mu=1):
    M = (total_n * threshold) / 2
    B = B_mult / total_n

    return A + (K - A) / (C + Q * np.exp(-B * (fp - M))) ** mu


def _point(x, y):
    return {'x': x, 'y': y}


def _get_line_value(x, threshold, to1=True):
    denom = (1 - threshold) if to1 else 1
    return (x - threshold) / denom


def _get_hinge_value(x, threshold, to1=True):
    return max(0, _get_line_value(x, threshold, to1))

#                  #
# Metric functions #
#                  #


def weighted_accuracy(real_values, pred_values, factor=5, weight=None):
    if weight is None:
        weight = factor / (factor + 1)

    tp, fp, tn, fn = _get_status(real_values, pred_values)

    return 1 - ((1 - weight) * fp / (fp + tn) + weight * fn / (fn + tp))


def sigmoid_metric(real_values, pred_values, mid=0.20, k_mult=5e-7, w_fn=0.5):
    tp, fp, tn, fn = _get_status(real_values, pred_values)

    x0 = (fp + tn) * mid
    k = (fp + tn) * k_mult

    return 1 - ((1 - w_fn) * _sigmoid(fp, x0, k) + w_fn * fn / (fn + tp))


def new_assymetric_metric(real_values, pred_values, phi_name):
    tp, fp, tn, fn = _get_status(real_values, pred_values)
    phi = {'sharp': _sharp_slope, 'sigmoid': _tunable_sigmoid}[phi_name]

    return 1 - (fn + phi(fp, fp + tn) * fp) / (tp + fp + fn + tn)


def new_sigmoid_metric(real_values, pred_values):
    return new_assymetric_metric(real_values, pred_values, 'sigmoid')


def new_sharp_metric(real_values, pred_values):
    return new_assymetric_metric(real_values, pred_values, 'sharp')


def hinge_metric(real_values, pred_values, threshold=0.2):
    tp, fp, tn, fn = _get_status(real_values, pred_values)
    total = tp + fp + tn + fn
    return 1 - (_hinge(threshold, fp / total) + fn / total)


def assymetric_bezier(real_values, pred_values, threshold=0.2, to1=True,
                      **kwargs):
    bezier = Bezier(threshold=threshold, to1=to1)
    tp, fp, tn, fn = _get_status(real_values, pred_values)
    total = tp + fp + tn + fn
    return 1 - (bezier.get_value(fp / total) + fn / total)


def assymetric_bezier_solve(real_values, pred_values, threshold=0.2, to1=True,
                            **kwargs):
    tp, fp, tn, fn = _get_status(real_values, pred_values)
    total = tp + fp + tn + fn
    return 1 - (solve_bezier(fp / total) + fn / total)


def assymetric_sharp(real_values, pred_values, threshold=0.2, to1=True,
                     **kwargs):
    tp, fp, tn, fn = _get_status(real_values, pred_values)
    total = tp + fp + tn + fn
    return 1 - (_get_hinge_value(fp / total, threshold, to1) + fn / total)
