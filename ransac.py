import sys
import random
import math

import numpy as np

INF = sys.float_info.max

def least_squares(x_values, y_values):
    n = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_x_squared = sum(x**2 for x in x_values)
    sum_xy = sum(x*y for x, y in zip(x_values, y_values))

    denominator = n * sum_x_squared - sum_x**2
    a = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y * sum_x_squared - sum_x * sum_xy) / denominator

    return a, b


def linear_fit(data):
    A, B = zip(*data)
    solution = least_squares(A, B)
    a, b = solution
    return a, b

def linear_model(parameters, x):
    a, b = parameters
    return a * x + b


def sqrt_err(a, b):
    err = b - a
    err = err * err
    err = math.sqrt(err)
    return err

def ransac(data, n, k, t, d):
    best_err = INF
    best_fit = None
    for i in range(k):
        print(f"{i}/{k}", end='\r')
        maybe_inliers = random.sample(data, n)
        maybe_model = linear_fit(maybe_inliers)
        also_inliers = []
        for (x,y) in [point for point in data if point not in maybe_inliers]:
            y_hat = linear_model(maybe_model, x)
            err = sqrt_err(y, y_hat)
            if err < t:
                also_inliers.append((x,y))

        if len(also_inliers) > d:
            better_model = maybe_model
            iteration_err = sum([ sqrt_err(linear_model(better_model, x), y)  for (x,y) in data ])

            if iteration_err < best_err:
                best_fit = better_model
                best_err = iteration_err

    return best_fit, best_err
def ransac3(points: np.ndarray,
           min_inliers: int = 4,
           max_distance: float = 0.15,
           outliers_fraction: float = 0.5,
           probability_of_success: float = 0.99):
    num_trials = int(math.log(1 - probability_of_success) /
                     math.log(1 - outliers_fraction ** 2))

    best_num_inliers = 0
    best_support = None
    for _ in range(num_trials):
        random_indices = np.random.choice(
            np.arange(0, len(points)), size=(2,), replace=False)
        assert random_indices[0] != random_indices[1]
        support = np.take(points, random_indices, axis=0)
        cross_prod = np.cross(support[1, :] - support[0, :],
                              support[1, :] - points)
        support_length = np.linalg.norm(support[1, :] - support[0, :])
        distances = np.abs(cross_prod) / support_length
        num_inliers = np.sum(distances < max_distance)
        if num_inliers >= min_inliers and num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_support = support
    if best_support is not None:
        support_start = best_support[0]
        support_vec = best_support[1] - best_support[0]
        offsets = np.dot(support_vec, (points - support_start).T)
        proj_vectors = np.outer(support_vec, offsets).T
        support_sq_len = np.inner(support_vec, support_vec)
        projected_vectors = proj_vectors / support_sq_len
        projected_points = support_start + projected_vectors

    return [list(projected_points[0].astype('int')),list(projected_points[-1].astype('int'))]