import sys
import random
import math

import numpy as np

INF = sys.float_info.max

def least_squares(x_values, y_values):
    """
    Solve the overdetermined system a * x + b = y using the least squares method.
    Parameters
    ----------
    x_values : list of float
        The x-values of the data points.
    y_values : list of float
        The y-values of the data points.
    Returns
    -------
    a, b : float
        The parameters of the line that best fits the data points.
    """
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
    # returns parameters for linear model ax+b = (a,b)
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
    # n - minimum number of data points required to estimate model parameters
    # k - maximum number of iterations to estimate model parameters
    # t - threshold where the error is acceptable
    # d - number of data points required to assert that a model fits the data

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
    """
   RANdom SAmple Consensus метод нахождения наилучшей
   аппроксимирующей прямой.

   :param points: Входной массив точек формы [N, 2]
   :param min_inliers: Минимальное количество не-выбросов
   :param max_distance: максимальное расстояние до поддерживающей прямой,
                        чтобы точка считалась не-выбросом
   :param outliers_fraction: Ожидаемая доля выбросов
   :param probability_of_success: желаемая вероятность, что поддерживающая
                                  прямая не основана на точке-выбросе
   :param axis: Набор осей, на которых рисовать график
   :return: Numpy массив формы [N, 2] точек на прямой,
            None, если ответ не найден.
   """

    # Давайте вычислим необходимое количество итераций
    num_trials = int(math.log(1 - probability_of_success) /
                     math.log(1 - outliers_fraction ** 2))

    best_num_inliers = 0
    best_support = None
    for _ in range(num_trials):
        # В каждой итерации случайным образом выбираем две точки
        # из входного массива и называем их "суппорт"
        random_indices = np.random.choice(
            np.arange(0, len(points)), size=(2,), replace=False)
        assert random_indices[0] != random_indices[1]
        support = np.take(points, random_indices, axis=0)

        # Здесь мы считаем расстояния от всех точек до прямой
        # заданной суппортом. Для расчета расстояний от точки до
        # прямой подходит функция векторного произведения.
        # Особенность np.cross в том, что функция возвращает только
        # z координату векторного произведения, а она-то нам и нужна.
        cross_prod = np.cross(support[1, :] - support[0, :],
                              support[1, :] - points)
        support_length = np.linalg.norm(support[1, :] - support[0, :])
        # cross_prod содержит знаковое расстояние, поэтому нам нужно
        # взять модуль значений.
        distances = np.abs(cross_prod) / support_length

        # Не-выбросы - это все точки, которые ближе, чем max_distance
        # к нашей прямой-кандидату.
        num_inliers = np.sum(distances < max_distance)
        # Здесь мы обновляем лучший найденный суппорт
        if num_inliers >= min_inliers and num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_support = support

    # Если мы успешно нашли хотя бы один суппорт,
    # удовлетворяющий всем требованиям
    if best_support is not None:
        # Спроецируем точки из входного массива на найденную прямую
        support_start = best_support[0]
        support_vec = best_support[1] - best_support[0]
        # Для расчета проекций отлично подходит функция
        # скалярного произведения.
        offsets = np.dot(support_vec, (points - support_start).T)
        proj_vectors = np.outer(support_vec, offsets).T
        support_sq_len = np.inner(support_vec, support_vec)
        projected_vectors = proj_vectors / support_sq_len
        projected_points = support_start + projected_vectors

    return [list(projected_points[0].astype('int')),list(projected_points[-1].astype('int'))]

# ----------------------------------------------

'''real_parameters = (15.2723, 3.32)

N = 100
data = [ (x, linear_model(real_parameters, x) + random.randint(-1,1) * (1 + random.random()) * random.randint(100,200) )  for x in range(N) ]
print(data)
answer = ransac(data, 8, 20, 100, 10)
answer2 = ransac2(data)
print(f"{answer=}")
print(f"{answer2}")
print(type(answer2))
x2, y2 = [answer2[0][0], answer2[1][0]], [answer2[0][1], answer2[1][1]]
print(x2, y2)
print(answer[1])
fitted_paramters = answer[0]

from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0, N, N)
#print(x)
y = x*real_parameters[0] + real_parameters[1]
#print(y)

if fitted_paramters:
    yhat = x*fitted_paramters[0] + fitted_paramters[1]
px, py = zip(*data)
plt.figure(0)
plt.plot(x, y, color='r')
if fitted_paramters:
    plt.plot(x, yhat, color='b')
plt.plot(x2, y2, color='g')
plt.scatter(px, py, color='gray')
plt.show()'''
