import numpy as np
from typing import Tuple

def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    diag_els = np.diag(X)
    if len(diag_els[diag_els >= 0]) == 0:
        return -1
    else:
        return diag_els.sum()

def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    if np.shape(x) != np.shape(y):
        return False
    x_nums, x_counts = np.unique(x, return_counts = True)
    y_nums, y_counts = np.unique(y, return_counts = True)
    if (np.any(x_nums != y_nums)) or (np.any(x_counts != y_counts)):
        return False
    return True



def max_prod_mod_3(x):
    """
    Вернуть максимальное прозведение соседних элементов в массиве x,
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    if (len(x) == 1):
        return -1
    tmp = np.roll(x, 1) * x
    ans = tmp[tmp % 3 == 0]
    if (len(ans) == 0):
        return -1
    return np.max(ans)



def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return np.sum(image * weights, axis = 2)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1. 3
    """
    v_x = np.repeat(x[:,0], x[:,1])
    v_y = np.repeat(y[:,0], y[:,1])
    if (len(v_x) != len(v_y)):
        return -1
    return (v_x * v_y).sum()


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    x_n = np.linalg.norm(X, axis = 1)
    y_n = np.linalg.norm(Y, axis = 1)

    scal = np.dot(X, Y.T)
    nul_x = np.where(x_n == 0)
    nul_y = np.where(y_n == 0)
    scal[nul_x] = 1
    scal[nul_y] = 1
    x_n[nul_x] = 1
    y_n[nul_y] = 1
    n_mult = (x_n[:, np.newaxis] * y_n.T)
    return np.divide(scal, n_mult)
