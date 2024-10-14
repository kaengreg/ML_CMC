from typing import List

def sum_non_neg_diag(X):
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X.
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    length = min(len(X), len(X[0]))
    ans = 0
    flag = False
    for i in range(length):
        if X[i][i] >= 0:
            flag = True
            ans += X[i][i]
    if flag == False: return -1
    return ans


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    x_len = len(x)
    flag = True
    if x_len != len(y):
        flag = False
    x.sort()
    y.sort()
    for i in range(0, x_len):
        if x[i] != y[i]:
            flag = False
    return flag


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    flag = False
    x_len = len(x)
    for i in range(x_len - 1):
        if (x[i] % 3 == 0) or (x[i + 1] % 3 == 0):
            if not flag:
                max = x[i] * x[i + 1]
                flag = True
            elif x[i] * x[i + 1] > max:
                max = x[i] * x[i + 1]
    if not flag:
        return -1
    return max

def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res = [[0] * len(image) for elem in image]
    for i in range(len(image)):
        elem = image[i]
        for j in range(len(elem)):
            item = elem[j]
            for k in range(len(item)):
                res[i][j] += item[k] * weights[k]

    return res

def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    def refactor_vect(x: List[list[int]]):
        tmp1 = []
        tmp2 = []
        res = []
        for i in range(len(x)):
            tmp1.append(x[i][0])
            tmp2.append(x[i][1])

        for i in range(len(tmp1)):
            for j in range(tmp2[i]):
                res.append(tmp1[i])
        return res

    v_x = refactor_vect(x)
    v_y = refactor_vect(y)
    flag = False

    if (len(v_x) != len(v_y)):
        return -1
    else:
        scalar = []
        for i in range(len(v_x)):
            scalar.append(v_x[i] * v_y[i])
        ans = 0
        for i in range(len(scalar)):
            ans += scalar[i]
    return ans

def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    x_len = len(X)
    x_len_i = len(X[0])
    y_len = len(Y)
    m = [[0] * y_len for i in range(x_len)]
    for i in range(x_len):
        for j in range(y_len):
            n_x = 0
            n_y = 0
            scal = 0
            for k in range(x_len_i):
                n_x += X[i][k] ** 2
                n_y += Y[j][k] ** 2
                scal += X[i][k] * Y[j][k]
            n_x = n_x ** 0.5
            n_y = n_y ** 0.5
            if (n_x == 0 or n_y == 0):
                m[i][j] = 1
            else:
                m[i][j] = scal / (n_x * n_y)
    return m


