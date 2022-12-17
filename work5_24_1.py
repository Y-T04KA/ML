import numpy as np
from scipy.stats import norm


def makeset(values):
    u_n_class = []
    for i in range(len(values)):
        if values[i] not in u_n_class:
            u_n_class += [values[i]]
    return u_n_class


def fit(data):
    u_n_class = []
    dt = []
    dt2 = np.zeros((len(data), len(data[0])))
    for i in range(len(data[0])):
        dt += [[]]
        for j in range(len(data)):
            dt[i] += [data[j][i]]
    for i in range(len(dt)):
        if isinstance(dt[i][0], str):
            u_n_class = makeset(dt[i])
        u_n_class.sort()
        for k in range(len(dt[i])):
            for j in range(len(u_n_class)):
                if dt[i][k] == u_n_class[j]:
                    dt[i][k] = j
    for i in range(len(data)):
        for j in range(len(dt)):
            dt2[i][j] = dt[j][i]
    return dt, dt2


def splitclass(data):
    cl = []
    for i in range(len(data)):
        cl += [data[i][3]]
    u_cl = makeset(cl)
    u_cl.sort()
    clas = []
    for i in range(len(u_cl)):
        clas += [[]]
        for j in range(len(data)):
            if data[j][3] == u_cl[i]:
                clas[i] += [data[j]]
    return clas


def countvalues(data, u_el):
    kol_el = [0] * len(u_el)
    for i in range(len(u_el)):
        for j in range(len(data)):
            if data[j] == u_el[i]:
                kol_el[i] += 1
    return kol_el


def calculateprob(values, values_set):
    p = []
    for i in range(len(values_set)):
        p += [round((values[i] + 1) / (sum(values) + len(values)), 3)]
    return p


def naive(data, dt, dt2):
    clas = splitclass(dt2)
    model_class = []
    kol_priz = len(dt2[0]) - 1
    for i in range(len(clas)):
        prior_prob = [(len(clas[i])) / len(dt2)]
        model_priz = []
        new_clas = []
        for j in range(len(clas[i][0])):
            new_clas += [[]]
            for k in range(len(clas[i])):
                new_clas[j] += [clas[i][k][j]]
        for j in range(kol_priz):
            if isinstance(data[0][j], str):
                values_set = makeset(dt[j])
                values_set.sort()
                values = countvalues(new_clas[j], values_set)
                prob = calculateprob(values, values_set)
                model_priz += [prob]
            else:
                model_priz += [[round(np.mean(new_clas[j]), 2), round(np.sqrt(np.var(new_clas[j])), 2)]]
        model_class += [prior_prob, model_priz]
    return model_class


def test(data, dots):
    new_dots, dots_per = fit(dots)
    zn_naiv = naive(data, dt, dt2)
    for i in range(len(dots)):
        pxc0 = zn_naiv[0][0] * zn_naiv[1][0][0] * zn_naiv[1][1][0] * (
            norm.pdf(dots_per[i][2], zn_naiv[1][2][0], zn_naiv[1][2][1]))
        pxc1 = zn_naiv[2][0] * zn_naiv[3][0][0] * zn_naiv[3][1][0] * (
            norm.pdf(dots_per[i][2], zn_naiv[3][2][0], zn_naiv[3][2][1]))
        classN = pxc0 / (pxc0 + pxc1)
        classY = pxc1 / (pxc0 + pxc1)
        if classN > classY:
            print("Класс точки ", dots[i], "- N")
        else:
            print("Класс точки ", dots[i], "- Y")
        print("Вероятность что точка принадлежит к классу N: ", round(classN, 5))
        print("Вероятность что точка принадлежит к классу Y: ", round(classY, 5))


data = [['F', 'L', 4.8, 'Y'],
        ['F', 'L', 3.9, 'N'],
        ['F', 'S', 0.8, 'N'],
        ['F', 'M', 4.2, 'Y'],
        ['F', 'S', 6.1, 'Y'],
        ['F', 'L', 7.2, 'Y'],
        ['M', 'M', 4.7, 'Y'],
        ['F', 'M', 4.2, 'N'],
        ['M', 'L', 2.1, 'N'],
        ['F', 'M', 0.9, 'N']]

dots = [['M', 'S', 6],
        ['F', 'S', 0.4],
        ['M', 'M', -1]]

dt, dt2 = fit(data)
naive(data, dt, dt2)
test(data, dots)
