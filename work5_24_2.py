import numpy as np
import matplotlib.pyplot as plt


def LDA(data, n_class):
    u_n_class = []
    for i in range(len(n_class)):
        if n_class[i] not in u_n_class:
            u_n_class += [n_class[i]]
    u_n_class.sort()
    lis = []
    ind = []
    for i in range(len(u_n_class)):
        lis += [[]]
        ind += [[]]
    for i in range(len(u_n_class)):
        for j in range(len(data)):
            if n_class[j] == u_n_class[i]:
                lis[i] += [data[j]]
                ind[i] += [j]
    nu = []
    for i in range(len(u_n_class)):
        sum_x, sum_y = 0, 0
        for j in range(len(lis[i])):
            sum_x += lis[i][j][0] / len(lis[i])
            sum_y += lis[i][j][1] / len(lis[i])
        nu += [[round(sum_x, 2), round(sum_y, 2)]]
    k = [nu[1][0] - nu[0][0], nu[1][1] - nu[0][1]]
    b = [[k[0] * k[0], k[0] * k[1]], [k[1] * k[0], k[1] * k[1]]]
    for i in range(len(u_n_class)):
        for j in range(len(lis[i])):
            lis[i][j] = [round(lis[i][j][0] - nu[i][0], 2), round(lis[i][j][1] - nu[i][1], 2)]
    s = []
    for i in range(len(u_n_class)):
        k1, k2, k3 = 0, 0, 0
        for j in range(len(lis[i])):
            k1 += lis[i][j][0] * lis[i][j][0]
            k2 += lis[i][j][0] * lis[i][j][1]
            k3 += lis[i][j][1] * lis[i][j][1]
        s += [k1, k2, k2, k3]
    s = [[s[0] + s[4], s[1] + s[5]], [s[2] + s[6], s[3] + s[7]]]
    s = np.linalg.inv(s)
    a = [[s[0][0] * b[0][0] + s[0][1] * b[1][0], s[0][0] * b[0][1] + s[0][1] * b[1][1]],
         [s[1][0] * b[0][0] + s[1][1] * b[1][0], s[1][0] * b[0][1] + s[1][1] * b[1][1]]]
    sch, sv = np.linalg.eig(a)
    if sch[0] > sch[1]:
        index = 0
    else:
        index = 1
    w = [sv[0][index], sv[1][index]]
    s = (np.sqrt(k[0] ** 2 + k[1] ** 2) * w[0] * 0.5 + np.sqrt(k[0] ** 2 + k[1] ** 2) * w[1] * 0.5)
    return lis, w, k, s, nu, ind, u_n_class


X = [[1.9, 4],
     [2.2, 3.3],
     [2, 4.1],
     [2.5, 12.6],
     [2.3, 2.2],
     [2.1, 12.6],
     [1.8, 2.8],
     [3, 13.8],
     [3.2, 12.1],
     [3.6, 12.5]]
y = [1, 1, 1, -1, 1, -1, 1, -1, -1, -1]

lis, w, k, s, nu, ind, u_n_class = LDA(X, y)
test = [-1.4, 18.4]
c_1 = w[0] * nu[0][0] + w[1] * nu[0][1]
c1 = w[0] * nu[1][0] + w[1] * nu[1][1]
pc = w[0] * test[0] + w[1] * test[1]

if abs(pc - c_1) < abs(pc - c1):
    print("Точка относится к классу -1")
    ind[0] += [10]
else:
    print("Точка относится к классу +1")
    ind[1] += [10]

X = X + [test] + [nu[0]] + [nu[1]]
color = ['r', 'b']
fig, ax = plt.subplots(figsize=(9, 7), dpi=80)
for i in range(len(ind)):
    for j in range(len(ind[i])):
        ax.scatter(X[ind[i][j]][0], X[ind[i][j]][1], c=color[i])
        ax.text(X[ind[i][j]][0] + 0.3, X[ind[i][j]][1], ind[i][j])
    ax.plot(nu[i][0], nu[i][1], marker="*")
    ax.text(nu[i][0] + 0.2, nu[i][1] - 0.2, 'Center %i' % u_n_class[i])
ax.arrow(2.5, 0.57, *w, length_includes_head=True, head_width=0.2,
         label="w", color="orange")
points = [*[np.dot(w, d) for d in X]]
min_projection = 10000
max_projection = 0
for i in range(len(points)):
    if points[i] > max_projection:
        max_projection = points[i]
    if points[i] < min_projection:
        min_projection = points[i]
min_point = [w[0] * min_projection, w[1] * min_projection]
max_point = [w[0] * max_projection, w[1] * max_projection]
ax.plot(*zip(min_point, max_point), 'pink', label="k")
for projection, p in zip(points, X):
    d = [w[0] * projection, w[1] * projection]
    ax.plot(*zip(d, p), ls="--", c="#b1b3b1")

ax.scatter(*np.dot(w, s), marker="^", label="Class separator", s=50, c="green")
plt.show()
