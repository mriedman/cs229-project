import numpy as np

val_data = np.loadtxt('./val_data.txt')
clusters = np.loadtxt('./clusters_ss.txt')

norms = np.concatenate([np.linalg.norm(val_data[:, :-1] - cl, axis=1).reshape((-1,1)) for cl in clusters], axis=1)
mins = np.argmin(norms, axis=1)
print(np.sum(mins == val_data[:, -1]) / val_data.shape[0])
print(np.sum(np.abs(mins - np.array(list(map(lambda x: 0 if x == 0 else 1, val_data[:, -1]))))) / val_data.shape[0])

a = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        a[i][j] = np.sum((mins == j) * (val_data[:, -1] == i))

print(a)
print(a / np.sum(a, axis=1).reshape((-1,1)))
