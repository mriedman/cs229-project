import numpy as np
from decision_funcs import em_predict

val_data = np.loadtxt('./val_data.txt')

ct, ct2, tot = 0, 0, 0
twos = lambda x: 0 if x == 0 else 1

a = np.zeros((3,3))

for row in val_data:
    tot += 1
    # print(row)
    pred = em_predict(row[:-1])
    if pred == row[-1]:
        ct += 1
    if twos(pred) == twos(row[-1]):
        ct2 += 1
    a[int(row[-1])][pred] += 1

print(ct/tot)
print(ct2/tot)

print(a)
print(a / np.sum(a, axis=1).reshape((-1,1)))
