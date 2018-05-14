import numpy as np
origin = np.zeros((10, 10))
A_1 = np.ones((10, 10))
A_2 = np.ones((10, 10))
A = origin.copy()
B = origin.copy()

# 渐入渐出融合
num = 5
for i in range(num+1):
    A_1[num - i, :] = (num - i) * 1 / 5
    A_2[:, num - i] = (num - i) * 1 / 5
# for i in range(num):
#     for j in range(num):
#         A_1[num - i, num + j] = 1
print(A_1)
print(A_2)
A = A_1 * A_2
# A = A_1 + A_2
# for i in range(num):
#     for j in range(num):
#         A[i, j] = A_1[i, j] * A_2[i, j]
print("乘法")
print("A")
print(A)
print("1-A")
print(1-A)

print("加法")

A = A_1 + A_2
for i in range(num):
    for j in range(num):
        A[i, j] = A[i, j] / 2
print("A")
print(A)
print("1-A")
print(1-A)