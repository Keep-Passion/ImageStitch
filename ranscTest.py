import skimage.measure
import numpy as np

# t = np.linspace(0, 2 * np.pi, 50)
# xc, yc = 20, 30
# a, b = 5, 10
# x = xc + a * np.cos(t)
# y = yc + b * np.sin(t)
#
# data = np.column_stack([x, y])
# print(data)
# np.random.seed(seed=1234)
# data += np.random.normal(size=data.shape)
#
# data[0] = (100, 100)
# data[1] = (110, 120)
# data[2] = (120, 130)
# data[3] = (140, 130)

data = np.array([[-8.59848633e+01,  -5.83740234e-01],
                [-5.82129974e+01, -3.07275879e+02],
                [-5.46425629e+01, -1.41938477e+01],
                [-8.59848633e+01, -5.83740234e-01],
                [-5.82129974e+01, -3.07275879e+02],
                [-5.46425629e+01, -1.41938477e+01]])
print(data)
model = skimage.measure.LineModelND
print(model.estimate(model, data))
#
print(np.round(model.params))

ransac_model, inliers = skimage.measure.ransac(data, model, 4, 1.0, max_trials=50)

print(abs(np.round(ransac_model.params)))
print(inliers)
print(sum(inliers))
