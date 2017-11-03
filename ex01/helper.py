from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()

print(digits.keys())

data = digits["data"]
images = digits["images"]
target = digits["target"]
target_names = digits["target_names"]

print(data.dtype)
print(data.shape)

img_number = 5
print(target_names[img_number])
img_shape = [8, 8]

img = images[img_number]  # np.reshape(data[img_number],img_shape)

# test dimensionality
assert 2 == np.size(np.shape(img))

plt.figure()
plt.gray()
plt.imshow(img, interpolation="nearest")
# plt.show()

X_all = data
y_all = target

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(digits.data, digits.target, test_size = 0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.4, random_state=0)

print(X_train.shape)
print()

# that's in incredibly loopy function
def dist_loop(training_set, test_set):
    # train_shape = training
    dist = np.zeros((training_set.shape[0], test_set.shape[0]))
    for i in range(training_set.shape[0]):
        for j in range(test_set.shape[0]):
            temp_sum = np.sqrt(np.sum(np.square(test_set[j,k]-training_set[i,k]) for k in range(training_set.shape[1])))
            dist[i,j] = temp_sum


    return dist


distMat = dist_loop(X_train, X_test)
print('ttt')