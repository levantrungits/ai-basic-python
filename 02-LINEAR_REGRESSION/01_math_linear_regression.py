import numpy as np
import matplotlib.pyplot as plt

# random data
A = [2,5,7,9,11,16,19,23,22,29,29,35,37,40,46]
b = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# visualize data
plt.plot(A, b, 'ro')

# change Row vector to Column vector
A = np.array([A]).T # A transpose
b = np.array([b]).T # b transpose

# create vector 1
ones = np.ones((A.shape[0], 1), dtype=np.int)

# combine 1 & A
# concat 2 Vectors to 1 Matrix
A = np.concatenate((A, ones), axis=1)
print(A)

# Use Fomular
# math y = ax + b => x = (A-transpose.A)(^-1).(A-transpose).b
#   inv ~ inverse ~ ^-1
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
print(x) # x = [a b]

# test data to draw
x0 = np.array([[1, 46]]).T  # 1 First to Max A
y0 = x0*x[0][0] + x[1][0] # y = ax + b

# draw x0,y0 ~ line y = ax + b
plt.plot(x0, y0)

# test Predicting data
x_test = 12
y_test = x_test * x[0][0] + x[1][0]
print(y_test)

# show image
plt.show()