%tensorflow_version 2.x
import tensorflow as tf

# 1. a TensorFlow constant tensor is created with the values [1, 2, 3].
# The numpy() method is used to extract and print the values.
constant_tensor = tf.constant([1, 2, 3])
print("1. Constant Tensor:", constant_tensor.numpy())

# 2. Two constant tensors a and b are created, and element-wise multiplication 
# is performed using tf.multiply(). The result is printed.
a = tf.constant([2, 4, 6])
b = tf.constant([1, 2, 3])
elementwise_product = tf.multiply(a, b)
print("2. Element-Wise Multiplication:", elementwise_product.numpy())

# 3. The tensor x is reshaped into a 2x3 matrix using tf.reshape(), and the reshaped tensor is printed.
x = tf.constant([1, 2, 3, 4, 5, 6])
reshaped_tensor = tf.reshape(x, (2, 3))
print("3. Reshaped Tensor:")
print(reshaped_tensor.numpy())

# 4. A matrix is created, and a slice of the middle column is extracted using matrix[:, 1]. The result is printed.
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
middle_column = matrix[:, 1]
print("4. Middle Column:")
print(middle_column.numpy())

# 5. A variable weights is created with random normal values of shape (3, 3) using tf.random.normal(). The initialized weights are printed.
weights = tf.Variable(tf.random.normal((3, 3)))
print("5. Initialized Weights:")
print(weights.numpy())

# 6. A tensor is created, and element-wise square root is calculated using tf.sqrt(). The result is printed.
tensor_to_sqrt = tf.constant([4.0, 9.0, 16.0])
sqrt_result = tf.sqrt(tensor_to_sqrt)
print("6. Element-Wise Square Root:")
print(sqrt_result.numpy())
