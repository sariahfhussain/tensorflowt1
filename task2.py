%tensorflow_version 2.x
import tensorflow as tf

# 1. Create a TensorFlow Constant
constant_tensor = tf.constant([1, 2, 3])
print("1. Constant Tensor:", constant_tensor.numpy())

# 2. Perform Element-Wise Operation
a = tf.constant([2, 4, 6])
b = tf.constant([1, 2, 3])
elementwise_product = tf.multiply(a, b)
print("2. Element-Wise Multiplication:", elementwise_product.numpy())

# 3. Reshape a Tensor
x = tf.constant([1, 2, 3, 4, 5, 6])
reshaped_tensor = tf.reshape(x, (2, 3))
print("3. Reshaped Tensor:")
print(reshaped_tensor.numpy())

# 4. Extract a Slice
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
middle_column = matrix[:, 1]
print("4. Middle Column:")
print(middle_column.numpy())

# 5. Variable Initialization
weights = tf.Variable(tf.random.normal((3, 3)))
print("5. Initialized Weights:")
print(weights.numpy())

# 6. Math Operations
tensor_to_sqrt = tf.constant([4.0, 9.0, 16.0])
sqrt_result = tf.sqrt(tensor_to_sqrt)
print("6. Element-Wise Square Root:")
print(sqrt_result.numpy())
