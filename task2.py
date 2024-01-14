'''
    program to demonstrates basic TensorFlow operations such as creating tensors,
    performing element-wise operations, reshaping tensors, extracting slices,
    initializing variables, and performing mathematical operations on tensors'''


# Import the TensorFlow library
import tensorflow as tf


#1. Create a TensorFlow Constant:
''' 
    Here, we create a constant tensor using tf.constant(). 
    Constants are tensors with fixed values that cannot be changed.
    In this case, we have a constant tensor with values [1, 2, 3].
'''
constant_tensor = tf.constant([1, 2, 3])
print("1. Constant Tensor:", constant_tensor.numpy())
''' The '.numpy()' method is used to convert the TensorFlow tensor to a NumPy array
    so that we can easily print and inspect the values. '''



# 2. Perform Element-Wise Operation:
'''    
    Now, we create two constant tensors 'a' and 'b' with values [2, 4, 6] and [1, 2, 3].
    We perform element-wise multiplication using the tf.multiply() function.
    Element-wise operations operate on corresponding elements of tensors.
    '''
a = tf.constant([2, 4, 6])
b = tf.constant([1, 2, 3])
elementwise_product = tf.multiply(a, b)
print("2. Element-Wise Multiplication:", elementwise_product.numpy())



# 3. Reshape a Tensor:
''' 
    Here, we have a tensor 'x' with values [1, 2, 3, 4, 5, 6].
    We reshape it into a 2x3 matrix using tf.reshape().
    Reshaping is a way to change the arrangement of elements in a tensor.
   '''
x = tf.constant([1, 2, 3, 4, 5, 6])
reshaped_tensor = tf.reshape(x, (2, 3))
reshaped_tensor = tf.reshape(x, (2, 3))
print("3. Reshaped Tensor:")
print(reshaped_tensor.numpy(), "\t")

#  We get a 2x3 matrix, and we will print it to check.



# 4. Extract a Slice:
'''  
    In this section, we have a matrix with values [[1, 2, 3], [4, 5, 6], [7, 8, 9]].
    We extract the middle column using slicing, specifically matrix[:, 1].
    The ':' means we take all rows, and '1' refers to the second column.
'''
matrix = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
middle_column = matrix[:, 1]
middle_column = matrix[:, 1]
print("4. Middle Column:")
print(middle_column.numpy())

#    Now we have a one-dimensional tensor containing the middle column values.



# 5. Variable Initialization:
'''
    Here, we introduce a variable 'weights' using tf.Variable().
    Variables are mutable tensors used to hold and update parameters in a model.
    We initialize 'weights' with random normal values of shape (3, 3).
'''
weights = tf.Variable(tf.random.normal((3, 3)))
print("5. Initialized Weights:")
print(weights.numpy())

#  We print the initialized weights for examination.



# 6. Math Operations:
'''
    In this final part, we have a tensor 'tensor_to_sqrt' with values [4.0, 9.0, 16.0].
    We calculate the element-wise square root using tf.sqrt().
'''
tensor_to_sqrt = tf.constant([4.0, 9.0, 16.0])
sqrt_result = tf.sqrt(tensor_to_sqrt)
print("6. Element-Wise Square Root:")
print(sqrt_result.numpy())
#    The result is a tensor with the square root of each element in 'tensor_to_sqrt'.
