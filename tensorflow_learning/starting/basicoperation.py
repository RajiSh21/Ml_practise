from tensorflow import constant, add, ones_like, reduce_sum

#0-dimensional tensors
A0 = constant([1])
B0 = constant([2])  

#1-dimensional tensors
A1 = constant([1, 2])
B1 = constant([3, 4])  

#1-dimensional tensors
A2 = constant([1, 2], [3, 4])
B2 = constant([5, 6], [7, 8]) 

#perfoming tensor addtion
c0 = add(A0, B0)
c1 = add(A1, B1)
c2 = add(A2, B2)

#performing tensor multiply
d0 = A0 * B0
d1 = A1 * B1
d2 = A2 * B2

#summing over tensor
B = reduce_sum(A2)

B0 = reduce_sum(A2, 0)
B1 = reduce_sum(A2, 1)
B2 = reduce_sum(A2, 0)


# Define tensors A1 and A23 as constants
A1 = constant([1, 2, 3, 4])
A23 = constant([[1, 2, 3], [1, 6, 4]])

# Define B1 and B23 to have the correct shape
B1 = ones_like(A1)
B23 = ones_like(A23)

# Perform element-wise multiplication
C1 = A1 * B1
C23 = A23 * B23

# Print the tensors C1 and C23
print('\n C1: {}'.format(C1.numpy()))
print('\n C23: {}'.format(C23.numpy()))