# Task 1: Vector and matrix basics (NumPy)
# TODO: Create two vectors (length 3) and compute:
# - dot product
# - L2 norm
# - cosine similarity
# TODO: Create a 2x3 matrix and multiply it by a length-3 vector


import numpy as np

##-----Create 2 Vectors---------
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])


# Print them
print("vector 1:", v1)
print("vector 2:", v2)


# Compute dot product
dot_product = np.dot(v1, v2)

print("Dot produkt:",dot_product)


# Compute L2 norm of v1
norm_vl = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)

# print ("L1 norm of v1:", norm_v1)

# Compute cosine similarity
cosine_similarity = dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2) )

print("Cosine similarity:", cosine_similarity)

# Create a 2*3 matrix
M = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Matrix:\n", M)
print("Matrix shape:", M.shape)

# Multiply Matrix by length -3 vector


# Matrix shape: (2,3)
# Vector shape: (3,)

# Result will be (2,).



# Multiply matrix by vector v1

result = np.dot(M, v1)

print("Matrix x vector result:", result)




