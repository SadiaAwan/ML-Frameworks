# Task 2: Eager vs graph execution
# TODO: Write a small function f(x) = x^3 + 2x
# TODO: Implement f(x) in ONE of:
# - PyTorch (eager)
# - TensorFlow with @tf.function (graph)
# - JAX with @jit (graph-like)
# TODO: Print the output and note how execution differs


# PYTORCH



import torch

# Define function f(x) = x^3 +2x
def f(x):
    return x**3 + 2*x

#Create an input tensor

# Create a tenssor with a value
x = torch.tensor(3.0)

print("Input x:", x)

# Call the function(Eager Execution)

# Call the function
y = f(x)

print("Output y:", y)


# TENSORFLOW

import tensorflow as tf

# Define f(x) = x³ + 2x (Graph Mode)

# Define function in graph mode
@tf.function
def f_graph(x):
    return x**3 + 2*x

# Create an input tensor

# Create a TensorFlow tensor
x = tf.constant(3.0)

print("Input x:", x)

# Call the function

# First call (graph is traced + compiled)
y1 = f_graph(x)
print("First output:", y1)

# Second call (already compiled, runs faster)
y2 = f_graph(x)
print("Second output:", y2)





# JAX
