import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

RANDOM_SEED = 42 				#Change for different random start
np.random.seed(RANDOM_SEED)

# sigmoid and sigmoid prime implementaion
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1-sigmoid(x))

# TEST SGIMOID AND SIGMOID PRIME 
"""
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 12, 6
x = np.linspace(-10., 10., num=100)
sig = sigmoid(x)
sig_prime = sigmoid_prime(x)

plt.plot(x, sig, label="sigmoid")
plt.plot(x, sig_prime, label="sigmoid prime")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(prop={'size' : 16})
plt.show()
"""

################################
### INITIALIZATION PHASE ###
epochs = 50000
input_size, hidden_size, output_size = 2, 3, 1
LR = 0.1 # learning rate
X = np.array([[0,0], [0,1], [1,0], [1,1]])							# INPUTS (4,2)
y = np.array([ [0],   [1],   [1],   [0]])							# CORRESPONDING OUTPUTS 
w_hidden = np.random.uniform(size=(input_size, hidden_size))		# matrix initiate random(2,3)
w_output = np.random.uniform(size=(hidden_size, output_size))		# matrix initiate random(3,1)


### TRAINING PHASE ###
for epoch in range(epochs):
 
    # Forward
    act_hidden = sigmoid(np.dot(X, w_hidden))			# multiply matrices (4,2)*(2,3) then apply sigmoid =(4,3)
    output = np.dot(act_hidden, w_output)				# multiply matrices (4,3)*(3,1) then apply sigmoid =(4,1)
    
    # Calculate error
    error = y - output
    
    if epoch % 5000 == 0:
    	print(f'error sum {sum(error)}')

    # Backward
    dZ = error * LR
    w_output += act_hidden.T.dot(dZ)								# traspose matrix and mul then update op weights
    dH = dZ.dot(w_output.T) * sigmoid_prime(act_hidden)
    w_hidden += X.T.dot(dH)


### TESTING PHASES ###
for X_test in X:
	# [0, 0] ,[0, 1] ,[1,0] ,[1,1]
	act_hidden = sigmoid(np.dot(X_test, w_hidden))
	print(np.dot(act_hidden, w_output))



# NUMPY .DOT DOCUMENTATION
# This function returns the dot product of two arrays. For 2-D vectors, it is the equivalent to matrix multiplication. For 1-D arrays, it is the inner product of the vectors.
# For Ndimensional arrays, it is a sum product over the last axis of a and the second-last axis of b.