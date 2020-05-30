import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    array_to_add = []
    for i in range(len(Y)): 
        array_to_add.append(-(Y[i] * np.log(P[i]) + (1 - Y[i])*(np.log(1-P[i])))) # Remember it is the NEGATIVE summation!!
        
    result = sum(array_to_add)
    return result # changes the array to a float


''' Their solution:
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    array_to_add = []
    for i in range(len(Y)): 
        array_to_add.append(-(Y[i] * np.log(P[i]) + (1 - Y[i])*(np.log(1-P[i])))) # Remember it is the NEGATIVE summation!!
        
    result = sum(array_to_add)
    return result # changes the array to a float
'''