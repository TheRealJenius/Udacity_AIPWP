import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    all_exp = np.exp(L) # exp = exponent ;  remember that with numpy we can broadcast e.g. using .add or .divide etc.
    total_exp = sum(all_exp) # we need to add them together to obtain the bottom denominator
    sm_function_result = []
    for i in range(len(L)): # Iterating through the lenght of the array of numbers
        sm_function_result.append(np.exp(L[i])/total_exp) # remember we can't directly assign to an empty list, we need to append the result at the end
    return sm_function_result # return the result so it can be used in another function

''' Their solution

import numpy as np

def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result
    
    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())
'''