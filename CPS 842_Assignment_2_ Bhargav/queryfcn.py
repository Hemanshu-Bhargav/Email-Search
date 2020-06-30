import os
import math
import sys
import tarfile as tar
import time
import numpy as np

lst = ['who','hello','is','hi']
arr = np.zeros(len(lst))
inp = input("query: ")
lstinpt = inp.split()
for item in lstinpt:
    i = lst.index(item)
    arr[i] += 1
print(arr)

#ideally should use function for reuse as is normal, but to prevent breakdown omit until after deadline
#def vectorsim(search_Terms, dict):
#vectorsim(search_Terms, )