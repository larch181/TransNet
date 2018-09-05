# file: test.py
import numpy as np
import matplotlib.pyplot as plt
import DataOperator
points = np.random.rand(1000,3)
K=8
index = np.zeros(K).astype(np.int32)
print(index)
DataOperator.farthest_point_samplin_func(points,index,8)

print(index)
