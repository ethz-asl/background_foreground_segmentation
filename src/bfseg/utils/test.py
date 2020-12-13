import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from bfseg.utils import NyuDataLoader
import numpy as np
import matplotlib.pyplot as plt
import math

train_ds, train_info, valid_ds, valid_info, test_ds, test_info = NyuDataLoader.NyuDataLoader(3,(720, 480), removeDepth=False ).getDataSets()

sum = 0
sqrd = 0
cnt = 0

for img,  label in train_ds.take(1):
    d = label['depth'].numpy()
    # d = (d - 260)/ 131.218
    valid_d = d[d!=0].ravel()
    sum += np.sum(valid_d)
    sqrd+= np.sum(valid_d**2)
    cnt += np.sum(valid_d > 0)

print(sum)
print(sqrd)
mean = sum/cnt
print(mean, "mean")
print(sqrd, "sqrd")
print(cnt, "cnt")
a = sqrd/cnt - mean**2
print(a)
print("stdev", math.sqrt(a))

# import bfseg.models.MultiTaskingModels as mtm
#
# mtm.PSPNet(classes = 2).summary()
#
