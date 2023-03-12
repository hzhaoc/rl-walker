import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict

def plot(arr):
    for k in arr:
        plt.plot(arr[k], label=k)
    plt.legend()
    plt.show()

#fnames = ["pendulum"]
#fnames = ["TestHumannoidEnv"]
#fnames = ["TestHalfCheetah"]
fnames = ["TestContinuous_MountainCarEnv"]

arr = defaultdict(list)
for fname in fnames:
    with open(rf"play/{fname}.txt", "r") as file:
        for line in file:
            arr[fname].append(float(line))

window = 1
arr2 = defaultdict(list)
if window > 1:
    for k in arr:
        for i, e in enumerate(arr[k]):
            size = window if i + 1 >= window else i + 1
            arr2[k].append(sum(arr[k][i:i+size])/size)
    plot(arr2)
else:
    plot(arr)