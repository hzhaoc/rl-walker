import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from collections import defaultdict
import os


moving_average_window = 100
# print(os.curdir)


def plot(arr):
    for k in arr:
        plt.plot(arr[k], label=k)
    plt.legend()
    # plt.show()


# fnames = ["TestPendulumEnv_1","TestPendulumEnv_2","TestPendulumEnv_3","TestPendulumEnv_TD3_1","TestPendulumEnv_TD3_2","TestPendulumEnv_TD3_3","TestPendulumEnv_TD3_4"]
# fnames = ["TestPendulumEnv_6","TestPendulumEnv_TD3_5"]
# fnames = ["TestHumannoidEnv"]
# fnames = ["TestHalfCheetahEnv_TD3"]
# fnames = ["TestContinuous_MountainCarEnv_DDPG","TestContinuous_MountainCarEnv_DDPG1","TestContinuous_MountainCarEnv_TD1"]
# fnames = ["TestReacherEnv_DDPG","TestReacherEnv_TD"]
# fnames = ["TestWalker2dEnv"]
# fnames = ["TestBipedalWalker_TD3_1","TestBipedalWalker_TD3_2"]
fnames = ["TestBipedalWalker_TD3_4", "TestBipedalWalker_TD3_5", "TestBipedalWalker_TD3_6", "TestBipedalWalker_TD3_7"]


arrByfname = defaultdict(list)
for fname in fnames:
    with open(rf"play/{fname}.txt", "r") as file:
        for line in file:
            arrByfname[fname].append(float(line.split(",")[1]))


plt.figure(1)
for j, k in enumerate([2,10,100]): # moving average window size
    B = defaultdict(list)
    for fname in arrByfname:
        A = arrByfname[fname]
        n = len(A)
        s = 0
        for i in range(n):
            s += A[i]
            if i >= k:
                s -= A[i-k]
            # A[i] = s / (min(i+1, k))
            B[fname].append(s / (min(i+1, k)))
    plt.subplot(3,1,j+1)
    plot(B)
plt.show()