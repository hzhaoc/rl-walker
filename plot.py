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


# TD3_4:  hidden size = 256  done reward = -1    update by step   batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# TD3_5:  hidden size = 256  done reward = -50   update by step   batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# TD3_6:  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# TD3_7:  hidden size = 16   done reward = -100  update by step   batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# TD3_8:  hidden size = 256  done reward = -100  update by epoch  batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# TD3_9:  hidden size = 400  done reward = -100  update by epoch  batch_size = 100   normal noise  critic_output_1               no_batch_norm   no_grad_clip    optim.adam
# TD3_10  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_1               no_batch_norm   grad_clip       optim.sgd
# TD3_11  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_action_space    no_batch_norm   grad_clip       optim.sgd
# TD3_12  hidden size = 256  done reward = -100  update by step   batch_size = 256   normal noise  critic_output_action_space    batch_norm      no_grad_clip    optim.adam





# [v4,v5,v6]:
# -> done reward 
# -> NOTE: use small done reward
test1 = ["TestBipedalWalker_TD3_4_0", "TestBipedalWalker_TD3_5", "TestBipedalWalker_TD3_6"]

# [v6,v7]: 
# -> hidden size 
# -> NOTE: use big hidden size
test2 = ["TestBipedalWalker_TD3_6", "TestBipedalWalker_TD3_7"]

# [v6,v8]: 
# -> update by epoch or step 
# -> NOTE: use step update
test3 = ["TestBipedalWalker_TD3_6", "TestBipedalWalker_TD3_8"]

# [bm0,bm_grad_clip1]
# -> test grad clip 
# -> NOTE: does not matter
test4 = ["TestBipedalWalker_TD3_benchmark_0", "TestBipedalWalker_TD3_benchmark_clipgrad_1"]

# [bm_gradclip_0,bm_gradclip_1]
# -> test multiple runs of same config 
# -> NOTE: agent can suffer from initial values
test5 = ["TestBipedalWalker_TD3_benchmark_clipgrad_0", "TestBipedalWalker_TD3_benchmark_clipgrad_1"]

# [bm0,v9]: mimic bencmark
# -> test the benchmark copy
# -> NOTE: copy is close to benchmark's stability and score (slightly worse. copy stable peak at ~295; benchmark stable peak at ~310)
test6  = ["TestBipedalWalker_TD3_benchmark_0", "TestBipedalWalker_TD3_9"]

# [v6,v9]: 
# -> compare performance of benchmark copy v9 to v6
# TD3_6:  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# TD3_9:  hidden size = 400  done reward = -100  update by epoch  batch_size = 100   normal noise  critic_output_1               no_batch_norm   no_grad_clip    optim.adam
# -> more stable and higher peak. (hidden size, batch update, batch size, noise type, critic output size, batch norm, grad clip, optimizer) improves agent
# -> NOTE: use v9. but need more tests to locate the real factor
test6_1  = ["TestBipedalWalker_TD3_6", "TestBipedalWalker_TD3_9"] 

# TODO: [v6, v10, v11]
# -> test critic output size AND / OR batch norm
# -> v6,v10:  no diff (size,batch norm)
# -> v6,v11:  no diff or batch norm slightly better (batch norm)
# -> v10.v11: no diff or size 1 slightly better (critic output size)
# -> NOTE: use output size 1 and batch norm
# TD3_6:  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# TD3_10  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_1               no_batch_norm   grad_clip       optim.sgd
# TD3_11  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_action_space    no_batch_norm   grad_clip       optim.sgd
# -> (critic output size, batch norm) does NOT improve agent
test7 = ["TestBipedalWalker_TD3_6", "TestBipedalWalker_TD3_10", "TestBipedalWalker_TD3_11"]

# [v9, v10]
# -> test hidden size, batch update, batch size, noise type, grad clip, optimizer
# TD3_9:  hidden size = 400  done reward = -100  update by epoch  batch_size = 100   normal noise  critic_output_1               no_batch_norm   no_grad_clip    optim.adam
# TD3_10  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_1               no_batch_norm   grad_clip       optim.sgd
# -> v9 has MORE STABLE and HIGHER score! (hidden size, batch update, batch size, noise type, grad clip, optimizer) improves agent
test8 = ["TestBipedalWalker_TD3_9", "TestBipedalWalker_TD3_10"]


# TODO: [v6, v12]
# -> test noise type, grad clip, optimizer (2nd half from test8)
# TD3_12  hidden size = 256  done reward = -100  update by step   batch_size = 256   normal noise  critic_output_action_space    batch_norm      no_grad_clip    optim.adam
# TD3_6:  hidden size = 256  done reward = -100  update by step   batch_size = 256   ou noise      critic_output_action_space    batch_norm      grad_clip       optim.sgd
# -> 


fnames =  test5


arrByfname = defaultdict(list)
for fname in fnames:
    with open(rf"test/{fname}.txt", "r") as file:
        for line in file:
            arrByfname[fname].append(float(line.split(",")[1]))


plt.figure(1)
for j, k in enumerate([1,10,100]): # moving average window size
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
        B[fname] = B[fname][0:10000]
    plt.subplot(3,1,j+1)
    plot(B)
plt.show()