from rl import Env
from rl.agent import *
from rl.action import *
from rl.agent.agentDDPG import AgentDDPG
from rl.util.utils import Queue
import random


MAX_STEPS = 500
MAX_EPOCHS = 20_000

def normedState(state):
    return np.multiply((state - np.array([-1, -1, -8])), np.array([1/2, 1/2, 1/16]))

def denormedAction(action):
    return action * 4 - 2

env = Env.make(name="TestPendulumEnv", render_mode="human") # NOTE 1
#env = Env.make(name="TestHumannoidEnv", render_mode="human")
agent = AgentDDPG(env=env, 
                  tau=0.05, # target network weight update rate
                  gamma=0.95, # net present value discount rate
                  critic_lr=1e-3, # critic network weight learning rate
                  actor_lr=1e-3, # actor network weight learning rate
                  bufsize=100_000, # experiance buffer size for sampling
                  optim_momentum=1e-1,  # stochastic gradient descent momentum
                  hidden_layer_size=256, # network approximator number of neuron / activations in each middle layer
                  actor_last_layer_weight_init=3e-3, # actor network last layer initial uniform distribution radius around 0
                  critic_last_layer_weight_init=3e-4, # critic network last layer initial uniform distribution range
        )

fileRewards = open("rewards.txt", "w+")

window = Queue(maxlen=500)
for j in range(MAX_EPOCHS):
    state, _ = env.reset(random.randint(1, 100))

    for i in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        next_state = normedState(next_state)
        print(f"epoch {j} step {i}: action {action}, reward {reward}")
        agent.buf.push(state, action, reward, next_state, terminated)
        if len(agent.buf) > 128:
            agent.update(128)

        window.append(reward)
        fileRewards.write(f"{round(window.avg, 2)}\n")

        if terminated or truncated:
            break
        state = next_state
        
fileRewards.close()
env.close()


""" NOTE 1
| Num | Action | Min  | Max |
|-----|--------|------|-----|
| 0   | Torque | -2.0 | 2.0 |

| Num | Observation      | Min  | Max |
|-----|------------------|------|-----|
| 0   | x = cos(theta)   | -1.0 | 1.0 |
| 1   | y = sin(theta)   | -1.0 | 1.0 |
| 2   | Angular Velocity | -8.0 | 8.0 |

reward = [-16.x, 0]
"""