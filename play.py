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

#env = Env.make(name="TestHumannoidEnv", render_mode="human")
env = Env.make(name="TestPendulumEnv", render_mode="human")
agent = AgentDDPG(env=env, 
                  tau=0.1, 
                  gamma=0.95, 
                  critic_lr=1e-3, 
                  actor_lr=1e-3,
                  bufsize=50_000,
                  momentum=0.9,
        )

fileRewards = open("rewards.txt", "w+")

"""
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

window = Queue(maxlen=500)
for j in range(MAX_EPOCHS):
    state, _ = env.reset(random.randint(1, 100))
    #state = normedState(state)

    for i in range(MAX_STEPS):
        action = agent.act(state)
        #action = denormedAction(action)
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
        
# fileActions.close()
fileRewards.close()
env.close()