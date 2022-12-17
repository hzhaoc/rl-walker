from rl import Env
from rl.agent import *
from rl.action import *


env = Env.make(name="TestHumannoidEnv", render_mode="human")
#agent = Agent()

state, info = env.reset(42)
for _ in range(10000):
    action = env.sampleAction()
    #action = agent.act()
    feedback = env.step(action)

    if feedback.terminated or feedback.truncated:
        state, info = env.reset()
env.close()