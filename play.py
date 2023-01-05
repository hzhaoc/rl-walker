from rl import Env
from rl.agent import *
from rl.action import *
from rl.agent.agentDDPG import AgentDDPG

MAX_STEPS = 2_500_000

env = Env.make(name="TestHumannoidEnv", render_mode="human")
agent = AgentDDPG(env)

state, info = env.reset(42)
for i in range(MAX_STEPS):
    print(f"step {i}...", sep=" ")

    #action = env.sampleAction()
    action = agent.act(state)
    print(f"    action taken {action}")

    feedback = env.step(action)
    
    agent.update(feedback)

    if feedback.terminated or feedback.truncated:
        state, info = env.reset()
    else:
        state = feedback.state
env.close()