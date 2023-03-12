from rl import Env
from rl.agent import *
from rl.agent.agentDDPG import AgentDDPG
from rl.agent.agentTD3 import AgentTD3
from rl.util.utils import Queue


MAX_STEPS = 500
MAX_EPOCHS = 1_000_000
i = 32
#                                              task          DDPG   TD3
EnvNames = {
    110: "TestReacherEnv",                    # locomotion,       
    111: "TestContinuous_MountainCarEnv",     # locomotion,   pass, ?
    112: "TestHumannoidEnv",                  # locomotion,   fail,     
    120: "TestWalker2dEnv",                   # locomotion,
    130: "TestHalfCheetahEnv",                # locomotion,   fail,  pass,
    140: "TestAntEnv",                        # locomotion,   fail,
    150: "TestHopperEnv",                     # locomotion,     
    160: "TestSwimmerEnv",                    # locomotion,   fail,

    220: "TestInvertedDoublePendulumEnv",     # stablization, fail,
    230: "TestPendulumEnv",                   # stablization, pass,  pass,   
}

torch.autograd.set_detect_anomaly(mode=False, check_nan=True)
env = Env.make(name=EnvNames[i], render_mode="human")
agentClass = AgentTD3 
agent = agentClass(env=env, 
                  critic_loss_weight_regularization_l2=0.0,
                  critic_gradient_clip=1e3,
                  actor_noise_switch=True,  # actor noise for exploration. some problem needs exploration more than others
                  actor_noise_sigma=0.2,  # volatility of noisee in actor
                  actor_noise_theta=0.1, # temporal scaler in actor noise
                  tau=0.05, # target network weight update rate
                  gamma=0.99, # net present value discount rate
                  critic_lr=1e-4, # critic network weight learning rate
                  actor_lr=1e-3, # actor network weight learning rate
                  exp_sample_size=256,
                  bufsize=100_000, # experiance buffer size for sampling
                  optim_momentum=1e-1,  # stochastic gradient descent momentum
                  actor_last_layer_weight_init=3e-3, # actor network last layer initial uniform distribution radius around 0
                  critic_last_layer_weight_init=3e-4, # critic network last layer initial uniform distribution range
                  critic_bn_eps=1e-4,  # critic network batch norm epsilon for scaling stability
                  critic_bn_momentum=1e-2,  # critic network batch norm runnign mean and standard deviation momentum 
                  #critic_layer_depth=3,  # critic network number of layers. action will be entered in the last second layer. MIN=3
                  #actor_layer_depth=3,  # actor network layer depth. MIN=3
                  actor_loss_weight_regularization_l2=0.0,
                  actor_gradient_clip=1e9,
                  update_delay=3,  # how many steps to update online actor, target actor, target critics for once.
        )

fileRewards = open(rf"play/{EnvNames[i]}.txt", "w+", 1)
for j in range(MAX_EPOCHS):
    r = 0.0
    state, _ = env.reset()
    agent.reset()

    for i in range(MAX_STEPS):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        r += reward
        #print(f"epoch {j} step {i}: rFwd {round(info['rFwd'], 2)}, rCtrl {round(info['rCtrl'], 2)}, rAlive {round(info['rAlive'], 2)}, rHeight {round(info['rHeight'], 2)}, rHead: {round(0.0, 2)}")
        agent.buf.push(state, action, reward, next_state, terminated)
        agent.update()

        if terminated: break
        state = next_state

    fileRewards.write(f"{round(r, 2)}\n")
fileRewards.close()
env.close()