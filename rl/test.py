from rl import Env
from rl.agent import *
from rl.agent.agentDDPG import AgentDDPG
from rl.agent.agentTD3 import AgentTD3
from rl.util.utils import Queue


MAX_STEPS = 500
MAX_EPOCHS = 1_000_000
i = 121
# agentClass = AgentDDPG
agentClass = AgentTD3
#################################################################################
#################################################################################  A: working. B: barely working. C: not working
#                                               task            DDPG        TD3
EnvNames = {
    110: "TestReacherEnv",                    # locomotion      C           C
    111: "TestContinuous_MountainCarEnv",     # locomotion      B           A
    112: "TestHumannoidEnv",                  # locomotion                  C
    120: "TestWalker2dEnv",                   # locomotion 
    121: "TestBipedalWalker",                 # locomotion      C           C
    130: "TestHalfCheetahEnv",                # locomotion                  A
    140: "TestAntEnv",                        # locomotion      
    150: "TestHopperEnv",                     # locomotion     
    160: "TestSwimmerEnv",                    # locomotion      

    220: "TestInvertedDoublePendulumEnv",     # stablization    
    230: "TestPendulumEnv",                   # stablization    A-          A
}
#################################################################################
#################################################################################

torch.autograd.set_detect_anomaly(mode=False, check_nan=True)
env = Env.make(name=EnvNames[i], render_mode="human")
agent = agentClass(env=env, 
                  critic_loss_weight_regularization_l2=0.0,
                  critic_gradient_clip=1e3,
                  actor_noise_switch=True,  # exploration switch
                  actor_noise_sigma=0.2,  # volatility of noise in actor
                  actor_noise_theta=0.1, # temporal scaler in actor noise
                  policy_noise=0.2, # noise for training only
                  noise_clip=0.5, # noise for training only
                  tau=0.05, # target network weight update rate
                  gamma=0.99, # net present value discount rate. the closer the 1, the longer term of reward memory
                  critic_lr=1e-4, # critic network weight learning rate
                  actor_lr=1e-3, # actor network weight learning rate
                  exp_sample_size=256,
                  bufsize=100_000, # experiance buffer size for sampling
                  optim_momentum=1e-1,  # stochastic gradient descent momentum
                  actor_last_layer_weight_init=3e-3, # actor network last layer initial uniform distribution radius around 0
                  critic_last_layer_weight_init=3e-4, # critic network last layer initial uniform distribution range
                  critic_bn_eps=1e-4,  # critic network batch norm epsilon for scaling stability
                  critic_bn_momentum=1e-2,  # critic network batch norm running mean and standard deviation momentum 
                  #critic_layer_depth=3,  # critic network number of layers. action will be entered in the last second layer. MIN=3
                  #actor_layer_depth=3,  # actor network layer depth. MIN=3
                  actor_loss_weight_regularization_l2=0.0,
                  actor_gradient_clip=1e9,
                  update_delay=2,  # how many steps it take to update online actor, target actor, target critics once.
        )



fileRewards = open(rf"play/{EnvNames[i]}.txt", "w+", 1)
for j in range(MAX_EPOCHS):
    r = 0.0
    state, _ = env.reset()
    agent.reset()

    for i in range(MAX_STEPS):
        action = agent.act(state)
        print(action)
        next_state, reward, terminated, truncated, info = env.step(action)

        r += reward
        #print(f"epoch {j} step {i}: rFwd {round(info['rFwd'], 2)}, rCtrl {round(info['rCtrl'], 2)}, rAlive {round(info['rAlive'], 2)}, rHeight {round(info['rHeight'], 2)}, rHead: {round(0.0, 2)}")
        agent.buf.push(state, action, reward, next_state, terminated)
        agent.update()

        if terminated:
            break
        state = next_state

    fileRewards.write(f"{round(r, 2)}\n")
fileRewards.close()
env.close()