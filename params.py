d = \
{
    "train": {
        "episodes": 10000,
        "steps": 2000
    },

    "agent": { # comment is benchmark proved to have worked
        "critic_loss_weight_regularization_l2": 0.0,
        "critic_gradient_clip": 1e6, # smaller cap prevents overfitting
        "actor_noise_switch": True,
        "actor_noise_sigma" :0.1,  # TEST bigger ones
        "actor_noise_theta": 0.05,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "tau": 0.005, # slower target update stablizes training
        "gamma": 0.99,
        "critic_lr": 1e-3,
        "actor_lr": 1e-3,
        "exp_sample_size": 256,
        "bufsize": 500000,
        "optim_momentum": 1e-1,
        "actor_last_layer_weight_init": 3e-3,
        "critic_last_layer_weight_init": 3e-4,
        "critic_bn_eps": 1e-4,
        "critic_bn_momentum": 1e-2,
        "actor_loss_weight_regularization_l2": 0.0,
        "actor_gradient_clip": 1e9, # smaller cap prevents overfitting
        "update_delay":2
        #TEST? optimizer: sgd, adam
        #TEST: update by step, update all steps in one go
        #TEST? action noise: ou noise vs normal noise
    }
}


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

class Params:
    def __init__(self, kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

params = {k: Params(v) for k, v in d.items()}
params = Params(params)