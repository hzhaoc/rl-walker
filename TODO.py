# TODO: compare our TD3 to others'
# TD3_4: hidden size = 256. done reward = -1
# TD3_5: hidden size = 256. done reward = -50
# TD3_6: hidden size = 256. done reward = -100
# TD3_7: hidden size = 16.  done reward = -100


# conclusions:
# done reward decrease -> result improve. v4,v5,v6
# hiddren size increase -> result improve. v6,v7



#  TODO: try different agents:
#       - A3C: did well on leg balancing
#       - SAC (I may have already implemented in in DDPG & TD3. may be just need more epochs to train), 
#       - PPO
#       - Adv. POET 

# TODO：exploration
        # - add intrinsic reward: curiosity driven 
        #    - maximize knolwedge about environment (forward dynamics. error between f(s_t, a_t)=a'_{t+1} and a_{t+1})
        #    - frequency based
        #    - maximize information gain. entroy based
        #    - memory-based. encourage state more distant to current visited state to be explored
        #    - drawbakcs
        #      - knolwedge fading. when states cease to be novel no exploration anymore
        #      - in addition to above, function approximator slow to catch up
        # noise based
        #   - can we make noise bigger when reward is high, and vice versa?
        # 
        
# TODO: when updating online actor, use online or offline critic?
# TODO: a Param class to contain all hyper parameters
# TODO: critic output shape = action# or 1?
# TODO: PPO and other potentially intrinsically better agents
# TODO: fix TD3 action output action with nan