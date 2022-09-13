from gym import Env 
from gym.spaces import Discrete, Box
import numpy as np
import random

class ShowerEnv(Env):

    def __init__(self):
        
        #possible actions - DOWN, STAY, UP
        self.action_space = Discrete(3)

        #array for shower temp
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))

        #set start temp
        self.state = 38 + random.randint(-3,3)

        #set shower length
        self.shower_length = 60


    def step(self, action):

        #apply action
        self.state += action - 1

        #reduce time length
        self.shower_length -= 1

        #calculate reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        #check time finished
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        #fluctuate temperature
        self.state += random.randint(-1,1)

        #set info placeholder
        info = {}

        return self.state, reward, done, info


    def render(self):
        pass


    def reset(self):
        pass

