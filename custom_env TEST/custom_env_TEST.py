from gym import Env 
from gym.spaces import Discrete, Box
import numpy as np
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from matplotlib import pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

MODEL_NUM = 2


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


    def printfunc(self, action):

        #print data
        print("---------------------")

        print(f"current temp is = {self.state} degrees")

        if action == 0:
            print("Agent adjusted temperature -1")
        elif action == 1:
            print("Agent left the temperature")
        elif action == 2:
            print("Agent adjusted temperature +1")

        print("---------------------")


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
        if self.shower_length % 2 == 0:
            self.state += random.randint(-1,1)

        #set info placeholder
        info = {}

        #print
        #self.printfunc(action)

        return self.state, reward, done, info


    def render(self):
        pass


    def reset(self):
        
        #reset temp
        self.state = 38 + random.randint(-3,3)

        #reset length
        self.shower_length = 60

        return self.state



#PREPROCESS ENV
#setup env
env = ShowerEnv()

#wrap in dummy env
env = DummyVecEnv([lambda: env])

#SAVE MODEL
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(MODEL_NUM))
            self.model.save(model_path)

        return True

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)


#IMPLEMENT RL MODEL

#new model
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.001, n_steps=512)

#load model
model = PPO.load('./train/best_model_1.zip')

#train model
#model.learn(total_timesteps=1000000, callback=callback)


#run model
state = env.reset()
done = False
culm_reward = 0
while not done:
    action, _state = model.predict(state)
    state, reward, done, info = env.step(action)
    culm_reward += reward

print(f"DONE = REWARD TOTAL: {culm_reward}")




