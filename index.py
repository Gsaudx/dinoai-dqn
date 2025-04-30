from mss import mss # Screen capture
import pydirectinput # Sending commands
import cv2 #frame processing
import numpy as np # Transformational framework
import pytesseract # OCR for game over extraction
from matplotlib import pyplot as plt #Visualize captured frames
import time # Bring time for pauses
from gym import Env # Environment components
from gym.spaces import Box, Discrete

# IMport os for file path management
import os
# IMport Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
#Check the environment
from stable_baselines3.common import env_checker

# Import the DQN algorithm
from stable_baselines3 import DQN

class WebGame(Env):
    def __init__(self): # Setup the environment action and observation shapes
        super().__init__()

        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)

        #Define extraction parameters for the game
        #mss() = To capture the screen (in this case the sections of the game)
        self.cap = mss()
        self.game_location = {'top':300, 'left':0, 'width':600, 'height':500}
        self.done_location = {'top':405, 'left':630, 'width':660, 'height':70}


    def step(self, action): # What is called to do something in the game
        # Action key - 0 = Jump(space), 1 = Duck(down), 2 = No action (no op)
        action_map = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }

        if action != 2:
            # If the action is not "no_op", we will send the command to the game
            pydirectinput.press(action_map[action])

        # Checking whether the game is done
        done, done_cap = self.get_done()

        # Get the next observation
        new_observation = self.get_observation()

        # For every frame alive we will give a reward of 1
        # It works like this because you get more points the longer you survive
        reward = 1 #TODO We could also apply some OCR to extract the score from the game and use it as a reward
        # Info dictionary
        info = {}

        return new_observation, reward, done, info
    
    def render(self): # Visualize the game
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitKey(1) & 0xFF == ord('q'): # If 'q' is pressed, close the render window
            self.close()

    def reset(self): # Restart the game
        time.sleep(1)
        pydirectinput.click(x=150, y=150) # Click somewhere on the screen to focus the game
        pydirectinput.press('space') # Restart the game
        return self.get_observation()

    def close(self): # Closes down the observation
        cv2.destroyAllWindows() # Close all windows

    def get_observation(self): # Get the part of the game that we weant (in this case, the dino)
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]

        #Grayscale --> Converting the color image to grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        #Resize --> Resizing the image to 100x83
        resized = cv2.resize(gray, (100, 83))
        #Add channels first --> Adding a channel to the image (format required by stable-baselines3)
        channel = np.reshape(resized, (1, 83, 100))
        
        return channel

    def get_done(self): # Get the done text
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        # Valid done text
        done_strings = ['GAME', 'GAHE'] #Pytesseract can read the text wrong sometimes (in this case, it reads "GAHE" instead of "GAME")
        #TODO Preprocess the image to get a better result
        # We could do a better pre processing of the image to get a better result, but for now we will just use the OCR (as it was done in the video)
                

        done = False
        # Here, pytesseract is taking the image (done_cap) and extracting the text from it
        # The [:4] is to get only the first 4 characters of the text (in this case, should represent "GAME")
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True

        return done, done_cap 
    
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            print('Saving models to {}'.format(self.save_path))

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

env = WebGame()
# Generates a random action using action_space (nesse caso, 0, 1 ou 2)
# env.action_space.sample()

# TESTING
obs = env.get_observation()
plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB))
done, done_cap = env.get_done()
print(done)
plt.imshow(done_cap)
pytesseract.image_to_string(done_cap)[:4]

# Play 10 games (testing the environment)
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
    print(f'Total Reward for episode {episode} is {total_reward}')

env_checker.check_env(env) # Check the environment

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=10, save_path=CHECKPOINT_DIR)

# Create the DQN model
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1200000, learning_starts=1000)

# Kick off training
model.learn(total_timesteps=88000, callback=callback)

for episode in range(1):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        time.sleep(0.01)
        # obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
    print(f'Total Reward for episode {episode} is {total_reward}')