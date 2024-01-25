import gym
from gym import spaces
import numpy as np
from run import GameController
from constants import *


class CustomEnv(gym.Env):
    # metadata = {'render.modes' : ['human']}
    def __init__(self):
        gamecontroller = GameController()
        max_x = SCREENWIDTH
        max_y = SCREENHEIGHT
        num_ghosts = 4
        eatenPellets = 244
        self.pygame = gamecontroller.startGame()
        self.action_space = spaces.Discrete(3)
        flat_dimensions = (
            [SCREENWIDTH, SCREENHEIGHT, 5]
            + [SCREENWIDTH, SCREENHEIGHT, 5] * (num_ghosts)
            + [max(eatenPellets, 1)]
        )
        multi_discrete_space = spaces.MultiDiscrete(flat_dimensions)
        self.observation_space = multi_discrete_space
        self.rewards = []
        self.timesteps = []
        self.scores = []
        self.timesteps_alive = 0

    def reset(self):
        self.pygame.restartGame()
        self.timesteps_alive = 0
        obs = self.pygame.observe()
        return obs

    def save_reward(self, filename="rewards.txt"):
        with open(filename, "w") as file:
            for i, (reward, timesteps_alive, score) in enumerate(
                zip(self.rewards, self.timesteps, self.scores)
            ):
                file.write(f"{i+1} {reward} {timesteps_alive} {score}\n")

    def step(self, action):
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        self.timesteps_alive += 1
        done = self.pygame.is_done()
        if done:
            self.rewards.append(reward)
            self.timesteps.append(self.timesteps_alive)
            self.scores.append(self.pygame.score)
        self.pygame.update("human", action)
        return obs, reward, done, {}

    def render(self, mode="human"):
        self.pygame.update()

    def close(self):
        pass
