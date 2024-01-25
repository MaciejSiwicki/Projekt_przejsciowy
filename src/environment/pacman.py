import pygame
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT
from vector import Vector
from constants import *
from entity import Entity
from sprites import PacmanSprites
import random


class Pacman(Entity):
    def __init__(self, node):
        Entity.__init__(self, node)
        self.name = PACMAN
        self.color = YELLOW
        self.direction = LEFT
        self.last_direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)

    def reset(self):
        Entity.reset(self)
        self.direction = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True

    def die(self):
        self.alive = False
        self.direction = STOP

    def update(self, dt, mode="human", action=None):
        self.sprites.update(dt)
        self.position += self.directions[self.direction] * self.speed * dt
        direction = self.getValidKey(mode, action)
        self.last_direction = self.direction
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(direction)
            if self.target is not self.node:
                self.direction = direction
            else:
                self.target = self.getNewTarget(self.direction)

            if self.target is self.node:
                self.direction = STOP
            self.setPosition()
        else:
            if self.oppositeDirection(direction):
                self.reverseDirection()

    def getValidKey(self, mode="human", action=None):
        if mode == "random":
            return random.choice([UP, DOWN, LEFT, RIGHT, STOP])
        if mode == "random_smart":
            if self.overshotTarget():
                return random.choice([UP, DOWN, LEFT, RIGHT, STOP])
            else:
                return self.direction
        if action is not None:
            observation_mapping = {0: RIGHT, 1: LEFT, 2: UP, 3: DOWN}
            return observation_mapping[action]
        return STOP

    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None

    def collideGhost(self, ghost):
        return self.collideCheck(ghost)

    def collideCheck(self, other):
        d = self.position - other.position
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRadius + other.collideRadius) ** 2
        if dSquared <= rSquared:
            return True
        return False
