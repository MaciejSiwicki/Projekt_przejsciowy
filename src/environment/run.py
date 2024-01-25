import pygame
from pygame.locals import QUIT
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from text import TextGroup
from sprites import MazeSprites
from vector import Vector
import numpy as np


class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.lives = 1
        self.score = 0
        self.textgroup = TextGroup()
        self.reward = 0

    def action(self, action):
        if self.pacman.alive:
            if action == RIGHT:  # Assuming -2 means move right
                return RIGHT
            elif action == LEFT:  # Assuming 2 means move left
                return LEFT
            elif action == UP:  # Assuming 1 means move up
                return UP
            elif action == DOWN:  # Assuming -1 means move down
                return DOWN

    def evaluate(self):
        if self.lives == 0:
            self.reward += -20

        if self.pacman.alive and self.pellets.isEmpty():
            self.reward += self.pellets.numEaten

        if (
            self.pacman.alive
            and self.pellets.numEaten - self.pellets.previousNumEaten == 1
        ):
            self.reward += 1.2

        if self.pacman.alive and not (
            self.pacman.direction in self.pacman.validDirections()
        ):
            self.reward += -10

        for ghost in self.ghosts:
            if 32 < Vector.magnitude(self.pacman.position - ghost.position) < 0:
                self.reward += -10

        return self.reward

    def is_done(self):
        if self.lives == 0 or (self.pellets.isEmpty() and self.pacman.alive):
            self.reward = 0
            return True
        else:
            return False

    def observe(self):
        #     print(self.pacman.node.neighbors)
        observation_mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
        return np.array(
            [
                self.pacman.position.x,
                self.pacman.position.y,
                observation_mapping[self.pacman.direction],
                self.ghosts.blinky.position.x,
                self.ghosts.blinky.position.y,
                observation_mapping[self.ghosts.blinky.direction],
                self.ghosts.pinky.position.x,
                self.ghosts.pinky.position.y,
                observation_mapping[self.ghosts.pinky.direction],
                self.ghosts.inky.position.x,
                self.ghosts.inky.position.y,
                observation_mapping[self.ghosts.inky.direction],
                self.ghosts.clyde.position.x,
                self.ghosts.clyde.position.y,
                observation_mapping[self.ghosts.clyde.direction],
                self.pellets.numEaten,
            ]
        ).flatten()

    # self.ghosts.XXX.mode.current,
    # self.pellets.powerpellets
    def restartGame(self):
        self.lives = 1
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def startGame(self):
        self.setBackground()
        self.mazesprites = MazeSprites(
            "utils/maze1.txt",
            "utils/maze1_rotation.txt",
        )
        self.background = self.mazesprites.constructBackground(self.background, 0)
        self.nodes = NodeGroup("utils/maze1.txt")
        self.nodes.setPortalPair((0, 17), (27, 17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12, 14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15, 14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("utils/maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2 + 11.5, 0 + 14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0 + 11.5, 3 + 14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4 + 11.5, 3 + 14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2 + 11.5, 3 + 14))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2 + 11.5, 3 + 14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)
        return self

    def update(self, mode="human", action=None):
        dt = self.clock.tick(30) / 1000
        self.textgroup.update(dt)
        self.pacman.update(dt, mode, action)
        self.ghosts.update(dt)
        self.pellets.update(dt)
        self.checkPelletEvents()
        self.checkGhostEvents()
        self.checkEvents()
        self.render()

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.updateScore(ghost.points)
                    self.textgroup.addText(
                        str(ghost.points),
                        WHITE,
                        ghost.position.x,
                        ghost.position.y,
                        8,
                        time=1,
                    )
                    self.ghosts.updatePoints()
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.pacman.die()

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()

    def render(self):
        self.screen.blit(self.background, (0, 0))  # type: ignore
        self.pellets.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        pygame.display.update()

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        self.pellets.previousNumEaten = self.pellets.numEaten
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky)
            if self.pellets.numEaten == 70:
                self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
                self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.restartGame()
