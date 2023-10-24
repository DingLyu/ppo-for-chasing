import gym
import json
import pygame
import numpy as np

window_size = 600
r = 20
r_d = 100


class Agent(object):
    def __init__(self, ID, r, r_d, x, y):
        self.ID = ID
        self.r = r
        self.r_d = r_d
        self.x = x
        self.y = y


class Obstacle(object):
    def __init__(self, ID, r, x, y):
        self.ID = ID
        self.r = r
        self.x = x
        self.y = y


class Target(object):
    def __init__(self, r, x, y):
        self.r = r
        self.x = x
        self.y = y


class World_v0(gym.Env):
    def __init__(self):
        self.agent = None
        self.target = None
        self.agent_location = []
        self.target_location = []

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(10, 590, shape=(2, 1), dtype=float),
                "target": gym.spaces.Box(10, 590, shape=(2, 1), dtype=float),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([10, 0]),
            1: np.array([0, 10]),
            2: np.array([-10, 0]),
            3: np.array([0, -10]),
        }
        self.render_mode = "human"
        self.window = None
        self.window_size = window_size
        self.r = r
        self.r_d = r_d

    def reset(self):
        with open('data1.json', 'r') as jsonfile:
            self.parameters = json.load(jsonfile)
        jsonfile.close()
        self.agent_location = np.array(self.parameters["agent"])
        self.target_location = np.array(self.parameters["target"])
        self.agent = Agent(0, self.r, self.r_d, self.agent_location[0], self.agent_location[1])
        self.target = Target(self.r, self.target_location[0], self.target_location[1])
        observation = self._get_obs()
        info = self._get_info()
        self._render_frame()
        return observation, info

    def step(self, action):
        direction = np.array(self._action_to_direction[action])
        self.agent_location = np.clip(
            self.agent_location + direction, self.r, self.window_size - self.r
        )
        self.agent.x = self.agent_location[0]
        self.agent.y = self.agent_location[1]
        self.target.x = self.target_location[0]
        self.target.y = self.target_location[1]
        terminated = self._cover_target()
        if terminated:
            reward = 1
        else:
            reward = 0
        observation = self._get_obs()
        info = self._get_info()
        self._render_frame()
        return observation, reward, terminated, info

    def _get_obs(self):
        return np.hstack((self.agent_location, self.target_location))

    def _get_info(self):
        return (self.agent_location, self.target_location)

    def _cover_target(self):
        return np.linalg.norm(self.agent_location - self.target_location) <= self.r_d

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))


        pygame.draw.circle(
            canvas,
            (255, 147, 0),
            (self.agent.x, self.agent.y),
            self.agent.r,
        )
        pygame.draw.circle(
            canvas,
            (68, 114, 196),
            (self.agent.x, self.agent.y),
            self.agent.r_d,
            1
        )

        pygame.draw.circle(
            canvas,
            (68, 114, 196),
            (self.target.x, self.target.y),
            self.target.r,
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class World_v1(gym.Env):
    def __init__(self):
        self.agent = None
        self.ostacles = []
        self.target = None
        self.agent_location = []
        self.obstacle_location = []
        self.target_location = []


        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(10, 590, shape=(2, 1), dtype=float),
                "obstacle": gym.spaces.Box(10, 590, shape=(2, 3), dtype=float),
                "target": gym.spaces.Box(10, 590, shape=(2, 1), dtype=float),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([10, 0]),
            1: np.array([0, 10]),
            2: np.array([-10, 0]),
            3: np.array([0, -10]),
        }
        self.render_mode = "human"
        self.window = None
        self.window_size = window_size
        self.r = r
        self.r_d = r_d

    def reset(self):
        with open('data2.json', 'r') as jsonfile:
            self.parameters = json.load(jsonfile)
        jsonfile.close()
        self.agent_location = np.array(self.parameters["agent"])
        self.obstacle_location = np.array(self.parameters["obstacle"])
        self.target_location = np.array(self.parameters["target"])
        self.agent = Agent(0, self.r, self.r_d, self.agent_location[0], self.agent_location[1])
        self.obstacles = [Obstacle(i, self.r, self.obstacle_location[i][0], self.obstacle_location[i][1]) for i in range(len(self.obstacle_location))]
        self.target = Target(self.r, self.target_location[0], self.target_location[1])
        observation = self._get_obs()
        info = self._get_info()
        self._render_frame()
        return observation, info

    def step(self, action):
        direction = np.array(self._action_to_direction[action])
        self.agent_location = np.clip(
            self.agent_location + direction, self.r, self.window_size - self.r
        )
        self.agent.x = self.agent_location[0]
        self.agent.y = self.agent_location[1]
        self.target.x = self.target_location[0]
        self.target.y = self.target_location[1]
        if self._cover_target():
            terminated = True
            reward = 100
        else:
            if self._is_collision():
                terminated = True
                reward = -1
            else:
                terminated = False
                reward = 1e-3 / (1 + np.linalg.norm(self.agent_location - self.target_location))
        observation = self._get_obs()
        info = self._get_info()
        self._render_frame()
        return observation, reward, terminated, info

    def _get_obs(self):
        return np.hstack((self.agent_location, np.hstack(self.obstacle_location), self.target_location))

    def _get_info(self):
        return (self.agent_location, self.target_location)

    def _cover_target(self):
        return np.linalg.norm(self.agent_location - self.target_location) <= self.r_d

    def _is_collision(self):
        for i in range(len(self.obstacle_location)):
            if np.linalg.norm(self.agent_location - self.obstacle_location[i]) <= 2 * self.r:
                return True
        return False


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))


        pygame.draw.circle(
            canvas,
            (255, 147, 0),
            (self.agent.x, self.agent.y),
            self.agent.r,
        )
        pygame.draw.circle(
            canvas,
            (68, 114, 196),
            (self.agent.x, self.agent.y),
            self.agent.r_d,
            1
        )

        for i in range(len(self.obstacles)):
            pygame.draw.circle(
                canvas,
                'black',
                (self.obstacles[i].x, self.obstacles[i].y),
                self.obstacles[i].r,
            )

        pygame.draw.circle(
            canvas,
            (68, 114, 196),
            (self.target.x, self.target.y),
            self.target.r,
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class World_v2(gym.Env):
    def __init__(self):
        self.agents = []
        self.obtacles = []
        self.target = None
        self.agent_location = []
        self.obstacle_location = []
        self.target_location = []
        self.successor = []

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(10, 590, shape=(2, 2), dtype=float),
                "obstacle": gym.spaces.Box(10, 590, shape=(2, 3), dtype=float),
                "target": gym.spaces.Box(10, 590, shape=(2, 1), dtype=float),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([10, 0]),
            1: np.array([0, 10]),
            2: np.array([-10, 0]),
            3: np.array([0, -10]),
        }
        self.render_mode = "human"
        self.window = None
        self.window_size = window_size
        self.r = r
        self.r_d = r_d

    def reset(self):
        with open('data3.json', 'r') as jsonfile:
            self.parameters = json.load(jsonfile)
        jsonfile.close()
        self.agent_location = np.array(self.parameters["agent"])
        self.obstacle_location = np.array(self.parameters["obstacle"])
        self.target_location = np.array(self.parameters["target"])
        self.agents = [Agent(i, self.r, self.r_d, self.agent_location[i][0], self.agent_location[i][1]) for i in range(len(self.agent_location))]
        self.obstacles = [Obstacle(i, self.r, self.obstacle_location[i][0], self.obstacle_location[i][1]) for i in range(len(self.obstacle_location))]
        self.target = Target(self.r, self.target_location[0], self.target_location[1])
        self.successor = [False, False, False]
        observation = self._get_obs()
        info = self._get_info()
        self._render_frame()
        return observation, info

    def step(self, action):
        direction = np.array([self._action_to_direction[_] for _ in action])
        self.agent_location = np.clip(
            self.agent_location + direction, self.r, self.window_size - self.r
        )
        for i in range(len(self.agents)):
            self.agents[i].x = self.agent_location[i][0]
            self.agents[i].y = self.agent_location[i][1]
        for j in range(len(self.obstacles)):
            self.obstacles[j].x = self.obstacle_location[j][0]
            self.obstacles[j].y = self.obstacle_location[j][1]
        self.target.x = self.target_location[0]
        self.target.y = self.target_location[1]
        reward = []
        cover = []
        collision = []
        for i in range(len(self.agents)):
            cover.append(self._cover_target(i))
            collision.append(self._is_collision(i))
            if self._cover_target(i):
                if not self.successor[i]:
                    reward.append(100)
                else:
                    reward.append(0)
                self.successor[i] = True
            else:
                if self._is_collision(i):
                    reward.append(-100)
                else:
                    if not self.successor[i]:
                        reward.append(1e-3 / (1 + np.linalg.norm(self.agent_location[i] - self.target_location)))
                    else:
                        reward.append(-1)
        if np.array(cover).all() or np.array(collision).any():
            terminated = True
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()
        self._render_frame()
        return observation, reward, terminated, info

    def _get_obs(self):
        return np.array([np.hstack((self.agent_location[i], np.hstack(self.obstacle_location), self.target_location)) for i in range(len(self.agents))])

    def _get_info(self):
        return (self.agent_location, self.obstacle_location, self.target_location)

    def _cover_target(self, i):
        return np.linalg.norm(self.agent_location[i] - self.target_location) <= self.r_d

    def _is_collision(self, i):
        for j in range(len(self.obstacle_location)):
            if np.linalg.norm(self.agent_location[i] - self.obstacle_location[j]) <= 2 * self.r:
                return True
        return False


    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        for i in range(len(self.agents)):
            pygame.draw.circle(
                canvas,
                (255, 147, 0),
                (self.agents[i].x, self.agents[i].y),
                self.agents[i].r,
            )
            pygame.draw.circle(
                canvas,
                (68, 114, 196),
                (self.agents[i].x, self.agents[i].y),
                self.agents[i].r_d,
                1
            )

        for j in range(len(self.obstacles)):
            pygame.draw.circle(
                canvas,
                'black',
                (self.obstacles[j].x, self.obstacles[j].y),
                self.obstacles[j].r,
            )

        pygame.draw.circle(
            canvas,
            (68, 114, 196),
            (self.target.x, self.target.y),
            self.target.r,
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()