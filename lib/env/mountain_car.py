"""
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering
from gym import wrappers
from datetime import datetime

class MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, trailer=False, show_velo=False):
        self.name = '2d_mountain_car'
        self.trailer = trailer # to show trails of the agent
        self.show_velo=False #to show velo of the agent
        self.last_few_positions = []
        self.trail_num = 40

        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5

        self.low = np.array([self.min_position, -self.max_speed])
        self.high = np.array([self.max_position, self.max_speed])

        self.viewer = None

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high)

        self._seed()
        self.reset()

    def set_state(self, state):
        self.state = state
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position, velocity = self.state
        velocity += (action-1)*0.001 + math.cos(3*position)*(-0.0025)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if (position==self.min_position and velocity<0): velocity = 0

        done = bool(position >= self.goal_position)
        reward = -1.0

        self.last_few_positions.append(position)
        if len(self.last_few_positions) == self.trail_num+1:
            del self.last_few_positions[0]

        self.state = (position, velocity)
        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def _render(self, mode='human', close=False, action_idx=None, action_vec=None):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

            # set trails
            if self.trailer:
                self.trail_trans = []
                trail_color_increment = 1.0 / self.trail_num
                for i in range(self.trail_num):
                    trail = rendering.make_circle(radius=5)
                    trail.set_color(1-trail_color_increment*i,0,0)
                    trans = rendering.Transform()
                    trail.add_attr(trans)
                    self.viewer.add_geom(trail)
                    self.trail_trans.append(trans)
                    
                    
        if action_idx is not None or action_vec is not None:
            actions = np.array([[-1,0],[0,0],[1,0]])
            if action_idx is not None:
                action_vertex = tuple(actions[action_idx] * scale * 0.1)
            elif action_vec is not None:
                action_vertex = tuple(np.sum(action_vec.reshape(-1,1)*actions,axis=0) * scale * 0.1)
            action_direction = rendering.make_polyline([(0,0),action_vertex])
            action_direction.add_attr(self.cartrans)
            action_direction.set_linewidth(2)
            action_direction.set_color(0, 1, 0)
            self.viewer.add_onetime(action_direction)


        if self.show_velo == True:
            velocity_vertex = tuple(np.array(self.state[2:4]) * scale * 10)
            velocity = rendering.make_polyline([(0,0),velocity_vertex])
            velocity.add_attr(self.cartrans)
            velocity.set_linewidth(2)
            velocity.set_color(0, 0, 1)
            self.viewer.add_onetime(velocity)


        if self.trailer:
            for i in range(len(self.last_few_positions)):
                self.trail_trans[i].set_translation((self.last_few_positions[i]-self.min_position)*scale, self._height(self.last_few_positions[i])*scale)
                

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))


        return self.viewer.render(return_rgb_array = mode=='rgb_array')

if  __name__ == '__main__':
    # test the environment with random actions

    env = MountainCarEnv(trailer=True, show_velo=True)
    add_str = datetime.now().time().isoformat()
    # env = wrappers.Monitor(env, './videos/' + add_str)
    state = env.reset()
    is_reset = False
    for t in range(100000):

        if is_reset:
            state = env.reset()

        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        env._render(action_idx=action)
        # env._render()


        if done:
            is_reset = True
