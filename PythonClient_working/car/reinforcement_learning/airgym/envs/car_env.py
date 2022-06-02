import setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from scipy.special import softmax


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)

        # for descrete action space
        # self.action_space = spaces.Discrete(16)

        # for continuous action space
        # self.action_space = spaces.Box(low=-1, high=1, shape=(16, ))
        self.action_space = spaces.Box( np.array([0, 0, -1]).astype(np.float32), np.array([1, 1, 1]).astype(np.float32)) # throttle, brake, steering

        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.DepthPerspective, True, False
        )

        self.car_controls = airsim.CarControls()
        self.car_state = None
        self.driving_start_time = time.time()

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        self.driving_start_time = time.time()
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        elif action == 1:
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
            self.car_controls.steering = -0.5
        elif action == 4:
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
            self.car_controls.steering = 0.25
        elif action == 5:
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
            self.car_controls.steering = -0.25
        elif action == 6:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            self.car_controls.steering = 0
        elif action == 7:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            self.car_controls.steering = 0.5
        elif action == 8:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            self.car_controls.steering = -0.5
        elif action == 9:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            self.car_controls.steering = 0.25
        elif action == 10:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
            self.car_controls.steering = -0.25
        elif action == 11:
            self.car_controls.throttle = 0
            self.car_controls.brake = 0
            self.car_controls.steering = 0
        elif action == 12:
            self.car_controls.throttle = 0
            self.car_controls.brake = 0
            self.car_controls.steering = 0.5
        elif action == 13:
            self.car_controls.throttle = 0
            self.car_controls.brake = 0
            self.car_controls.steering = -0.5
        elif action == 14:
            self.car_controls.throttle = 0
            self.car_controls.brake = 0
            self.car_controls.steering = 0.25
        else:
            self.car_controls.throttle = 0
            self.car_controls.brake = 0
            self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        time.sleep(3)

    def _do_cts_action(self, action):
        self.car_controls.brake = 0
        self.car_controls.throttle = 1
        # print('throttle_val = ', action[0])
        # print('brake_val = ', action[1])
        # print('steering_val = ', action[2])

        # throttle, brake 동시에 0보다 클 경우 차가 멈춤
        if action[0] >= action[1]:
            self.car_controls.throttle = float(action[0])
            self.car_controls.brake = float(0)
            self.car_controls.steering = float(action[2])
            # print('throttle_val = ', self.car_controls.throttle)
        else:
            self.car_controls.throttle = float(0)
            self.car_controls.brake = float(action[1])
            self.car_controls.steering = float(action[2])
            # print('brake_val = ', self.car_controls.brake)

        # print('steering_val = ', action[2])

        self.car.setCarControls(self.car_controls)
        time.sleep(3)


    def transform_obs(self, response):
        img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (response.height, response.width))

        from PIL import Image

        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):
        MAX_SPEED = 300
        MIN_SPEED = 10
        THRESH_DIST = 3.5
        BETA = 3

        pts = [
            np.array([x, y, 0])
            for x, y in [
                (0, -1), (130, -1), (130, 125), (0, 125),
                (0, -1), (130, -1), (130, -128), (0, -128),
                (0, -1),
            ]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            dist = min(
                dist,
                np.linalg.norm(
                    np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                )
                / np.linalg.norm(pts[i] - pts[i + 1]),
            )

        # print(dist)
        if dist > THRESH_DIST:
            reward = -3
        else:
            driving_end_time = time.time()
            reward_dist = 2 * (math.exp(-BETA * dist) - 0.5)
            reward_speed = (
                (self.car_state.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            ) - 0.5
            reward_time = (driving_end_time - self.driving_start_time) / 3
            print('reward_dist:', reward_dist)
            reward = reward_dist + reward_speed + reward_time
            if self.car_state.speed <= 1:
                if self.car_controls.throttle == 0:
                    reward = -2
                    print('stopped')
            print('reward_time:', reward_time)

        done = 0
        if reward < -1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed <= 1:
                done = 1
        if self.state["collision"]:
            reward = -3
            print('collided')
            done = 1

        print('reward:',reward)

        return reward, done

    def step(self, action):
        # for continuous action_space
        # action = softmax(action) # if descrete, ignore this
        # action = int(np.random.choice(16, 1, p=action)) # if descrete, ignore this
        
        # cts_action
        # action = np.random.choice(0,

        # for discrete action
        # self._do_action(action)

        # for cts action
        self._do_cts_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action(1)
        return self._get_obs()
