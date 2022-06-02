import setup_path
import gym
import airgym
import time
import tensorflow as tf
import numpy as np

from stable_baselines3 import DQN, A2C, DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# Check if gpu is used or not
print(tf.__version__)
print()

print(tf.config.list_physical_devices('GPU'))
print()

# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-car-sample-v0",
                ip_address="127.0.0.1",
                image_shape=(84, 84, 1),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

# print('wrapped env!')

# Initialize RL algorithm type and parameters - DQN
#model_dqn = DQN(
#    "CnnPolicy",
#    env,
#    learning_rate=0.0005,
#    verbose=1,
#    batch_size=32,
#    train_freq=4,
#    target_update_interval=10000,
#    learning_starts=1000,
#    buffer_size=400000,
#    max_grad_norm=10,
#    exploration_fraction=0.1,
#    exploration_initial_eps=0.8,
#    exploration_final_eps=0.01,
#    device="auto",
#    tensorboard_log="./tb_logs/",
#)

# Initialize RL algorithm type and parameters - A2C
#model_a2c = A2C(
#    "CnnPolicy",
#    env,
#    learning_rate=0.0005,
#    verbose=1,
#    ent_coef=0.5,
#    device="auto",
#    tensorboard_log="./tb_logs/",
#)

n_actions = env.action_space.shape[-1]
mean_arr = np.zeros(n_actions)
mean_arr += [0.5, 0.5, 0] # throttle : 0~1, brake : 0~1, steering : -1~1
mean_arr = list(map(float, mean_arr)) # change datatype to Python float
print(mean_arr)
# action_noise = OrnsteinUhlenbeckActionNoise(mean=mean_arr, sigma=float(1.5) * np.ones(n_actions))
action_noise = NormalActionNoise(mean=mean_arr, sigma=float(0.5) * np.ones(n_actions))

# Initialize RL algorithm type and parameters - DDPG
model_ddpg = DDPG(
    "CnnPolicy",
    env,
    learning_rate=0.001,
    tau=0.005,
    gamma=0.9,
    verbose=1,
    batch_size=64,
    action_noise=action_noise,
    buffer_size=400000,
    device="auto",
    tensorboard_log="./tb_logs/",
)

# print('made model!')

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model_ddpg.learn(
    total_timesteps=5e5, tb_log_name="ddpg_airsim_car_run_" + str(time.time()), **kwargs
)

# Save policy weights
model_ddpg.save("dqn_airsim_car_policy")
