from pathlib import Path

import gym
import numpy as np
import matplotlib.pyplot as plt
# global figindex 
# figindex = 0

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
# from ray.rllib.models.tf.attention_net import GTrXLNet


from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from minxuan.observation_adapter import observation_adapter

tf = try_import_tf()

# ==================================================
# Discrete Action Space
# "keep_lane", "slow_down", "change_lane_left", "change_lane_right"
# Action space should match the input to the action_adapter(..) function below.
# ==================================================
ACTION_SPACE = gym.spaces.Discrete(4)
ACTIONS = [
    "keep_lane",
    "slow_down",
    "change_lane_left",
    "change_lane_right",
]

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),

        # To make car learn to slow down, overtake or dodge
        # distance to the closest car in each lane
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # time to collide to the closest car in each lane
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),

        # treated as a boolean
        "ego_will_crash": gym.spaces.Box(low=0, high=1, dtype=np.int8, shape=(1,)),
        "speed_of_closest": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "proximity": gym.spaces.Box(low=-1e10, high=1e10, shape=(6,)),
        "headings_of_cars": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center


def reward_adapter(env_obs, env_reward):
    # lane centering
    obs = observation_adapter(env_obs)
    center_penalty = -np.abs(obs["distance_from_center"])

    # penalize flip occurences (taking into account that the vehicle spawns in the air)
    flip_penalty = 0
    if (
            env_obs.ego_vehicle_state.speed >= 7
            and env_obs.ego_vehicle_state.position[2] > 0.85
    ):
        flip_penalty = -2 * env_obs.ego_vehicle_state.speed

    # penalise sharp turns done at high speeds
    steering_penalty = 0
    if env_obs.ego_vehicle_state.speed > 15:
        steering_penalty = -pow(
            (env_obs.ego_vehicle_state.speed - 15)
            / 5
            * (env_obs.ego_vehicle_state.steering)
            * 45
            / 4,
            2,
        )

    # penalize close proximity to other cars
    crash_penalty = -5 if bool(obs["ego_will_crash"]) else 0

    # penalize violation of traffic rules
    violation_penalty = 0 # if np.abs(obs["distance_from_center"]-1)< 1e-4 else 0
    
    #test = env_obs.occupancy_grid_map[1]
    #print(test.shape)
    #print(np.unique(test.reshape([256,256])))
    
    #global figindex
    #img = env_obs.top_down_rgb[1]
    #figindex += 1
    #name = "figure/fig%d.png" % figindex 
    #plt.imsave(name, img)

    # preferred speed = 15 m/s
    speed_reward = 5 * np.sum(15 - np.abs(np.array([1.0 * env_obs.ego_vehicle_state.speed]) - 15))

    total_reward = np.sum([1.0 * env_reward, ])
    total_penalty = np.sum([0.1 * center_penalty, 1 * steering_penalty, crash_penalty, violation_penalty])

    if flip_penalty != 0:
        return float((-total_reward + speed_reward + total_penalty) / 200.0)
    else:
        return float((total_reward + speed_reward + total_penalty) / 200.0)


def action_adapter(model_action):
    assert model_action in [0, 1, 2, 3]
    return ACTIONS[model_action]


#    throttle, brake, steering = model_action
#    return np.array([throttle, brake, steering * np.pi * 0.25])


class TrainingModel(FullyConnectedNetwork):
    NAME = "FullyConnectedNetwork"


ModelCatalog.register_custom_model(TrainingModel.NAME, TrainingModel)


class RLLibTFSavedModelAgent(Agent):
    def __init__(self, path_to_model, observation_space):
        path_to_model = str(path_to_model)  # might be a str or a Path, normalize to str
        self._prep = ModelCatalog.get_preprocessor_for_space(observation_space)
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        tf.compat.v1.saved_model.load(
            self._sess, export_dir=path_to_model, tags=["serve"]
        )
        self._output_node = self._sess.graph.get_tensor_by_name("default_policy/add:0")
        self._input_node = self._sess.graph.get_tensor_by_name(
            "default_policy/observation:0"
        )

    def __del__(self):
        self._sess.close()

    def act(self, obs):
        obs = self._prep.transform(obs)
        # These tensor names were found by inspecting the trained model
        res = self._sess.run(self._output_node, feed_dict={self._input_node: [obs]})
        action = res[0]
        return action


rllib_agent = {
    "agent_spec": AgentSpec(
        interface=AgentInterface.from_type(AgentType.Full, action=ActionSpaceType.Lane, max_episode_steps=500),
        agent_params={
            "path_to_model": Path(__file__).resolve().parent / "model",
            "observation_space": OBSERVATION_SPACE,
        },
        agent_builder=RLLibTFSavedModelAgent,
        observation_adapter=observation_adapter,
        reward_adapter=reward_adapter,
        action_adapter=action_adapter,
    ),
    "observation_space": OBSERVATION_SPACE,
    "action_space": ACTION_SPACE,
}
