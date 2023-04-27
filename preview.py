from dm_control import viewer
import os.path

from src.utils import ParserBuilder
from src.env import MujocoEnv

import torch

def main(args):
    agent = torch.load(os.path.join(args.save_directory, "agent.pth"))
    env = MujocoEnv(args.domain_name, args.task_name)
    env.reset()
    agent.eval()

    reward = 0
    episode_history = []
    timestep = env.reset()
    episode_history.append((timestep, None))

    while not timestep.last():
        action = agent(timestep.observation)
        print(action)
        timestep = env.step(action)
        reward += timestep.reward or 0.
    print(reward)

    viewer.launch(env.env, policy=lambda observation: agent(observation.observation["observations"]))

if __name__ == "__main__":
    parser = ParserBuilder().add_argument(
        name="save_directory",
        help="directory containing the agent"
    ).add_argument(
        name="domain_name",
        default="cartpole",
        help="mujoco domain name"
    ).add_argument(
        name="task_name",
        default="swingup",
        help="mujoco task name"
    ).build()
    args = parser.parse_args()
    main(args)