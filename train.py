from src.coach import Coach, CoachConfig
from src.utils import ParserBuilder
from src.env import MujocoEnv

import wandb

def main(args):
    # load environment
    env = MujocoEnv("cartpole", "swingup")

    wandb.init(project=args.project_name, dir=args.save_directory, config=args)
    wandb.config.update(args)

    coach_config = CoachConfig.from_args(args)

    coach = Coach(
        env,
        config=coach_config
    )
    coach.run()

if __name__ == "__main__":
    parser = ParserBuilder().add_dataclass(
        CoachConfig()
    ).add_argument(
        name="project_name",
        default="simulator",
        help="wandb project name"
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