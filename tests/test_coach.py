from glob import glob
from copy import copy
import os.path
import torch

from typing import Any, Callable, List

from src.coach import Coach, CoachConfig
from src.utils import TrainingConfig
from src.env import MujocoEnv

from tests.config import cleanup, requires_cleanup, SAVE_DIR

env = MujocoEnv("cartpole", "swingup")

coach = Coach(
    env,
    config=CoachConfig(
        save_directory=SAVE_DIR,
        num_iterations=2,
        sub_directories="test-{iteration:02}",
        best_checkpoint_path="best",
        agent_env_experiences=2,
        agent_sim_experiences=3,
        evaluation_runs=1,
        simulator_epochs=4,
        training_config=TrainingConfig(
            iterations=1,
            batch_size=16,
            lr=1e-2,
            optimizer_type="AdamW",
        )
    )
)

def mock_learn(configs: List[TrainingConfig]) -> Callable[[Any], None]:
    def learn(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, TrainingConfig):
                configs.append(copy(arg))
        for v in kwargs.values():
            if isinstance(v, TrainingConfig):
                configs.append(copy(v))
    return learn


@requires_cleanup
def test_directory_structure():
    assert coach.save_directory == SAVE_DIR
    assert coach.config.sub_directories == "test-{iteration:02}"
    assert coach.sub_directories == "test-{iteration:02}"
    coach.run()
    assert os.path.exists(SAVE_DIR)
    for directory in ("test-00", "test-01", "test-02", "test-best"):
        assert os.path.exists(os.path.join(SAVE_DIR, directory))
        assert len(glob(os.path.join(SAVE_DIR, directory, "*"))) > 0
    assert not os.path.exists(os.path.join(SAVE_DIR, "test-03"))

@requires_cleanup
def test_agent_training_config(monkeypatch):
    configs: List[TrainingConfig] = []
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)
    monkeypatch.setattr(coach.agent, "learn", mock_learn(configs))
    coach.run()
    assert len(configs) == 2
    for config in configs:
        assert config.iterations == 3
        assert config.batch_size == 16
        assert config.lr == 1e-2
        assert config.optimizer_type == "AdamW"

@requires_cleanup
def test_simulator_training_config(monkeypatch):
    configs: List[TrainingConfig] = []
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: None)
    fit = coach.simulator.fit
    monkeypatch.setattr(
        coach.simulator,
        "fit",
        lambda *args, **kwargs: mock_learn(configs)(*args, **kwargs) or fit(*args, **kwargs)
    )
    coach.run()
    assert len(configs) == 2
    for config in configs:
        assert config.iterations == 4
        assert config.batch_size == 16
        assert config.lr == 1e-2
        assert config.optimizer_type == "AdamW"