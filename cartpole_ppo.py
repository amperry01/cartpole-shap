from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.test_utils import add_rllib_example_script_args
import os

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .training(
        lr=0.0003,
        num_epochs=6,
        vf_loss_coeff=0.01,
    )
    .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[32],
            fcnet_activation="linear",
            vf_share_layers=True,
        ),
    )
)

def train_and_save_model():
    # init PPO algo
    algo=PPO(config=config)

    # define checkpoint directory
    base_checkpoint_dir = "/home/aperry/git_repos/xcarly-alyssa/cartpole_shap/checkpoints/"
    os.makedirs(base_checkpoint_dir, exist_ok=True)

    # function to get the next checkpoint directory
    def get_next_checkpoint_dir(base_dir):
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        next_checkpoint_num = len(existing_dirs) + 1
        next_checkpoint_dir = os.path.join(base_dir, f"checkpoint_{next_checkpoint_num:06d}")
        return next_checkpoint_dir
    
    # training loop
    num_iterations = 100
    for i in range(num_iterations):
        result = algo.train()
        if 'episode_return_mean' in result['env_runners']:
            print(f"Iteration {i}: reward = {result['env_runners']['episode_return_mean']}")
        else:
            print(f"Iteration {i}: 'episode_return_mean' key not found in the result")

        # save model checkpoints periodically
        if (i + 1) % 10 == 0:
            checkpoint_dir = get_next_checkpoint_dir(base_checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Creating checkpoint directory at: {checkpoint_dir}")
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"Model checkpoint saved at: {checkpoint_path}")

    return algo

if __name__ == "__main__":
    algo = train_and_save_model()
