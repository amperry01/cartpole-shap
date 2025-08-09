import numpy as np
from pathlib import Path
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
)

def get_latest_checkpoint(checkpoint_path: str) -> str:
    import os
    import glob

    checkpoint_path = os.path.expanduser(checkpoint_path)
    dir_str = os.path.join(checkpoint_path, "checkpoint_*")

    # find all matching checkpoint directories
    all_checkpoints = glob.glob(dir_str)

    if not all_checkpoints:
        raise FileNotFoundError("No checkpoint files found in directory")
    
    # find most recent checkpoint by modification time
    most_recent_checkpoint = max(all_checkpoints, key=os.path.getmtime)

    return most_recent_checkpoint

def load_components_from_checkpoint(path) -> RLModule:
    rl_module = RLModule.from_checkpoint(
        Path(path, COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE)
    )

    return rl_module

def summarize_episode(episode: SingleAgentEpisode):
    print(f"Environment Steps: {episode.env_steps()}")
    print(f"Total Reward: {sum(episode.rewards)}")
    print(f"Terminated: {episode.is_terminated}")
    print(f"Truncated: {episode.is_truncated}")

def run_episode(rl_module: RLModule):
    # extract single-agent module from wrapped multi-agent rl module
    single_module = rl_module["default_policy"]
    
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    episode = SingleAgentEpisode(observations=[obs])

    # attempt to get action distribution class for inference
    try:
        action_dist_class = rl_module.get_inference_action_dist_cls()
    except NotImplementedError:
        action_dist_class = TorchCategorical

    # track true inputs for SHAP
    shap_observations = []

    while not episode.is_done:
        shap_observations.append(obs) # store state BEFORE action 

        # prepare input
        input_dict = {"obs": torch.tensor([obs], dtype=torch.float32)}
        output = single_module.forward_inference(input_dict, explore=False)

        # extract logits and create action distribution
        action_dist_inputs = output['action_dist_inputs']
        action_dist = action_dist_class.from_logits(action_dist_inputs)

        # sample action
        action = action_dist.sample()[0].item() # using item over numpy because I just need a single discrete action rather than an array

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode.add_env_step(
            observation=next_obs, 
            action=action, 
            reward=reward, 
            terminated=terminated, 
            truncated=truncated
        )

        obs = next_obs
        env.render()

    env.close()
    return episode, shap_observations

def explain_model(torch_model, observations):
    import shap

    shap.initjs()

    # convert to tensor for SHAP compatibility
    obs_tensor = torch.tensor(observations, dtype=torch.float32)
    print(f"obs_tensor: {obs_tensor.shape}") # (500, 4) aka (observations, num_features)
    print(f"single logit model obs_tensor: {torch_model(obs_tensor[:100]).shape}") # (100, 1) aka (batch_size, single_logit_output)

    explainer = shap.DeepExplainer(torch_model, obs_tensor[:100])
    shap_values = explainer.shap_values(obs_tensor[:100]) # outputs (batch_size, num_features, num_outputs)
    shap_values = np.array(shap_values).squeeze(-1) # outputs (batch_size, num_features), which shap expects for plotting
    print(f"shap_values shape: {shap_values.shape}") # (100, 4) aka (batch_size, num_features)

    shap.summary_plot(
        shap_values, 
        obs_tensor[:100],
        feature_names=["Cart Position", "Cart Velocity", "Pole Angle", "Pole Velocity at Tip"]  
    )

    plt.show()

    # shap.plots._waterfall.waterfall_legacy(
    #     explainer.expected_value[0],    # expected value
    #     shap_values[0][0],              # SHAP values for first observation
    #     observations[0]                 # first observation
    # )

    # plt.show()

class PolicyWrapper(nn.Module):
    def __init__(self, policy_module):
        super().__init__()
        self.policy_module = policy_module

    def forward(self, x):
        # RLlib policies expect input like: {"obs": tensor}
            # ensure input is a batch of obs tensors
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        out = self.policy_module({"obs": x})
        return out["action_dist_inputs"]
    
class LogitSelectorModel(nn.Module):
    """ SHAP's DeepExplainer expects a 1D output per sample, not multiple values per input (e.g. [num_samples, 1])
        so, here we wrap the model to return only one logit per sample (e.g. left logit or right logit)
    """
    def __init__(self, model, index=0): 
        super().__init__()
        self.model = model
        self.index = index
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32) # shape: [batch]
        return self.model(x)[:, self.index].unsqueeze(1) # shape: [batch, 1]

if __name__ == "__main__":
    
    checkpoint_path = "/home/aperry/git_repos/xcarly-alyssa/cartpole_shap/checkpoints"
    latest_checkpoint = get_latest_checkpoint(checkpoint_path)
    print(f"Most recent checkpoint: {latest_checkpoint}")

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

    # load components
    rl_module = load_components_from_checkpoint(latest_checkpoint)

    # access TorchPPO policy + wrap it for SHAP compatibility 
    torch_model = rl_module._rl_modules["default_policy"]
    wrapped_model = PolicyWrapper(torch_model)

    # run and summarize episode to collect observations and actions
    episode, shap_obs = run_episode(rl_module)
    summarize_episode(episode)
    breakpoint()
    # extract observations for SHAP
    observations = np.array(shap_obs)

    # 1D ouput models, will explain each action/logit separately 
    left_model = LogitSelectorModel(wrapped_model, index=0)
    right_model = LogitSelectorModel(wrapped_model, index=1)

    # explain using SHAP
    explain_model(left_model, observations)
    explain_model(right_model, observations)
