import os
from typing import NamedTuple, Optional
import numpy as np
import torch
import argparse
import math
from colorama import Fore
from datetime import datetime
from mlagents_envs.environment import UnityEnvironment, BaseEnv
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
from collections import deque
import csv
from dqn import QNetwork, Experience, ExperienceReplayBuffer, DQNAgent, save_checkpoint, load_checkpoint
from config import Config


def get_environment(config: Config) -> BaseEnv:
    channel = EnvironmentParametersChannel()
    file_name = None
    if config.RuntimeArgs.run_in_unity:
        file_name = None
        print(Fore.CYAN + "Environment set. Press play within Unity" + Fore.RESET)
    elif os.name == 'nt':
        file_name='Build/GridWorld.exe'

    # Load
    env = UnityEnvironment(file_name=file_name, side_channels=[channel])        

    # Set the channel environment accordingly
    channel.set_float_parameter("num_targets", config.Game.num_targets)
    channel.set_float_parameter("num_fires", config.Game.num_fires)
    allow_light_source = 1.0 if config.Game.allow_light_source else 0.0
    channel.set_float_parameter("allow_light_source", allow_light_source)
    channel.set_float_parameter("step_reward", config.Game.step_reward)
    channel.set_float_parameter("target_reward", config.Game.target_reward)
    channel.set_float_parameter("fire_reward", config.Game.fire_reward)
    channel.set_float_parameter("max_steps", config.Game.max_steps)
    channel.set_float_parameter("time_between_decisions", config.RuntimeArgs.time_between_decisions)

    return env

def run_agent(config: Config):
    env = get_environment(config)
    # For any writing we have to do of stats
    csv_writer = None
    f = None

    env.reset()
    # Get Behavior Spec
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    # Get num actions.
    # There is only one action branch, i.e. direction that it can move (up, down, left, right, or stay)
    num_actions = spec.discrete_action_branches[0]

    # Initialize the agent, episode, eps, etc. and determine whether or not we are loading from a checkpoint
    # Start with the defaults
    episode = 1
    eps = config.DQN.eps_start
    eps_end = config.DQN.eps_end
    eps_decay = config.DQN.eps_decay
    # Create agent
    agent = DQNAgent(spec.observation_shapes[0], num_actions, config)
    # Override from --test checkpoint.
    if config.RuntimeArgs.test != None:
        episode, eps, eps_end, eps_decay = load_checkpoint(agent, config.RuntimeArgs.test)
        # It either was forcefully stopped or saved gracefully.
        # In either case eps and episode are not changed until after the saving of the agent, so change it now
        eps = max(eps_end, eps * eps_decay)
        episode += 1
    # Override from loading from checkpoint
    elif config.DQN.load_from_checkpoint != None:
        episode, eps, eps_end, eps_decay = load_checkpoint(agent, config.DQN.load_from_checkpoint)
        # It either was forcefully stopped or saved gracefully.
        # In either case eps and episode are not changed until after the saving of the agent, so change it now
        eps = max(eps_end, eps * eps_decay)
        episode += 1

    # Initialize stats
    maxlen = config.Stats.sliding_window_average
    running_avg_reward = deque(maxlen=maxlen)
    running_avg_wins = deque(maxlen=maxlen)
    chkpt_save = config.Stats.save_checkpoint_every_n_episodes
    should_save_chkpt = bool(config.Stats.save_checkpoint_every_n_episodes) and bool(config.RuntimeArgs.save_checkpoint)
    stats_save = config.Stats.save_stats_every_n_episodes
    should_save_stats = bool(config.Stats.save_stats_every_n_episodes) and bool(config.RuntimeArgs.save_stats)

    # Determine if we save stats
    if should_save_stats:
        f = open(config.RuntimeArgs.save_stats, "w+", newline="")
        csv_writer = csv.writer(f, delimiter=",")
        csv_writer.writerow(["Episode", "Running Average Reward", "Running Average Win"])

    # Are we training or testing
    train = True if config.RuntimeArgs.train else False

    # Continually iterate through episodes
    while True:
        episode_reward = 0.0
        did_win = False
        # Each episode runs until the agent wins or loses.
        # It's easier to put that run in a try-except in case you decide to Ctrl+C out.
        # In that case you can properly close the file handles, environment and save the
        # agent if needed.
        try:
            episode_reward, did_win = _run_agent_one_ep(env, agent, config, eps, behavior_name, train)
        except:
            env.close()
            if config.Stats.save_on_shutdown:
                time = datetime.now().strftime("%d-%m-%y-%H.%M.%S")
                path = os.path.join(os.getcwd(), f"abrupt_shutdown-{time}.tar")
                save_checkpoint(agent, episode, eps, eps_end, eps_decay, path)
            if csv_writer != None:
                f.close()
        
        # Track stats
        running_avg_reward.appendleft(episode_reward)
        running_avg_wins.appendleft(1 if did_win else 0)

        if episode < maxlen:
            print(f"Episode {episode}, reward: {episode_reward}")
        elif episode >= maxlen and episode % 20 == 0:
            print(f"Window Average <episode {episode}>:\n\t" +\
                  f"Reward: {sum(running_avg_reward) / len(running_avg_reward)}\n\t" + \
                  f"Wins: {sum(running_avg_wins) / len(running_avg_wins)}")

        # Do we need to save checkpoint?
        if should_save_chkpt and episode % chkpt_save == 0:
            save_checkpoint(agent, episode, 
                            eps, eps_end, eps_decay,
                            os.path.join(config.RuntimeArgs.save_checkpoint, f"checkpoint_ep{episode}.tar"))
        # Any stats to save?
        if should_save_stats and episode % stats_save == 0 and len(running_avg_reward) == maxlen:
            avg_reward = sum(running_avg_reward) / len(running_avg_reward)
            avg_win = sum(running_avg_wins) / len(running_avg_wins)
            csv_writer.writerow([episode, avg_reward, avg_win])

        # Decay epsilon and increment episode
        eps = max(eps_end, eps * eps_decay)
        episode += 1

def _run_agent_one_ep(env: BaseEnv, agent: DQNAgent, config: Config,
                        eps: float, behavior_name: str, train: Optional[bool] = True):
    # Get a starting state
    env.reset()

    decision_steps, terminal_steps = env.get_steps(behavior_name)
    state = decision_steps.obs[0]

    agent_id = decision_steps.agent_id[0]
    done = False
    did_win = False
    episode_reward = 0.0
    import time
    while not done:
        reward = 0.0
        # Get and perform an action
        action = agent.act(decision_steps.obs[0], eps)
        env.set_actions(behavior_name, np.expand_dims(action, 0).reshape(-1, 1))
        env.step()

        decision_steps, terminal_steps = env.get_steps(behavior_name)
        # Determine S', R, Done
        next_state = None
        if agent_id in decision_steps:
            reward += decision_steps.reward[0]
            next_state = decision_steps.obs[0]
        if agent_id in terminal_steps:
            terminal_reward = terminal_steps.reward[0]
            # Add win/loss
            did_win = True if math.isclose(terminal_reward, 1.0) else False
            reward += terminal_reward
            next_state = terminal_steps.obs[0]
            done = True

        assert next_state is not None, f"next_state cannot be None. Agent {agent_id} did not appear in decision or terminal steps"

        if train:
            # Learn from (S, A, R, S')
            experience = Experience(state, action, reward, next_state, done)
            agent.step(experience)

        # Set new state
        state = next_state

        episode_reward += reward
    
    return (episode_reward, did_win)