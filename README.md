# GridWorld

The "hello world" of reinforcement learning in my opinion. So why make this? I want to be able to provide more advanced environments for machine learning agents, and Unity offers great flexibility for that. This is more of an example of how to create a Unity environment from scratch and interfact with it directly using Python.

- [Installation](#installation)
  - [Installing PyTorch](#installing-pytorch)
  - [Remaining Installation](#remaining-installation)
- [Understanding Config](#understanding-config)
  - [DQN](#dqn)
  - [Experience Replay](#experience-replay)
  - [Stats](#stats)
  - [Game](#game)
- [Command Line Arguments](#command-line-arguments)
  - [Specifying Config](#specifying-config)
  - [Training](#training)
  - [Testing](#testing)
  - [Saving Checkpoints](#saving-checkpoints)
  - [Loading Checkpoints](#loading-checkpoints)
  - [Saving stats](#saving-stats)
  - [Examples](#examples)


This uses a custom double DQN: one for local and one for a target network. This agent learns based off raw pixels on the screen.
## Installation

This is split into two sections. One is dedicated to installing PyTorch, while the other installs the remaining dependencies.

### Installing PyTorch

I'm not making this part of the `requirements.txt` because there might be different ways you want to install PyTorch. Take a look at https://pytorch.org/.</br>
If you want this to run with CUDA, you also need to install proper CUDA dependencies. PyTorch does not ship with CUDA. A list of CUDA toolkits can be found [here](https://developer.nvidia.com/cuda-toolkit-archive). Make sure you toolkit version matches whta you download for PyTorch.

But what about TensorFlow? No.

### Remaining Installation

In order to install everything else, just run `pip install -r requirements.txt`.

## Understanding Config

The config file specifies the behavior of the game within Unity along with behavior within Python.

### DQN

Specific to controlling attributes related to the DQN.

<b>loss</b> - The loss function to use. Defaults to `mse_loss`. Can be any `_loss` function found [here](https://pytorch.org/docs/stable/nn.functional.html).<br>
<b>optimizer</b> - The optimizer to use. Defaults to `Adam`. Currently only supports default hyperparameters to the optimizer. Must be found [here](https://pytorch.org/docs/stable/optim.html). <br>
<b>device</b> - Device to run the DQN on. This will attempt to run it on `cuda:0` if available and use `cpu` otherwise.<br>
<b>optimizer</b> - Optimizer to use. Defaults to ADAM.<br>
<b>load_from_checkpoint</b> - If you want to start the DQN from a `checkpoint.tar` file, this is where you can specify the path to it.<br>
<b>eps_start</b> - Starting epsilon value. This controls exploration. Defaults to `1.0` and must be within `(0.0, 1.0]`.<br>
<b>eps_end</b> - Ending epsilon value. Defaults to `0.01` and must be within `(0.0, 1.0]`.<br>
<b>eps_decay</b> - Decay rate of epsilon. Defaults to `0.99` and must be within `(0.0, 1.0)`.<br>
<b>tau</b> - Controls the amount of soft update between target and local network. Defaults to `1e-3` and must be larger than `0.0`.
<b>gamma</b> - Discount factor for the target network. Defaults to `0.99` and must be within `(0.0, 1.0)`.
<b>soft_update_every_n_episodes</b> - The frequency to perform a soft update of the local network. Defaults to `4`.

### Experience Replay

Specific to controlling attributes related to experience replay. Experience replay is used within the DQN.

<b>memory_size</b> - Amount of prior experiences to keep track of. Defaults to `10,000`.
<b>batch_size</b> - Number of batches of prior experiences to train on at a time.

### Stats

Specific to controlling attributes related to saving and tracking stats.

<b>save_checkpoint_every_n_episodes</b> - The frequency to save checkpoints of the agent at. Defaults to `None`, i.e. will not save.<br>
<b>NOTE:</b> Checkpoints are around 10MB in size. Keep this in mind when deciding how frequently to save.<br>
<b>sliding_window_average</b> - The window size to keep track of statistics. Defaults to `100`.<br>
<b>save_stats_every_n_episodes</b> - Frequency to save stats at. Defaults to `None`, i.e. will not save.<br>
<b>save_on_shutdown</b> - Whether or not to save the agent on shutdown. Defaults to `True`. Helpful if you kill an agent with `Ctrl + C` but what a recent snapshot, or if your computer crashes.<br>

### Game

Specific to controlling attributes related to the actual Unity game.

<b>num_targets</b> - Number of targets (goals) in the game. Defaults to `2`.<br>
<b>num_fires</b> - Number of fires in the game. Defaults to `4`.<br>
<b>allow_light_source</b> - Whether or not to allow a light source in the game. This ultimately adds some reflection to the game. Defaults to `True`.<br>
<b>step_reward</b> - The reward the agent receives each time step. Defaults to `-0.1`.<br>
<b>target_reward</b> - The reward the agent receives when touching the target. Defaults to `1.0`.<br>
<b>fire_reward</b> - The reward the agent receives when touching fire. Defaults to `-1.0`<br>

## Command Line Arguments

Command line arguments are used in conjuction with the [config](#understanding-config). I wanted to split them into a config section that controls more of the hyperparameters, and a command line section which just does a little additional setup.

### Specifying Config

`-c /path/to/file.config or --config /path/to/file.config` can be used to specify the config file to load. This will load `default_settings.config` if not specified.

### Training

`--train` can be added to specify you want to enter training mode with an agent. This will load any configs necessary and can also be used with a [checkpoint](#dqn).

### Testing

Want to test your agent? Easy. You can add `--test /path/to/checkpoint.tar` which will load an agent.<br>
<b>NOTE:</b> This will also load the epsilon value associated with that checkpoint. This is to ensure that agents can still behave appropriately with some uncertainty in the environment.

### Saving Checkpoints

This is used in conjunction with config [stats](#stats). If it's not set in the config, then adding this argument won't do anything.<br>
If it is set, specify `save-checkpoint /path/to/folder` which you would like to save checkpoints to<br>
<b>NOTE:</b>The specified directory must not exist. It will be created.

### Loading Checkpoints

There may be a need to load from a checkpoint if your computer crashes or you wish to continue training from a certain point in time. For this you can specify `--load-checkpoint /path/to/checkpoint.tar`.

### Saving Stats

This is only needed if you have specified a frequency to save [stats](#stats). You can give a new location with `--save-stats /path/to/stats_to_create.csv`.

### Running Within Unity

There may be a time where you want to tweak behavior of the game without rebuilding. It's easy to do if you run the game within Unity. By specifying `--run-in-unity` you can do that. Have the Unity environment up and then you will be prompted to press `play` within Unity once Python has created the environment.

### Changing Speed of the Environment

It's possible that you might want to run the game within Unity and slow it down. By default the training happens as quickly as possible, but when you are testing, you may want to slow the actual game speed to see what is going on. You can specify `--time-between-decisions 0.5` and there will be 0.5 seconds between each decision that is made (roughly). This works both in Unity and within the prebuild binaries.

### Examples

`grid_world.py --train` will begin training with the default config file.<br>
`grid_world.py --test "C:\users\chris\documents\checkpoints\checkpoint_20000.tar"` will test the agent loaded from that checkpoint with the default config file.
`grid_world.py --train -c "C:\users\chris\documents\custom_setting.config" --load-checkpoint "C:\users\chris\documents\checkpoints\checkpoint_20000.tar"` will begin training from the specified checkpoint and using the custom config file.<br>
`grid_world.py --train -c "C:\users\chris\documents\custom_setting.config" --save-checkpoint "C:\users\chris\documents\new_checkpoints" --save-stats "C:\users\chris\documents\custom_setting.csv"` will begin training with a custom config file, saving checkpoints to a folder which will be created, and saving stats under a file which will also be created.<br>
`grid_world.py --train --run-in-unity` will begin training with all default parameters and will run within Unity.<br>
`grid_world.py --test "C:\users\chris\documents\checkpoints\checkpoint_20000.tar" --run-in-unity` will begin testing an saved checkpoint within Unity.<br>
