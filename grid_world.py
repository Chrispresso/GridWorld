import argparse
import os
from config import Config, DotNotation
from run_agent import run_agent
import colorama
from colorama import Fore, Back


def parse_args_and_init():
    parser = argparse.ArgumentParser(description="GridWorld AI")
    parser.add_argument("-c", "--config", dest="config", default=None, required=False, help="Config file to use")
    parser.add_argument("--train", action="store_true", default=False, help="If set, enter training mode", required=False)
    parser.add_argument("--test", dest="test", default=None, help="/path/to/checkpoint.tar to test", required=False)
    parser.add_argument("--save-checkpoint", dest="save_checkpoint", default=None, required=False, help="/path/to/folder to save checkpoints")
    parser.add_argument("--load-checkpoint", dest="load_checkpoint", default=None, required=False, help="/path/to/checkpoint.tar to load from")
    parser.add_argument("--save-stats", dest="save_stats", default=None, required=False, help="/path/to/stats_to_create.csv")
    parser.add_argument("--run-in-unity", dest="run_in_unity", default=False, action="store_true", required=False, help="Whether or not to run within the Unity environment rather than a binary game")
    parser.add_argument("--time-between-decisions", dest="time_between_decisions", default=0.0, required=False, help="Time between making decisions in the environment. Will slow down visualization")

    args = parser.parse_args()
    colorama.init()

    # Ensure that only --train or --test is set, not both
    if bool(args.train) and bool(args.test):
        raise Exception("Can only set --train or --test, not both")

    # Gotta do something
    if not (bool(args.train) or bool(args.test)):
        raise Exception("Must set --train or --test")

    # If we load from a checkpoint we need to be in train mode
    if bool(args.load_checkpoint) and (bool(args.train) ^ bool(args.load_checkpoint)):
        raise Exception("Can only load from a checkpoint in --train mode. If you want to test, run --test /path/to/checkpoint.tar")

    # If --config is not specified, just use the default one
    if args.config is None:
        config_path = os.path.join(os.getcwd(), 'default_settings.config')
    else:
        config_path = args.config
    config = Config(config_path)
        
    # Create the folder for checkpoints if it doesn't exist
    if bool(args.save_checkpoint) and not os.path.exists(args.save_checkpoint):
        os.makedirs(args.save_checkpoint)

    # Initialize stats file if needed
    if args.save_stats:
        if os.path.exists(args.save_stats):
            raise Exception("Cannot save stats to an already existing stats file")

    # Add the runtime arguments to config
    config._config_dict['RuntimeArgs'] = {}
    config._config_dict['RuntimeArgs']['train'] = bool(args.train)
    config._config_dict['RuntimeArgs']['test'] = args.test
    config._config_dict['RuntimeArgs']['save_stats'] = args.save_stats
    config._config_dict['RuntimeArgs']['save_checkpoint'] = args.save_checkpoint
    config._config_dict['RuntimeArgs']['run_in_unity'] = bool(args.run_in_unity)
    config._config_dict['RuntimeArgs']['time_between_decisions'] = args.time_between_decisions
    new_dot = DotNotation(config._config_dict)
    config.__dict__.update(new_dot.__dict__)

    # If the user is trying to save the stats but there isn't a frequency to save at, warn
    if bool(args.save_stats) and not bool(config.Stats.save_stats_every_n_episodes):
        print(Fore.YELLOW + "[WARNING] you requested to save stats, but there is no frequency " + \
              "specified under [Stats].save_stats_every_n_episodes in the config file." + \
              Fore.RESET)
    
    # If the user is trying to save checkpoints but there isn't a frequency to save at, warn
    if bool(args.save_checkpoint) and not bool(config.Stats.save_checkpoint_every_n_episodes):
        print(Fore.YELLOW + "[WARNING] you requested to save checkpoints, but there is no frequency " + \
              "specified under [Stats].save_checkpoint_every_n_episodes in the config file." + \
              Fore.RESET)

    # If we plan to save checkpoints, there needs to be a place to save them, just warn
    if not bool(args.save_checkpoint) and bool(config.Stats.save_checkpoint_every_n_episodes):
        print(Fore.YELLOW + "[WARNING] You requested to save checkpoints in the config file, but did not specify --save-checkpoint /path/to/save."+\
              "No checkpoints will be saved for this run."+\
              Fore.RESET)

    # If we plan to save stats, there needs to be a place to save them, just warn
    if not bool(args.save_stats) and bool(config.Stats.save_stats_every_n_episodes):
        print(Fore.YELLOW + "[WARNING] You requested to save stats in he config file, but did not specify --save-stats /path/to/new_file.csv\n"+\
              "No stats will be saved for this run." +\
              Fore.RESET)

    return config
    

if __name__ == "__main__":
    config = parse_args_and_init()
    run_agent(config)