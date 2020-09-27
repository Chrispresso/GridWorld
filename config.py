import configparser
import os
import torch
import numpy as np
from typing import Any, Dict, Optional
import torch.nn.functional as F
import torch


class DefaultType:
    def __init__(self, types: Any, default_value: Any, options: Optional[Any] = None):
        self.types = types
        self.default_value = default_value
        self.options = options

# A mapping from parameters name -> final type
_params = {
    # DQN specific
    'DQN': {
        'loss': DefaultType(str, 'mse_loss', lambda x: 'loss' in x and x in dir(F)),
        'optimizer': DefaultType(str, 'Adam', lambda x: x in dir(torch.optim)),
        'device': DefaultType(str, 'cuda:0' if torch.cuda.is_available() else 'cpu'),
        'load_from_checkpoint': DefaultType(str, None),
        'eps_start': DefaultType(float, 1.0, lambda x: x > 0.0 and x <= 1.0),
        'eps_end': DefaultType(float, 0.01, lambda x: x > 0.0 and x <= 1.0),
        'eps_decay': DefaultType(float, 0.99, lambda x: x > 0.0 and x < 1.0),
        'tau': DefaultType(float, 1e-3, lambda x: x > 0),
        'gamma': DefaultType(float, 0.99, lambda x: x > 0.0 and x < 1.0),
        'soft_update_every_n_episodes': DefaultType(int, 4, lambda x: x > 0)
    },
    # Experience replay
    'ExperienceReplay': {
        'memory_size': DefaultType(int, 10_000),
        'batch_size': DefaultType(int, 64)
    },
    # Stats
    'Stats': {
        # 'save_checkpoint_every_n_episodes': DefaultType(int, 10_000),
        'save_checkpoint_every_n_episodes': DefaultType(int, None),
        'sliding_window_average': DefaultType(int, 100),
        # 'save_stats_every_n_episodes': DefaultType(int, 1),
        'save_stats_every_n_episodes': DefaultType(int, None),
        'save_on_shutdown': DefaultType(bool, True)
    },
    # Game
    'Game': {
        'num_targets': DefaultType(int, 2),
        'num_fires': DefaultType(int, 4),
        'allow_light_source': DefaultType(bool, True),
        'step_reward': DefaultType(float, -0.1),
        'target_reward': DefaultType(float, 1.0),
        'fire_reward': DefaultType(float, -1.0),
        'max_steps': DefaultType(int, 250),
    }
}

class DotNotation(object):
    def __init__(self, d: Dict[Any, Any]):
        for k in d:
            # If the key is another dictionary, keep going
            if isinstance(d[k], dict):
                self.__dict__[k] = DotNotation(d[k])
            # If it's a list or tuple then check to see if any element is a dictionary
            elif isinstance(d[k], (list, tuple)):
                l = []
                for v in d[k]:
                    if isinstance(v, dict):
                        l.append(DotNotation(v))
                    else:
                        l.append(v)
                self.__dict__[k] = l
            else:
                self.__dict__[k] = d[k]
    
    def __getitem__(self, name) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]

    def __str__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return str(self)


class Config(object):
    def __init__(self,
                 filename: str
                 ):
        self.filename = filename
        
        if not os.path.isfile(self.filename):
            raise Exception('No file found named "{}"'.format(self.filename))

        with open(self.filename) as f:
            self._config_text_file = f.read()

        self._config = configparser.ConfigParser(inline_comment_prefixes='#')
        self._config.read(self.filename)

        self._verify_sections()
        self._create_dict_from_config()
        self._set_dict_types()
        dot_notation = DotNotation(self._config_dict)
        self.__dict__.update(dot_notation.__dict__)


    def _create_dict_from_config(self) -> None:
        d = {}
        for section in self._config.sections():
            d[section] = {}
            for k, v in self._config[section].items():
                d[section][k] = v

        self._config_dict = d

    def _set_dict_types(self) -> None:
        # defaults
        for section in _params:
            for k, v in _params[section].items():
                try:
                    _type = _params[section][k]
                except:
                    raise Exception('No value "{}" found for section "{}" in _params')
                # Determine if there's something to overwrite the value with
                if section in self._config.sections() and k in self._config[section]:
                    self._parse_type(_type.types, section, k, self._config[section][k])
                # Otherwise if it's default, take that
                elif isinstance(_type, DefaultType):
                    _new_type, default_val = _type.types, _type.default_value
                    try:
                        self._config_dict[section][k]  # Just try to access
                        self._parse_type(_new_type, section, k, v)
                    except:
                        if section not in self._config_dict:
                            self._config_dict[section] = {}
                        self._config_dict[section][k] = default_val

    def _parse_type(self, _type: Any, section: str, k: str, v: Any) -> None:
        # Normally _type will be int, str, float or some type of built-in type.
        # If _type is an instance of a tuple, then we need to split the data
        if isinstance(_type, tuple):
            if len(_type) == 2:
                cast = _type[1]
                v = v.replace('(', '').replace(')', '')  # Remove any parens that might be present 
                self._config_dict[section][k] = tuple(cast(val) for val in v.split(','))
            else:
                raise Exception('Expected a 2 tuple value describing that it is to be parse as a tuple and the type to cast it as')
        elif isinstance(_type, list):
            cast = _type[1]
            v = v.replace('[', '').replace(']', '')  # Remove any brackets that might be present
            self._config_dict[section][k] = [cast(val) for val in v.split(',')]
        elif 'lambda' in v:
            try:
                self._config_dict[section][k] = eval(v)
            except:
                pass
        # Is it a bool?
        elif _type == bool:
            self._config_dict[section][k] = _type(eval(v))
        # Otherwise parse normally
        else:
            self._config_dict[section][k] = _type(v)

        # Ensure that if there are options provided that they match
        if type(self._config_dict[section][k]) != type(lambda : None):
            default = _params[section][k]
            if type(default.options) == type(lambda x: x):
                _v = default.types(v)  # Cast to whatever required type
                if not default.options(_v):
                    raise Exception(f"Option [{section}].{k} has an invalid value.")
            if default.options and self._config_dict[section][k] not in default.options:
                raise Exception(f'Option [{section}].{k} must be one of {default.options}')

    def _verify_sections(self) -> None:
        # Validate sections
        for section in self._config.sections():
            # Make sure the section is allowed
            if section not in _params:
                raise Exception('Section "{}" has no parameters allowed. Please remove this section and run again.'.format(section))

    def _get_reference_from_dict(self, reference: str) -> Any:
        path = reference.split('.')
        d = self._config_dict
        for p in path:
            d = d[p]
        
        assert type(d) in (tuple, int, float, bool, str, list)
        return d

    def _is_number(self, value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False