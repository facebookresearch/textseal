# Copyright (c) Meta Platforms, Inc. and affiliates.


import argparse
import logging
import json

from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import Type, TypeVar

logger = logging.getLogger()

T = TypeVar("T")

def set_struct_recursively(cfg, strict: bool = True):
    # Set struct mode for the current level
    OmegaConf.set_struct(cfg, strict)

    # Traverse through nested dictionaries and lists
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if isinstance(value, (DictConfig, ListConfig)):
                set_struct_recursively(value, strict)
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                set_struct_recursively(item, strict)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def dataclass_to_dict(dataclass_instance: T) -> dict:
    """
    Converts a dataclass instance to a dictionary, recursively for nested structures.
    """
    if isinstance(dataclass_instance, dict):
        return dataclass_instance

    return OmegaConf.to_container(
        OmegaConf.structured(dataclass_instance), resolve=True
    )


def load_config_file(config_file, dataclass_cls: Type[T]) -> T:
    config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    return dataclass_from_dict(dataclass_cls, config)


def dump_config(config, path, log_config=True):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        if log_config:
            logger.info("Using the following config for this run:")
            logger.info(yaml_dump)
        f.write(yaml_dump)


def convert_value(v):
    # First try to parse as JSON (for lists, dicts, etc.)
    if v.startswith(('[', '{')):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass
    
    # Then try other type conversions
    for cast in (str2bool, int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    return v


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 'yes'):
        return True
    elif v.lower() in ('false', 'no'):
        return False
    raise ValueError(f"Invalid boolean value: {v}")


def set_nested(config, keys, value):
    """Set nested key in a dict given a list of keys."""
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = value


def cfg_from_cli():
    parser = argparse.ArgumentParser(description="Build CLI config overrides")
    args, unk_args = parser.parse_known_args()

    def parse_unknown_args(arg_list):
        config = {}
        key = None
        for arg in arg_list:
            # If the argument starts with '--', it's a key.
            if arg.startswith('--'):
                key = arg.lstrip('--')
            # Otherwise, it's a value.
            else:
                if key is not None:
                    keys = key.split('.')
                    # If there are more than one value for this key, collect all remaining values
                    values = [arg]
                    next_idx = unk_args.index(arg) + 1
                    while next_idx < len(arg_list) and not arg_list[next_idx].startswith('--'):
                        values.append(arg_list[next_idx])
                        next_idx += 1
                    # Convert each value and create list if multiple values
                    converted_values = [convert_value(v) for v in values]
                    val = converted_values[0] if len(converted_values) == 1 else converted_values
                    set_nested(config, keys, val)
                    key = None
        return config

    overrides = parse_unknown_args(unk_args)
    return overrides
