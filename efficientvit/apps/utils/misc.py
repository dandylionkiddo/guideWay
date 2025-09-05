import os

import yaml

__all__ = [
    "parse_with_yaml",
    "parse_unknown_args",
    "partial_update_config",
    "resolve_and_load_config",
    "load_config",
    "dump_config",
]


def parse_with_yaml(config_str: str) -> str | dict:
    try:
        # add space manually for dict
        if "{" in config_str and "}" in config_str and ":" in config_str:
            out_str = config_str.replace(":", ": ")
        else:
            out_str = config_str
        return yaml.safe_load(out_str)
    except ValueError:
        # return raw string if parsing fails
        return config_str


def parse_unknown_args(unknown: list) -> dict:
    """Parse unknown args."""
    index = 0
    parsed_dict = {}
    while index < len(unknown):
        key = unknown[index]
        if not key.startswith("--"):
            index += 1 # Skip non-flag arguments
            continue
        key = key[2:] # Remove --

        # Check if it's a boolean flag or a key-value pair
        if index + 1 < len(unknown) and not unknown[index + 1].startswith("--"):
            # It's a key-value pair
            val = unknown[index + 1]
            parsed_dict[key] = parse_with_yaml(val)
            index += 2
        else:
            # It's a boolean flag (e.g., --eval_only) or last argument
            # We assume boolean flags are handled by argparse directly
            # So, if it's an unknown standalone flag, we ignore it for opt_args
            index += 1
    return parsed_dict


def partial_update_config(config: dict, partial_config: dict) -> dict:
    for key in partial_config:
        

        # If the key exists in config, and both are dictionaries, recurse
        if key in config and isinstance(config[key], dict) and isinstance(partial_config[key], dict):
            partial_update_config(config[key], partial_config[key])
        # If the key exists in config, and config[key] is a list, but partial_config[key] is a dict, overwrite the list with the dict
        elif key in config and isinstance(config[key], list) and isinstance(partial_config[key], dict):
            config[key] = partial_config[key]
        # Otherwise (key not in config, or types don't match, or one is not a dict), just overwrite
        else:
            config[key] = partial_config[key]
    return config


def resolve_and_load_config(path: str, config_name="config.yaml") -> dict:
    path = os.path.realpath(os.path.expanduser(path))
    if os.path.isdir(path):
        config_path = os.path.join(path, config_name)
    else:
        config_path = path
    if os.path.isfile(config_path):
        pass
    else:
        raise Exception(f"Cannot find a valid config at {path}")
    config = load_config(config_path)
    return config


class SafeLoaderWithTuple(yaml.SafeLoader):
    """A yaml safe loader with python tuple loading capabilities."""

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


SafeLoaderWithTuple.add_constructor("tag:yaml.org,2002:python/tuple", SafeLoaderWithTuple.construct_python_tuple)


def load_config(filename: str) -> dict:
    """Load a yaml file."""
    filename = os.path.realpath(os.path.expanduser(filename))
    return yaml.load(open(filename), Loader=SafeLoaderWithTuple)


def dump_config(config: dict, filename: str) -> None:
    """Dump a config file"""
    filename = os.path.realpath(os.path.expanduser(filename))
    yaml.dump(config, open(filename, "w"), sort_keys=False)
