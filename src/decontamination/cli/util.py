import argparse
import collections.abc
import inspect
import os
import re
import sys
import types
import typing
from dataclasses import fields, is_dataclass
from pathlib import Path

import yaml
from typing_extensions import get_type_hints


# --- Custom YAML Loader ---
class IncludeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        try:
            self._root = os.path.dirname(stream.name)
        except AttributeError:
            self._root = os.path.abspath(".")
        super().__init__(stream)


def include_constructor(loader, node):
    filename = os.path.join(loader._root, loader.construct_scalar(node))
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Included YAML file not found: {filename}")
    with open(filename, "r") as f:
        return yaml.load(f, Loader=IncludeLoader)


yaml.add_constructor("!include", include_constructor, IncludeLoader)
# --------------------------


def get_base_type(field_type):
    origin = getattr(field_type, "__origin__", None)
    if origin is typing.Union or isinstance(
        field_type, getattr(types, "UnionType", type(None))
    ):
        args = [a for a in typing.get_args(field_type) if a is not type(None)]
        return args[-1] if args else str
    return field_type


def get_doc_strings(obj) -> dict[str, str]:
    docs = {}
    obj_doc = inspect.getdoc(obj)
    if obj_doc:
        args_section = re.search(r"Args:\s*(.*)", obj_doc, re.DOTALL | re.IGNORECASE)
        if args_section:
            pattern = r"(\w+):\s*(.*?)(?=\n\s*\w+:|\Z)"
            matches = re.findall(pattern, args_section.group(1), re.DOTALL)
            docs.update(
                {name.strip(): " ".join(desc.split()) for name, desc in matches}
            )
    return docs


def _cast_value(target_type, value):
    """Recursively casts raw YAML strings/dicts/lists into their type-hinted objects (like Path)."""
    if value is None or value is inspect._empty:
        return value

    base_type = get_base_type(target_type)
    origin = typing.get_origin(base_type) or base_type

    # 1. Handle Paths directly
    if origin is Path:
        return Path(value)

    # 2. Handle lists (e.g., list[Path])
    if origin in (list, collections.abc.Sequence) and isinstance(value, list):
        args = typing.get_args(base_type)
        if args:
            inner_type = args[0]
            return [_cast_value(inner_type, v) for v in value]

    # 3. Handle dicts (e.g., dict[Path, Path])
    if origin in (dict, collections.abc.Mapping) and isinstance(value, dict):
        args = typing.get_args(base_type)
        if args and len(args) == 2:
            k_type, v_type = args
            return {
                _cast_value(k_type, k): _cast_value(v_type, v) for k, v in value.items()
            }

    # 4. Handle standard primitives (if they aren't already that type)
    if isinstance(origin, type) and not isinstance(value, origin):
        try:
            return origin(value)
        except TypeError, ValueError:
            pass

    return value


class CLI:
    def __init__(self, description="CLI Application"):
        self.description = description
        self.commands = {}
        self.command_configs = {}

    def command(self, _func=None, *, default_config: str | None = None):
        def decorator(func):
            self.commands[func.__name__] = func
            self.command_configs[func.__name__] = default_config
            return func

        if _func is None:
            return decorator
        return decorator(_func)

    def __call__(self, args_list=None):
        parser = argparse.ArgumentParser(description=self.description)
        subparsers = parser.add_subparsers(dest="__subcommand__", required=True)

        command_meta = {}

        for name, func in self.commands.items():
            func_doc = inspect.getdoc(func)
            sub_parser = subparsers.add_parser(
                name,
                description=func_doc.split("\n\n")[0] if func_doc else None,
                help=func_doc.split("\n")[0] if func_doc else None,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            sub_parser.add_argument(
                "--config", type=str, help="Path to a YAML config file"
            )
            sub_parser.add_argument(
                "--no-config", action="store_true", help="Ignore default config"
            )
            sub_parser.add_argument(
                "--dump-config", action="store_true", help="Dump YAML config"
            )

            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            func_docs = get_doc_strings(func)

            def add_dataclass_fields(p_obj, dclass, prefix):
                field_docs = get_doc_strings(dclass)
                group = p_obj.add_argument_group(f"Settings for {prefix}")

                for field in fields(dclass):
                    full_name = f"{prefix}.{field.name}"
                    cli_flag = full_name.replace("_", "-")
                    base_type = get_base_type(field.type)
                    help_text = field_docs.get(field.name, f"Value for {field.name}")

                    origin = typing.get_origin(base_type)
                    if origin is list or origin is collections.abc.Sequence:
                        continue

                    if is_dataclass(base_type):
                        add_dataclass_fields(p_obj, base_type, full_name)
                    else:
                        group.add_argument(
                            f"--{cli_flag}",
                            type=base_type
                            if origin is None
                            else str,  # Fallback generic types to str for argparse
                            default=argparse.SUPPRESS,
                            help=help_text,
                            dest=full_name,
                        )

            for param_name, param in sig.parameters.items():
                p_type = type_hints.get(param_name, str)
                base_p_type = get_base_type(p_type)

                origin = typing.get_origin(base_p_type)
                if origin is list:
                    inner = typing.get_args(base_p_type)
                    if inner and is_dataclass(get_base_type(inner[0])):
                        continue

                p_help = func_docs.get(param_name, "")
                if is_dataclass(base_p_type):
                    add_dataclass_fields(sub_parser, base_p_type, param_name)
                else:
                    cli_flag = param_name.replace("_", "-")
                    sub_parser.add_argument(
                        f"--{cli_flag}",
                        type=base_p_type
                        if origin is None
                        else str,  # Protect argparse from Generics
                        default=argparse.SUPPRESS,
                        help=p_help,
                        dest=param_name,
                    )

            command_meta[name] = {"func": func, "sig": sig, "type_hints": type_hints}

        parsed_args = vars(parser.parse_args(args_list))
        subcommand_name = parsed_args.pop("__subcommand__")

        meta = command_meta[subcommand_name]
        target_func = meta["func"]
        sig = meta["sig"]
        type_hints = meta["type_hints"]

        config_data = {}
        explicit_config = parsed_args.get("config")
        default_config = self.command_configs.get(subcommand_name)
        bypass_config = parsed_args.get("no_config")

        target_path = None
        if bypass_config:
            target_path = None
        elif explicit_config:
            target_path = Path(explicit_config)
            if not target_path.exists():
                raise FileNotFoundError(f"Missing config: '{target_path.resolve()}'")
        elif default_config:
            potential_path = Path(default_config)
            if potential_path.exists():
                target_path = potential_path

        if target_path:
            with open(target_path, "r") as f:
                config_data = yaml.load(f, Loader=IncludeLoader) or {}

        def get_val(path, default_dict):
            if path in parsed_args:
                return parsed_args[path]
            keys = path.split(".")
            val = default_dict
            for k in keys:
                if isinstance(val, dict) and k in val:
                    val = val[k]
                else:
                    return inspect._empty
            return val

        def _dict_to_dataclass(target_class, data: dict):
            if not is_dataclass(target_class) or not isinstance(data, dict):
                return data

            init_data = {}
            hints = typing.get_type_hints(target_class)
            for f in fields(target_class):
                if f.name not in data:
                    continue
                val = data[f.name]
                f_type = hints.get(f.name, str)
                f_base = get_base_type(f_type)

                origin = typing.get_origin(f_type)
                if origin is list and typing.get_args(f_type):
                    inner_type = get_base_type(typing.get_args(f_type)[0])
                    if is_dataclass(inner_type) and isinstance(val, list):
                        init_data[f.name] = [
                            _dict_to_dataclass(inner_type, v) for v in val
                        ]
                        continue

                if is_dataclass(f_base):
                    init_data[f.name] = _dict_to_dataclass(f_base, val)
                else:
                    # KEY FIX: Cast values inside dataclasses (e.g. converting string to Path)
                    init_data[f.name] = _cast_value(f_type, val)

            return target_class(**init_data)

        def construct(target_type, prefix):
            target = get_base_type(target_type)

            origin = typing.get_origin(target_type)
            if origin is list:
                inner_args = typing.get_args(target_type)
                if inner_args:
                    inner_type = get_base_type(inner_args[0])
                    if is_dataclass(inner_type):
                        raw_val = get_val(prefix, config_data)
                        if raw_val is not inspect._empty and isinstance(raw_val, list):
                            return [
                                _dict_to_dataclass(inner_type, item) for item in raw_val
                            ]

            if is_dataclass(target):
                init_data = {}
                for f in fields(target):
                    val = construct(f.type, f"{prefix}.{f.name}")
                    if val is not inspect._empty:
                        init_data[f.name] = val
                return target(**init_data)

            # KEY FIX: Cast top-level primitives from YAML into their required types
            raw_val = get_val(prefix, config_data)
            return _cast_value(target_type, raw_val)

        final_kwargs = {}
        for p_name in sig.parameters:
            val = construct(type_hints.get(p_name, str), p_name)
            if val is not inspect._empty:
                final_kwargs[p_name] = val

        if parsed_args.get("dump_config"):
            bound_args = sig.bind(**final_kwargs)
            bound_args.apply_defaults()

            def _to_dict(obj):
                if is_dataclass(obj):
                    return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, (list, tuple)):
                    return [_to_dict(i) for i in obj]
                if isinstance(obj, dict):
                    return {_to_dict(k): _to_dict(v) for k, v in obj.items()}
                return obj

            clean_config = {k: _to_dict(v) for k, v in bound_args.arguments.items()}
            yaml_str = yaml.dump(clean_config, sort_keys=False)
            colored_yaml = re.sub(
                r"(^|\n)(\s*)([\w-]+):", r"\1\2\033[36m\3\033[0m:", yaml_str
            )
            print(
                f"\033[1m--- Resolved Configuration for '{subcommand_name}' ---\033[0m"
            )
            print(colored_yaml, end="")
            sys.exit(0)

        return target_func(**final_kwargs)
