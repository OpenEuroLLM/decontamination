import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from platform import machine
from typing import Generator

import pandas as pd

NUM_PROCESSES = min(os.cpu_count() or 1, 32)


class Architecture(Enum):
    arm64 = "arm64"
    x86_64 = "amd64"

    def from_platform() -> "Architecture":
        arch = machine()
        if arch in ("x86_64", "AMD64"):
            return Architecture.x86_64
        elif arch in ("aarch64", "arm64"):
            return Architecture.arm64
        else:
            raise RuntimeError(f"Unsupported architecture: {arch}")

    @cached_property
    def docker_image_suffix(self) -> str:
        return self.value


def run_command_streaming(command: list[str]) -> tuple[int, str]:
    """
    Run a subprocess, streaming stdout/stderr to the current
    terminal while capturing output.

    Args:
        command: The command to run as a list of strings.

    Returns:
        A tuple of (returncode, output) where returncode is
        the exit code of the command and output is the captured stdout/stderr.
    """
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)

    return proc.wait(), "".join(output_lines)


@dataclass
class TypedRow[T]:
    def from_row(row: pd.Series) -> T:
        raise NotImplementedError("Must be implemented by subclasses")


class TypedDataFrame[T: TypedRow]:
    def __init__(self, df: pd.DataFrame, type: type[T]):
        self.df = df
        self.type = type

    def __iter__(self) -> Generator[T, None, None]:
        for idx, row in self.df.iterrows():
            yield self.type.from_row(row)

    def add(self, item: T) -> None:
        self.df = pd.concat(
            [self.df, pd.DataFrame([item.__dict__])],
            ignore_index=True,
        )

    def new[D: TypedRow](type: type[D]) -> TypedDataFrame[D]:
        return TypedDataFrame(
            pd.DataFrame(columns=list(type.__annotations__.keys())),
            type,
        )


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"
