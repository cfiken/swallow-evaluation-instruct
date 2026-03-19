# Adapted from syllapy 0.7.2 (https://github.com/mholtzscher/syllapy).
# Copyright (c) 2018, Michael Holtzscher
# Licensed under the MIT License. See lighteval/licenses/syllapy.MIT.

"""Loads vendored reference data to memory."""

import csv
from importlib import resources


def load_dict() -> dict[str, int]:
    """Loads reference data to dictionary."""
    words = {}
    resource = resources.files(__package__).joinpath("data.csv")

    with resource.open(newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        for row in reader:
            words[row[0]] = int(row[1])
    return words
