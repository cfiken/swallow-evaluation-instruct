# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Tests adapted from syllapy 0.7.2 upstream tests."""

from string import punctuation

import pytest

from lighteval.tasks.extended.ifbench import _vendor_syllapy as syllapy


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("dog!!!!!", 1),
        (None, 0),
        (True, 0),
        (2, 0),
        ("", 0),
        (" ", 0),
        ("ostentatious", 4),
        ("because", 2),
        ("woman", 2),
        ("international", 5),
        ("Norway", 2),
        ("norway", 2),
        ("Ohio", 3),
        ("ohio", 3),
        ("part-time", 2),
        ("one-on-one", 3),
        ("four-at-a-time", 4),
        ("4-at-a-time", 0),
        ("zero-for-2", 0),
        ("d0g", 0),
        ("4dog", 0),
        ("dog123", 0),
    ],
)
def test_vendor_syllapy_count_examples(value, expected):
    assert syllapy.count(value) == expected


@pytest.mark.parametrize("punct", list(punctuation))
def test_vendor_syllapy_punctuation_only(punct):
    assert syllapy.count(punct) == 0
