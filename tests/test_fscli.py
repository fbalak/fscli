#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fscli` package."""

import pytest
import os.path

from click.testing import CliRunner

from fscli import fscli
from fscli import cli


@pytest.fixture
def traindata():
    """Returns training dataset used in tests."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "testdata", "traindata")

@pytest.fixture
def testdata():
    """Returns testing dataset used in tests."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "testdata", "testdata")

@pytest.fixture
def target_attribute():
    """Returns name of target attribute used in test datasets."""
    return "Class"

@pytest.mark.parametrize("task",
                         ["RandomForestClassifier", "SVC", "MultinomialNB"])
def test_classification(task, traindata, target_attribute, testdata):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--target_attribute={}'.format(target_attribute),
        '--test={}'.format(testdata)])
    assert result.exit_code == 0


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'fscli.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    print(result.output)
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output
