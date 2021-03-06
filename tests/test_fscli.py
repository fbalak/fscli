#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fscli` package."""

import pytest
import os.path

from click.testing import CliRunner

from fscli import cli


@pytest.fixture
def traindata():
    """Returns training dataset used in tests."""
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "testdata", "traindata")


@pytest.fixture
def testdata():
    """Returns testing dataset used in tests."""
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "testdata", "testdata")


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
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["RandomForestClassifier", "SVC", "MultinomialNB"])
@pytest.mark.parametrize("fs",
                         ["VarianceThreshold", "SelectFdr"])
def test_classification_with_fs(
        task, fs, traindata, target_attribute, testdata):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--fs_task={}'.format(fs),
        '--target_attribute={}'.format(target_attribute),
        '--test={}'.format(testdata)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["RandomForestClassifier", "SVC", "MultinomialNB"])
def test_classification_cv(task, traindata, target_attribute):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--target_attribute={}'.format(target_attribute)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["RandomForestClassifier", "MultinomialNB"])
@pytest.mark.parametrize(
    "fs",
    ["VarianceThreshold", "SelectFdr", "fromModel", "RFE"])
def test_classification_with_fs_cv(
        task, fs, traindata, target_attribute):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--fs_task={}'.format(fs),
        '--target_attribute={}'.format(target_attribute)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["KMeans"])
def test_clustering(task, traindata, target_attribute, testdata):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--target_attribute={}'.format(target_attribute),
        '--test={}'.format(testdata)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["KMeans"])
@pytest.mark.parametrize("fs",
                         ["VarianceThreshold", "SelectFdr"])
def test_clustering_with_fs(
        task, fs, traindata, testdata, target_attribute):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--target_attribute={}'.format(target_attribute),
        '--fs_task={}'.format(fs),
        '--test={}'.format(testdata)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["KMeans"])
def test_clustering_cv(task, traindata, target_attribute):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--target_attribute={}'.format(target_attribute),
        '--dataset={}'.format(traindata)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["KMeans"])
@pytest.mark.parametrize("fs",
                         ["VarianceThreshold", "SelectFdr"])
def test_clustering_with_fs_cv(
        task, fs, traindata, target_attribute):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--target_attribute={}'.format(target_attribute),
        '--fs_task={}'.format(fs)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["LinearRegression", "Ridge", "Lasso"])
def test_regression(task, traindata, target_attribute, testdata):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--target_attribute={}'.format(target_attribute),
        '--test={}'.format(testdata)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["LinearRegression", "Ridge", "Lasso"])
@pytest.mark.parametrize("fs",
                         ["VarianceThreshold", "SelectFdr"])
def test_regression_with_fs(
        task, fs, traindata, target_attribute, testdata):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--fs_task={}'.format(fs),
        '--target_attribute={}'.format(target_attribute),
        '--test={}'.format(testdata)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["LinearRegression", "Ridge", "Lasso"])
def test_regression_cv(task, traindata, target_attribute):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--target_attribute={}'.format(target_attribute)])
    print(result.output)
    assert result.exit_code == 0


@pytest.mark.parametrize("task",
                         ["LinearRegression", "Ridge", "Lasso"])
@pytest.mark.parametrize("fs",
                         ["VarianceThreshold", "SelectFdr"])
def test_regression_with_fs_cv(
        task, fs, traindata, target_attribute):
    """Calls cli with specified machine learning tasks."""
    runner = CliRunner()
    result = runner.invoke(cli.main, [
        task,
        '--dataset={}'.format(traindata),
        '--fs_task={}'.format(fs),
        '--target_attribute={}'.format(target_attribute)])
    print(result.output)
    assert result.exit_code == 0
