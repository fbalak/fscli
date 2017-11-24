from sklearn import metrics as mx
from sklearn.model_selection import KFold
import pandas as pd
try:
    from fscli import featureselection
except ImportError as err:
    import featureselection


def classification(source, model, target_att, test_source="", fs_task=False):
    """Performs classification on given data.

    Params:
        source -- Path to the file that is used to train.
        model -- Object loaded from file with trained model.
        target_att -- Name of attribute in source that is considered as target.
        test_source -- Path to the file that is used to test.
        fs_task -- String with name of used feature selection algorithm.
    """

    results = dict.fromkeys([
        "predictions",
        "score",
        "model",
        "features",
        "removed_features",
        "selected_features",
        "feature_importances",
        "measures"])
    results["predictions"] = []

    # Basic metrics used for classification and feature selection evaluation.
    metrics = dict.fromkeys(["accuracy", "recall", "precision", "f_measure"])
    metrics["accuracy"] = []
    metrics["recall"] = []
    metrics["precision"] = []
    metrics["f_measure"] = []
    results["removed_features"] = []
    results["selected_features"] = []
    results["feature_importances"] = []

    cfr = model
    print(model)

    # Object for reading train data and test data
    csv = pd.read_csv(source)

    # Numpy array with values from source path without feature names and
    # target values.
    train = csv.ix[:, csv.columns != target_att].values

    # List of feature names
    features = csv.columns.tolist()

    # Numpy array with target values
    target = csv[target_att].values

    if fs_task:
        # Pipeline with fitted model and feature selection filter or only
        # fitted model.
        cfr = featureselection.get_fs_model(cfr, fs_task, train, target)

    if test_source:

        # Numpy array with values from test_source path without feature names
        # and target values.
        test_csv = pd.read_csv(test_source)
        test = test_csv.ix[:, csv.columns != target_att].values

        # Numpy array with test target values
        test_target = test_csv[target_att].values

        cfr.fit(train, target)
        prediction = cfr.predict(test)
        results["predictions"].append(prediction)
        metrics["accuracy"].append(mx.accuracy_score(test_target, prediction))
        metrics["precision"].append(
            mx.precision_score(test_target, prediction, average="macro"))
        metrics["recall"].append(
            mx.recall_score(test_target, prediction, average="macro"))
        metrics["f_measure"].append(
            mx.f1_score(test_target, prediction, average="macro"))
    else:
        cv = KFold(n_splits=4, shuffle=True)
        for train_idx, test_idx in cv.split(train):
            cfr.fit(train[train_idx], target[train_idx])
            prediction = cfr.predict(train[test_idx])
            results["predictions"].append(prediction)
            metrics["accuracy"].append(
                mx.accuracy_score(target[test_idx], prediction))
            metrics["precision"].append(
                mx.precision_score(
                    target[test_idx],
                    prediction,
                    average="macro"))
            metrics["recall"].append(
                mx.recall_score(target[test_idx], prediction, average="macro"))
            metrics["f_measure"].append(
                    mx.f1_score(target[test_idx], prediction, average="macro"))
    # results["score"] = cfr.score(test, test_target)

    if fs_task:
        original_features = features[:]
        if fs_task == "RFE":
            selected_features = []
        elif fs_task == "fromModel":
            selected_features = featureselection.get_selected_features(
                cfr,
                original_features)
        else:
            selected_features = featureselection.get_selected_features(
                cfr.named_steps["feature_selection"], original_features)
        removed_features = [i for i in features if i not in selected_features]
        results["removed_features"].append(removed_features)
        results["selected_features"].append(selected_features)

    results["model"] = cfr
    results["metrics"] = metrics
    return results


def clustering(source, model, target_att, test_source="", fs_task=False):
    """Performs clustering on given data.

    Params:
        source -- Path to the file that is used to train.
        model -- Object loaded from file with trained model.
        target_att -- Name of attribute in source that is considered as target.
        test_source -- Path to the file that is used to test.
        fs_task -- String with name of used feature selection algorithm.
    """

    results = dict.fromkeys([
        "predictions",
        "score",
        "model",
        "features",
        "removed_features",
        "selected_features",
        "feature_importances",
        "measures"])
    results["predictions"] = []

    # Basic metrics used for clustering and feature selection evaluation.
    metrics = dict.fromkeys(["homogeneity", "f_measure"])
    metrics["homogeneity"] = []
    metrics["completeness"] = []
    metrics["fowlkes"] = []
    metrics["v_measure"] = []
    results["removed_features"] = []
    results["selected_features"] = []
    results["feature_importances"] = []

    cfr = model
    print(model)

    # Object for reading train data and test data
    csv = pd.read_csv(source)

    # Numpy array with values from source path without feature names and
    # target values.
    train = csv.ix[:, csv.columns != target_att].values

    # List of feature names
    features = csv.columns.tolist()

    # Numpy array with target values
    target = csv[target_att].values

    if fs_task:
        # Pipeline with fitted model and feature selection filter or only
        # fitted model.
        cfr = featureselection.get_fs_model(cfr, fs_task, train, target)

    if test_source:

        # Numpy array with values from test_source path without feature names
        # and target values.
        test_csv = pd.read_csv(test_source)
        test = test_csv.ix[:, csv.columns != target_att].values

        # Numpy array with test target values
        test_target = test_csv[target_att].values

        cfr.fit(train, target)
        prediction = cfr.predict(test)
        results["predictions"].append(prediction)
        metrics["homogeneity"].append(
            mx.homogeneity_score(test_target, prediction))
        metrics["completeness"].append(
            mx.completeness_score(test_target, prediction))
        metrics["fowlkes"].append(
            mx.fowlkes_mallows_score(test_target, prediction))
        metrics["v_measure"].append(
            mx.v_measure_score(test_target, prediction))
    else:
        cv = KFold(n_splits=4, shuffle=True)
        for train_idx, test_idx in cv.split(train):
            cfr.fit(train[train_idx], target[train_idx])
            prediction = cfr.predict(train[test_idx])
            results["predictions"].append(prediction)
            metrics["homogeneity"].append(
                mx.homogeneity_score(target[test_idx], prediction))
            metrics["completeness"].append(
                mx.completeness_score(target[test_idx], prediction))
            metrics["fowlkes"].append(
                mx.fowlkes_mallows_score(target[test_idx], prediction))
            metrics["v_measure"].append(
                mx.v_measure_score(target[test_idx], prediction))
    # results["score"] = cfr.score(test, test_target)

    if fs_task:
        original_features = features[:]
        if fs_task == "RFE":
            selected_features = []
        elif fs_task == "fromModel":
            selected_features = featureselection.get_selected_features(
                cfr,
                original_features)
        else:
            selected_features = featureselection.get_selected_features(
                cfr.named_steps["feature_selection"], original_features)
        removed_features = [i for i in features if i not in selected_features]
        results["removed_features"].append(removed_features)
        results["selected_features"].append(selected_features)

    results["model"] = cfr
    results["metrics"] = metrics
    return results


def regression(source, model, target_att, test_source="", fs_task=False):
    """Performs regression on given data.

    Params:
        source -- Path to the file that is used to train.
        model -- Object loaded from file with trained model.
        target_att -- Name of attribute in source that is considered as target.
        test_source -- Path to the file that is used to test.
        fs_task -- String with name of used feature selection algorithm.
    """

    results = dict.fromkeys([
        "predictions",
        "score",
        "model",
        "features",
        "removed_features",
        "selected_features",
        "feature_importances",
        "measures"])
    results["predictions"] = []

    # Basic metrics used for regression and feature selection evaluation.
    metrics = dict.fromkeys(
        ["explained_variance", "neg_mean_absolute_error",
         "neg_mean_squared_error", "neg_mean_squared_log_error", "r2",
         "neg_median_absolute_error"])
    metrics["explained_variance"] = []
    metrics["neg_mean_absolute_error"] = []
    metrics["neg_mean_squared_error"] = []
    metrics["neg_mean_squared_log_error"] = []
    metrics["r2"] = []
    metrics["neg_median_absolute_error"] = []
    results["removed_features"] = []
    results["selected_features"] = []
    results["feature_importances"] = []

    cfr = model
    print(model)

    # Object for reading train data and test data
    csv = pd.read_csv(source)

    # Numpy array with values from source path without feature names and
    # target values.
    train = csv.ix[:, csv.columns != target_att].values

    # List of feature names
    features = csv.columns.tolist()

    # Numpy array with target values
    target = csv[target_att].values

    if fs_task:
        # Pipeline with fitted model and feature selection filter or only
        # fitted model.
        cfr = featureselection.get_fs_model(cfr, fs_task, train, target)

    if test_source:

        # Numpy array with values from test_source path without feature names
        # and target values.
        test_csv = pd.read_csv(test_source)
        test = test_csv.ix[:, csv.columns != target_att].values

        # Numpy array with test target values
        test_target = test_csv[target_att].values

        cfr.fit(train, target)
        prediction = cfr.predict(test)
        results["predictions"].append(prediction)
        metrics["explained_variance"].append(
            mx.explained_variance_score(test_target, prediction))
        metrics["neg_mean_absolute_error"].append(
            mx.mean_absolute_error(test_target, prediction))
        metrics["neg_mean_squared_error"].append(
            mx.mean_squared_error(test_target, prediction))
        metrics["neg_mean_squared_log_error"].append(
            mx.mean_squared_log_error(test_target, prediction))
        metrics["r2"].append(
            mx.r2_score(test_target, prediction))
        metrics["neg_median_absolute_error"].append(
            mx.median_absolute_error(test_target, prediction))
    else:
        cv = KFold(n_splits=4, shuffle=True)
        for train_idx, test_idx in cv.split(train):
            cfr.fit(train[train_idx], target[train_idx])
            prediction = cfr.predict(train[test_idx])
            results["predictions"].append(prediction)
            metrics["explained_variance"].append(
                mx.explained_variance_score(target[test_idx], prediction))
            metrics["neg_mean_absolute_error"].append(
                mx.mean_absolute_error(target[test_idx], prediction))
            metrics["neg_mean_squared_error"].append(
                mx.mean_squared_error(target[test_idx], prediction))
            metrics["neg_mean_squared_log_error"].append(
                mx.mean_squared_log_error(target[test_idx], prediction))
            metrics["r2"].append(
                mx.r2_score(target[test_idx], prediction))
            metrics["neg_median_absolute_error"].append(
                mx.median_absolute_error(target[test_idx], prediction))
    # results["score"] = cfr.score(test, target[test_idx])

    if fs_task:
        original_features = features[:]
        if fs_task == "RFE":
            selected_features = []
        elif fs_task == "fromModel":
            selected_features = featureselection.get_selected_features(
                cfr,
                original_features)
        else:
            selected_features = featureselection.get_selected_features(
                cfr.named_steps["feature_selection"], original_features)
        removed_features = [i for i in features if i not in selected_features]
        results["removed_features"].append(removed_features)
        results["selected_features"].append(selected_features)

    results["model"] = cfr
    results["metrics"] = metrics
    return results
