from sklearn import metrics as mx
from sklearn.model_selection import KFold
import pandas as pd
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
