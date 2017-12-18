from sklearn.svm import LinearSVC
import sklearn.feature_selection as fs_scikit
from sklearn.feature_selection import SelectFromModel
import sys
from sklearn.pipeline import Pipeline
import numpy as np


def get_selected_features(model, features):
    """Estimates support of given fitted model and compares it to given list
    of features. Returns list of features selected through feature selection.
    """
    try:
        support = model.get_support()
    except:
        # Different methods use different ways to get support.
        try:
            support = model.support_
        except:
            try:
                support = [True if i >= np.mean(model.coef_[0])
                           else False for i in model.coef_[0]]
            except:
                try:
                    support = model.named_steps[
                        "feature_selection"].get_support()
                except AttributeError as err:
                    print("Attribute error:{0}".format(err))
                    sys.exit(1)
    if support is not None:
        for idx, val in enumerate(support):
            if not val:
                features[idx] = None
    features = list(filter(None, features))
    return features


def get_fs_model(model, method, train, target=None, cv=None):
    """Connects given model with specified feature selection method and trains
    the final structure.
    """
    if method == "RFE":
        model = fs_scikit.RFE(model, 5, step=3)
        if target is not None:
            return model.fit(train, target)
        else:
            return model.fit(train)
    if method == "RFECV":
        model = fs_scikit.RFECV(model, 3, cv=cv)
        if target is not None:
            return model.fit(train, target)
        else:
            return model.fit(train)
    elif method == "linearSVC":
        sel = SelectFromModel(LinearSVC(loss='l2', penalty='l1', dual=False))
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "fromModel":
        fm = fs_scikit.SelectFromModel(model)
        if target is not None:
            fm.fit(train, target)
        else:
            fm.fit(train)
        model = Pipeline([
            ('feature_selection', fm),
            ('data_mining', model)
        ])

    # elif method == "Anova":
        # ANOVA SVM-C
        # anova_filter = fs_scikit.SelectKBest(f_regression, k=5)
        # model = Pipeline([
        #     ('feature_selection', anova_filter),
        #     ('data_mining', model)
        # ])
    elif method == "VarianceThreshold":
        sel = fs_scikit.VarianceThreshold(threshold=(.8 * (1 - .8)))
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectPercentile":
        sel = fs_scikit.SelectPercentile(fs_scikit.f_classif, percentile=10)
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectFpr":
        sel = fs_scikit.SelectFpr(alpha=0.1)
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectFdr":
        sel = fs_scikit.SelectFdr(alpha=0.1)
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "SelectFwe":
        sel = fs_scikit.SelectFwe(alpha=0.1)
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    elif method == "ch2":
        sel = fs_scikit.SelectKBest(fs_scikit.chi2, k=2)
        model = Pipeline([
            ('feature_selection', sel),
            ('data_mining', model)
        ])
    else:
        print("Feature selection method was not found: "+method)
        sys.exit(1)
    return model
