# -*- coding: utf-8 -*-

"""Console script for fscli."""

from time import gmtime, strftime
from sklearn.externals import joblib
from fstester.machinelearning import machinelearning
import click
# Classification
from fstester.coefbugrepair import coefbugrepair
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis 
# Clustering
from sklearn import cluster
from sklearn import mixture
import numpy as np
np.set_printoptions(threshold=np.inf)


@click.command()
@click.argument('task')
@click.argument('dataset')
@click.option('--featureselection', '-f', help='Feature selection algorithm.')
@click.option('--test', '-t',
    help='Test dataset - when not set, cross validation is used.')
@click.option('--model', '-m', help='Trained model.')
@click.option('--save', '-s', help='Path where to save trained model.')
def main(args=None):
    """Console script for fscli."""
    if model == None:
        # Classification
        if task == "RandomForestClassifier":
            tasktype = "classification"
            model = RandomForestClassifier(n_estimators=10)
        elif task == "KNeighborsClassifier":
            tasktype = "classification"
            model = KNeighborsClassifier(20)
        elif task == "SVC":
            tasktype = "classification"
            model = SVC(kernel="linear", C=0.025)
        elif task == "DecisionTreeClassifier":
            tasktype = "classification"
            model = coefbugrepair.DecisionTreeClassifierWithCoef(max_depth=5)
        elif task == "AdaBoostClassifier":
            tasktype = "classification"
            model = AdaBoostClassifier()
        elif task == "GaussianNB":
            tasktype = "classification"
            model = GaussianNB()
        elif task == "MultinomialNB":
            # For demonstration on text data
            # https://classes.soe.ucsc.edu/cmps290c/Spring12/lect/14/CEAS2006_corrected-naiveBayesSpam.pdf
            #
            # The term Multinomial Naive Bayes simply lets us know that each p(fi|c) is a multinomial distribution, rather than some other distribution. This works well for data which can easily be turned into counts, such as word counts in text.
            # http://stats.stackexchange.com/questions/33185/difference-between-naive-bayes-multinomial-naive-bayes
            tasktype = "classification"
            model = MultinomialNB()
        elif task == "QDA":
            tasktype = "classification"
            model = QuadraticDiscriminantAnalysis()
        elif task == "LDA":
            tasktype = "classification"
            model = LinearDiscriminantAnalysis()
        # Clustering
        elif task == "KMeans":
            tasktype = "clustering"
            model = cluster.KMeans(n_clusters=4)
        elif task == "AF":
            tasktype = "clustering"
            model = cluster.AffinityPropagation(preference=-50)
        elif task == "MeanShift":
            tasktype = "clustering"
            model = cluster.MeanShift(bin_seeding=True)
        elif task == "Agglomerative":
            tasktype = "clustering"
            model = cluster.AgglomerativeClustering()
        elif task == "DBSCAN":
            tasktype = "clustering"
            model = cluster.DBSCAN(eps=0.3, min_samples=10)
        elif task == "Birch":
            tasktype = "clustering"
            model = cluster.Birch()
        elif task == "Gaussian":
            tasktype = "clustering"
            model = mixture.GMM(n_components=2, covariance_type='full')

    if model !=None:
        if tasktype == "classification":
            results = machinelearning.classification(source,model,target,test,fs_task)
        elif tasktype == "clustering":
            results = machinelearning.clustering(source,model,target,test,fs_task)

        click.echo("Results")
        pprint.pprint(results["score"])

        if dump_folder != False:
            if dump_folder != "":
                directory = dump_folder
            else:
                directory = '../models/model'+str(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
            if not os.path.exists(directory):
                os.makedirs(directory)
            joblib.dump(results["model"], directory+'/model.pkl')
            click.echo("Dump file of model was created: " + directory+'/model.pkl')

        return results

    else:
        click.echo("On input is wrong task type")
        sys.exit(1)


if __name__ == "__main__":
    main()
