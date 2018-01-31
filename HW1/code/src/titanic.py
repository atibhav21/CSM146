"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import operator

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        classes = np.unique(y);
        self.probabilities_ = {};
        total_number = y.shape[0]
        for i in classes:
            self.probabilities_[int(i)] = len(np.where(y == i)[0])/float(total_number)
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        #print(self.probabilities_)
        n = X.shape[0]

        y = np.random.choice(2, n, p=[self.probabilities_[0], self.probabilities_[1]])
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        #plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    
    
    for i in range(1, ntrials+1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=i)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_train) # compute the training error
        train_error += 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)

        y_pred = clf.predict(X_test) # compute the test error
        test_error += 1 - metrics.accuracy_score(y_test, y_pred, normalize=True)

    train_error = float(train_error) / ntrials
    test_error = float(test_error) / ntrials # average the errors out
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)

    plt.close('all')
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    majority_vote_error = train_error
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    rand_clf = RandomClassifier()
    rand_clf.fit(X, y)
    y_pred = rand_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)

    random_clf_error = train_error;
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    decision_tree_clf = DecisionTreeClassifier(criterion="entropy")
    decision_tree_clf.fit(X, y)
    y_pred = decision_tree_clf.predict(X)
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    for k in (3, 5, 7):   
        k_Nearest_clf = KNeighborsClassifier(n_neighbors=k)
        k_Nearest_clf.fit(X, y)
        y_pred = k_Nearest_clf.predict(X)
        train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
        print('\t-- training error (k = %d): %.3f' % (k, train_error))

    # Redeclare it for part e
    k_Nearest_clf = KNeighborsClassifier(n_neighbors=5)

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    for classifier in (clf, rand_clf, decision_tree_clf, k_Nearest_clf):
        train_error, test_error = error(classifier, X, y)
        print('\t--Train Error:%.3f Test Error%.3f Classifier: %s' % (train_error, test_error, classifier.__class__.__name__ ))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    x_points = []
    y_points = []
    for k in range(1, 50, 2):
        x_points.append(k)
        k_Nearest_clf = KNeighborsClassifier(n_neighbors=k)
        k_y = 1 - cross_val_score(k_Nearest_clf, X, y, scoring='accuracy', cv=10)
        y_points.append(sum(k_y) / len(k_y))
    plt.plot(x_points, y_points)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Average Error')
    plt.show()
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    x_points = []
    y_test_points = []
    y_train_points = []
    for k in range(1, 21):
        decision_tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=k)
        train_error, test_error = error(decision_tree_clf, X, y)
        x_points.append(k)
        y_test_points.append(test_error)
        y_train_points.append(train_error)
    plt.plot(x_points, y_train_points, label='Training Error')
    plt.plot(x_points, y_test_points, label='Test Error')
    plt.plot(x_points, [majority_vote_error] * len(x_points), label='Majority Vote Classifier error')
    plt.plot(x_points, [random_clf_error] * len(x_points), label='Random Classifier error')
    plt.legend(loc='upper right')
    plt.xlabel('Depth')
    plt.ylabel('Average Error')
    plt.ylim(ymax=0.7)
    plt.show()
    #plt.close('all')
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9, random_state=42)
    fraction_percentages = [x / 10.0 for x in range(1, 11)]
    fraction_indices = [int(i * X_train.shape[0]) for i in fraction_percentages]

    k_Nearest_clf = KNeighborsClassifier(n_neighbors=7)
    decision_tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6)

    x_points = fraction_percentages[:]
    y_points_decision_train = []
    y_points_knn_train = []
    y_points_decision_test =[]
    y_points_knn_test = []

    for end_index in fraction_indices:
        X_train_set = X_train[:end_index+1]
        y_train_set = y_train[:end_index+1]

        k_Nearest_clf.fit(X_train_set, y_train_set)
        decision_tree_clf.fit(X_train_set, y_train_set)

        y_pred_knn_train = k_Nearest_clf.predict(X_train_set)
        y_pred_decision_train = decision_tree_clf.predict(X_train_set)

        y_pred_knn_test = k_Nearest_clf.predict(X_test)
        y_pred_decision_test = decision_tree_clf.predict(X_test)

        train_error_knn = 1 - metrics.accuracy_score(y_train_set, y_pred_knn_train, normalize=True)
        test_error_knn = 1 - metrics.accuracy_score(y_test, y_pred_knn_test, normalize=True)

        train_error_decision = 1 - metrics.accuracy_score(y_train_set, y_pred_decision_train, normalize=True)
        test_error_decision = 1 - metrics.accuracy_score(y_test, y_pred_decision_test, normalize=True)

        y_points_decision_train.append(train_error_decision)
        y_points_decision_test.append(test_error_decision)
        y_points_knn_train.append(train_error_knn)
        y_points_knn_test.append(test_error_knn)


    plt.plot(x_points, y_points_decision_train, label="Decision Tree Training Error")
    plt.plot(x_points, y_points_decision_test, label="Decision Tree Test Error")
    plt.plot(x_points, y_points_knn_train, label="KNearest Training Error")
    plt.plot(x_points, y_points_knn_test, label="KNearest Test Error")
    plt.plot(x_points, [majority_vote_error] * len(x_points), label='Majority Vote Classifier error')
    plt.plot(x_points, [random_clf_error] * len(x_points), label='Random Classifier error')
    plt.ylim(ymax=0.8)
    plt.legend(loc='upper right')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Error')
    plt.show()
        
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
