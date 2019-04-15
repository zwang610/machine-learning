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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

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
        freqDict = Counter(y).most_common(2)
        length = freqDict[0][1] + freqDict[1][1]
        self.probabilities_ = (freqDict[0][1] if freqDict[0][0] == 0 else freqDict[1][1])/length
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
        
        y = np.random.choice(2,X.shape[0],p=[self.probabilities_,1-self.probabilities_])
        ### ========== TODO : END ========== ####
        
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
        plt.show()

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
    for index in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=test_size, random_state=index)
        clf.fit(X_train, y_train)                  # fit training data using the classifier
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        train_error += 1 - metrics.accuracy_score(y_train, y_pred_train, normalize=True)
        test_error += 1 - metrics.accuracy_score(y_test, y_pred_test, normalize=True)
    train_error = train_error/ntrials
    test_error = test_error/ntrials
        
    
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

       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    randclf = RandomClassifier()
    randclf.fit(X, y)
    randy_pred = randclf.predict(X)
    randtrain_error = 1 - metrics.accuracy_score(y, randy_pred, normalize=True)
    print('\t-- training error: %.3f' % randtrain_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    dtclf = DecisionTreeClassifier(criterion='entropy')
    dtclf.fit(X, y)
    dty_pred = dtclf.predict(X)
    dttrain_error = 1 - metrics.accuracy_score(y, dty_pred, normalize=True)
    print('\t-- training error: %.3f' % dttrain_error)
    
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
    knnclf3 = KNeighborsClassifier(n_neighbors=3)
    knnclf3.fit(X, y)
    knny_pred = knnclf3.predict(X)
    knntrain_error = 1 - metrics.accuracy_score(y, knny_pred, normalize=True)
    print('\t-- k = 3 training error: %.3f' % knntrain_error)
    knnclf5 = KNeighborsClassifier(n_neighbors=5)
    knnclf5.fit(X, y)
    knny_pred = knnclf5.predict(X)
    knntrain_error = 1 - metrics.accuracy_score(y, knny_pred, normalize=True)
    print('\t-- k = 5 training error: %.3f' % knntrain_error)
    knnclf7 = KNeighborsClassifier(n_neighbors=7)
    knnclf7.fit(X, y)
    knny_pred = knnclf7.predict(X)
    knntrain_error = 1 - metrics.accuracy_score(y, knny_pred, normalize=True)
    print('\t-- k = 7 training error: %.3f' % knntrain_error)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    train_error, test_error = error(clf, X, y)
    print('\t-- Average result of major classifier training error = {:.3f}, testing error = {:.3f}'.format(train_error , test_error))
    train_error, test_error = error(randclf, X, y)
    print('\t-- Average result of random classifier training error = {:.3f}, testing error = {:.3f}'.format(train_error , test_error))
    train_error, test_error = error(dtclf, X, y)
    print('\t-- Average result of decision dree classifier training error = {:.3f}, testing error = {:.3f}'.format(train_error , test_error))
    train_error, test_error = error(knnclf5, X, y)
    print('\t-- Average result of K=5 nearest neightbors training error = {:.3f}, testing error = {:.3f}'.format(train_error , test_error))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    errors = []
    ks = []
    for k in range(1,50,2):
        errorfold =  cross_val_score(KNeighborsClassifier(n_neighbors=k), X, y, cv=10)
        ks.append(k)
        errors.append(1-np.mean(errorfold))
    plt.plot(ks,errors)
    plt.xlabel('number of nearest neighbors')
    plt.ylabel('error rate')
    plt.show()
    bestk=np.argmin(errors)*2+1
    print('\t-- best number of neighbors: %.3f' % bestk)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    train_errors = []
    test_errors = []
    ds = []
    for d in range(1,21,1):
        train_errorfold, test_errorfold = error(DecisionTreeClassifier(criterion='entropy',max_depth=d), X, y)
        ds.append(d)
        train_errors.append(np.mean(train_errorfold))
        test_errors.append(np.mean(test_errorfold))
    bestd=np.argmin(test_errors)
    red_patch = mpl.patches.Patch(color='red', label='training error')
    blue_patch = mpl.patches.Patch(color='blue', label='test error')
    plt.plot(ds,train_errors,'r', ds,test_errors, 'b')
    plt.xlabel('depth of the decision tree')
    plt.ylabel('error rate')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()
    print('\t-- best depth limit: %.3f' % bestd)
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
    index = []
    dtTrainErrs = []
    dtTestErrs = []
    knnTrainErrs = []
    knnTestErrs = []
    for i in range(1, 11):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        index.append(i*0.1)
        for k in range(100):
            X_train2, y_train2 = X_train, y_train
            if i != 10:
                X_train2, _, y_train2, _ = train_test_split(X_train, y_train, test_size=(1-i*0.1), random_state=k)
            dtClf = DecisionTreeClassifier(criterion='entropy', max_depth=5)
            knnClf = KNeighborsClassifier(n_neighbors=7, p=2)

            dtClf.fit(X_train2, y_train2)
            dt_y_train_pred = dtClf.predict(X_train2)
            train_error1 = 1 - metrics.accuracy_score(y_train2, dt_y_train_pred, normalize=True)
            sum1 += train_error1
            
            dt_y_test_pred = dtClf.predict(X_test)
            test_error1 = 1 - metrics.accuracy_score(y_test, dt_y_test_pred, normalize=True)
            sum2 += test_error1


            knnClf.fit(X_train2, y_train2)
            knn_y_train_pred = knnClf.predict(X_train2)
            train_error2 = 1 - metrics.accuracy_score(y_train2, knn_y_train_pred, normalize=True)
            sum3 += train_error2
            
            knn_y_test_pred = knnClf.predict(X_test)
            test_error2 = 1 - metrics.accuracy_score(y_test, knn_y_test_pred, normalize=True)
            sum4 += test_error2
        dtTrainErrs.append(sum1/100)
        dtTestErrs.append(sum2/100)
        knnTrainErrs.append(sum3/100)
        knnTestErrs.append(sum4/100)

    plt.clf()
    plt.plot(index, dtTrainErrs, 'r', \
            index, dtTestErrs, 'r--', \
            index, knnTrainErrs, 'b', \
            index, knnTestErrs, 'b--')
    red_patch = mpl.lines.Line2D([],[],color='red', label='training error of DT')
    reddot_patch = mpl.lines.Line2D([],[],color='red', label='test error of DT', linestyle='--')
    blue_patch = mpl.lines.Line2D([],[],color='blue', label='training error of KNN')
    bluedot_patch = mpl.lines.Line2D([],[],color='blue',label='test error of KNN', linestyle='--')

    plt.legend(handles=[red_patch, reddot_patch, blue_patch, bluedot_patch])
    plt.xlabel('proportion of 90% training set')
    plt.ylabel('error rate')
    plt.show()
        
    ### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
