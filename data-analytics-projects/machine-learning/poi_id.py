#!/usr/bin/python

import sys
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Additional libraries used

from time import time
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *


### Functions
### Main functions section, that will outline the code used for our functions

### Pipelines for various classifiers
### Steps will be the same regardless of classifier

### Please pass through one of the following classifiers (nb_clf, dt_clf, svc_clf, kNN_clf) as argument for clf parameter
### please pass through a feature selection method (SelectKBest)
def classifier_pipeline(clf, feature_selection):
    ### Determine the type of classifier passed in out of the four classifiers
    classifier_name = 'TempName'
    steps = []
    param_grid = {}
    
    if clf == nb_clf:
        classifier_name = 'Naive Bayes'
        param_grid = {'feature_selection__k': [1, 2, 3, 4, 5]}
        steps = [('feature_selection', feature_selection),
                 ('classifier', clf)]
    elif clf == dt_clf:
        classifier_name = 'DecisionTree'
        param_grid = {'feature_selection__k': [1, 2, 3, 4, 5],
                      'classifier__min_samples_split': [3, 4, 5, 6, 7, 8, 9, 10],
                      'classifier__max_features': ['auto', 'sqrt', 'log2', None],
                      'classifier__criterion': ['gini', 'entropy']}
        steps = [('feature_selection', feature_selection),
                 ('classifier', clf)]
    elif clf == svc_clf:
        classifier_name = 'SVC'
        param_grid = {'feature_selection__k': [1, 2, 3, 4, 5],
                      'classifier__C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'classifier__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        steps = [('feature_selection', feature_selection),
                 ('scaler', MinMaxScaler()),
                 ('reduce_dim', PCA()),
                 ('classifier', clf)]
    elif clf == kNN_clf:
        classifier_name = 'K-Nearest Neighbors'
        param_grid = {'feature_selection__k': [1, 2, 3, 4, 5],
                      'classifier__weights': ['uniform', 'distance'],
                      'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
        steps = [('feature_selection', feature_selection),
                 ('scaler', MinMaxScaler()),
                 ('reduce_dim', PCA()),
                 ('classifier', clf)]
    elif clf == ab_clf:
        classifier_name = 'AdaBoost'
        param_grid = {'feature_selection__k': [1, 2, 3, 4, 5],
                      'classifier__algorithm': ['SAMME', 'SAMME.R'],
                      'classifier__n_estimators': [2, 10, 25, 50, 100, 150],
                      'classifier__learning_rate': [.5, 1, 1.5, 2]}
        steps =  [('feature_selection', feature_selection),
                  ('scaler', MinMaxScaler()), 
                  ('reduce_dim', PCA()),
                  ('classifier', clf)]
    else:
        print 'Non pre-designated classifier supplied as arguement, please pass through one of the following classifiers (nb_clf, dt_clf, svc_clf, kNN_clf)'
        exit()
    
 
    
    pipe = Pipeline(steps)   

    
    return classifier_name, pipe, param_grid


def strat_shuff_split_cv_model(dataset, features_list):
    ### extract, format, and split the data using featureFormat and targetFeatureSplit
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = model_selection.StratifiedShuffleSplit(n_splits=100, test_size = 0.3, random_state = 12)
    for train_idx, test_idx in cv.split(features, labels): 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
    return cv, features_train, labels_train, features_test, labels_test


### Defines the function evaluate_classifers that will evaluate a variety of supervised learning classifiers and \ 
### records the accuracy, recall, and precision scores. Finds optimal params for each classifier via GridSearchCV \
### and provides the best params for each classifier based on f1 score.
def evaluate_optimized_classifiers(clf, feature_selection, dataset, features_list):

    cv, features_train, labels_train, features_test, labels_test = strat_shuff_split_cv_model(dataset, features_list)
    classifier_name, pipe, param_grid = classifier_pipeline(clf, feature_selection)
    ### get estimator params if need (commented out)
    #print pipe.get_params().keys()
    if classifier_name == 'SVC':
        pca = PCA(svd_solver='randomized', whiten=True).fit(features_train)
        features_train_pca = pca.transform(features_train)
        features_test_pca = pca.transform(features_test)
        grid_search = model_selection.GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv)
        grid_search.fit(features_train_pca, labels_train)
        pred = grid_search.predict(features_test_pca)
    else:
        pass
    grid_search = model_selection.GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv)
    grid_search.fit(features_train, labels_train)
    pred = grid_search.predict(features_test)
    accuracy_results = accuracy_score(y_true=labels_test, y_pred=pred)
    recall_results = recall_score(y_true=labels_test, y_pred=pred)
    precision_results = precision_score(y_true=labels_test, y_pred=pred)

    if classifier_name == 'Naive Bayes':
        print
        print "Naive Bayes Classifier Results:"
    elif classifier_name == 'DecisionTree':
        print
        print "Decision Tree Classifier Results:"
    elif classifier_name == 'SVC':
        print
        print "SVC Classifier Results:"
    elif classifier_name == 'K-Nearest Neighbors':
        print
        print "K-Nearest Neighbors Classifier Results:"
    elif classifier_name == 'AdaBoost':
        print
        print "Adaboost Classifier Results:"
    else:
        print "You messed up"
    print "Accuracy Score: {}".format(accuracy_results)
    print "Recall Score: {}".format(recall_results)
    print "Precision Score: {}".format(precision_results)
    print "Best Estimator:", grid_search.best_estimator_
    print "Best Parameters: ", grid_search.best_estimator_.get_params()



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### This is the section where I answered a lot of dataset questions I had and where I explored the data.
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features = ['to_messages','email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
email_features.remove('email_address')
poi_features = ['poi']
features_list = poi_features + email_features + financial_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### print out the names of the data points for visual inspection (commented out for grader's convenience), this is how I found \
### the names "THE TRAVEL AGENCY IN THE PARK" and "TOTAL" and decided they should be removed since they aren't people.
'''
names_of_people = sorted(data_dict.keys())
print "Names of the People found in dataset, sorted by last name:", names_of_people
'''
### export the dataset into a .csv file for visual inspection (commented out for grader's convenience). Compare with enron61702insiderpay.pdf \
### to verify accuracy of financial data. I found that the values for "BELFER, ROBERT A" and "BHATNAGAR, SANJAY"
### This is how I noticed "LOCKHART EUGENE E" contained only "NaN" values, will remove him so he doesn't distort our results.
'''
fieldnames = ['name'] + data_dict['METTS MARK'].keys()

with open('enron data.csv', 'wb') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames)

    writer.writeheader()
    for name in data_dict.keys():
        if name != "TOTAL" or name != "THE TRAVEL AGENCY IN THE PARK":
            n = {'name':name}
            n.update(data_dict[name])
            writer.writerow(n)
'''
### Dataset Exploration
print "Exploring the Dataset:"
### print the total number of data points in the dataset. Each data point represents a person. 
print "Total Number of Data points:", len(data_dict)

### print the total number of features in the dataset. Each feature describes attributes or statistics concerning an individual. 
print "Total Number of Features:", len(data_dict['METTS MARK'])

### print the number of POI vs non-POI data points.
num_poi = 0
num_non_poi = 0

for k in data_dict.keys():
    if data_dict[k]['poi'] == 1:
        num_poi += 1
    else: 
        num_non_poi += 1

print "Total Number of Persons of Interest:", num_poi
print "Total Number of Non-Persons of Interest:", num_non_poi

### Task 2: Remove outliers and Cleaned missing values "NaN"
### Removal of Outliers using the pop function
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### loop through keys where total_stock_value or total_payments are greater than 0, for each feature check if the value is "NaN", if so, change to 0.
for k in data_dict:
    if data_dict[k]['total_stock_value'] >0 or data_dict[k]['total_payments'] >0:
        for val in financial_features:
            if data_dict[k][val] == 'NaN':
                data_dict[k][val] = 0

### Corrected the incorrect Values for "BALFER ROBERT" and "BHATNAGAR SANJAY" column by column
data_dict['BELFER ROBERT']['deferred_income'] = -102500
data_dict['BELFER ROBERT']['deferral_payments'] = 0
data_dict['BHATNAGAR SANJAY']['other'] = 0
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BHATNAGAR SANJAY']['expenses'] = 137864
data_dict['BELFER ROBERT']['director_fees'] = 102500
data_dict['BHATNAGAR SANJAY']['director_fees'] = 0
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BHATNAGAR SANJAY']['total_payments'] = 137864
data_dict['BELFER ROBERT']['exercised_stock_options'] = 0
data_dict['BHATNAGAR SANJAY']['exercised_stock_options'] = 15456290
data_dict['BELFER ROBERT']['restricted_stock'] = 44093
data_dict['BHATNAGAR SANJAY']['restricted_stock'] = 2604490
data_dict['BELFER ROBERT']['restricted_stock_deferred'] = -44093
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred'] = -2604490
data_dict['BELFER ROBERT']['total_stock_value'] = 0
data_dict['BHATNAGAR SANJAY']['total_stock_value'] = 15456290

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
dataset = data_dict
my_dataset = data_dict

### create a new feature in the data called total_financial_value which is the sum of a persons total_payments and total_stock_value
for k in my_dataset:
    current_total_stock_value = float(my_dataset[k]['total_stock_value'])
    total_val = float(my_dataset[k]['total_payments'] + current_total_stock_value)
    if total_val == 0:
        my_dataset[k]['total_financial_value'] = 0
    else:
        my_dataset[k]['total_financial_value'] = total_val

my_features_list = features_list + ['total_financial_value']

### Extract features and labels from dataset for local testing using Feature_Importances Feature Selection
### (COMMENTED OUT DUE TO POOR PERFORMANCE RESULTS)
'''
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size = 0.3, random_state=42)

### Find the most important features for use by a DecisionTree algorithm

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### print the number of training points in starter code
print
print "Number of Training data points:", len(features_train)

### prints the accuracy
print
print "Accuracy:", accuracy_score(labels_test, pred)

### stores the feature_importances as a ndarray(multi-dimensional) of shape to the variable important_features
### stores the indices that are used to sort the important_features array
### prints the important features
important_features = clf.feature_importances_
indices = np.argsort(important_features)[::-1]
temp_important_features_list = []
print
print "Important Features:"
### loop for the available features in my_features_list
for i in range(len(my_features_list)-1):
    ### if the features_importances is computed greater than 0.2 for any given feature, add that feature_name to the \
    ### temp_important_features_list
    if important_features[indices[i]]>0.2:
        temp_important_features_list.append(my_features_list[i+1])
    ### use format to print the feature no (index + 1) which excludes "poi", feature_name, and the feature_importances score \
    ### for the index positioned at the current value for i within indices
    print "{} feature {} ({})".format(i+1, my_features_list[i+1], important_features[indices[i]])

### combines the temp list with the poi features list so we can evaluate the precision and recall for dt using the features \
### selected through the features_importances property for the DecisionTree module.
dt_most_important_features_list = poi_features +  temp_important_features_list

### prints the number of features in dt_most_important_features_list
print
print "The Number of most important features according to the DecisionTree feature_importances property:", len(dt_most_important_features_list)

### prints the names of the features
print
print "The features names:", dt_most_important_features_list
'''
### Extract features and labels from dataset for local testing using SelectKBest Feature Selection
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
### Now that use the feature_selector function to find the best features to use
### Features with a variance below 85% are removed
selector = VarianceThreshold(threshold=0.85 * (1 - 0.85))
features = selector.fit_transform(features)

### Univariate feature selection utilizing SelectKBest
selection = SelectKBest()
selection.fit_transform(features, labels)
print
print "Features ranked by score:"
scores = zip(my_features_list[1:],selection.scores_)
scores_sorted = sorted(scores, key = lambda x: x[1], reverse=True)

### Plot to help me know how many features I want to use

feat=[]
val=[]
for f,v in scores_sorted:
    feat.append(f)
    val.append(v)
y_pos = np.arange(len(feat))
plt.bar(y_pos,val,align='center')
plt.xticks(y_pos,feat,rotation=90)
plt.xlabel("Features")
plt.ylabel("SelectKBest Score")
plt.show()

print scores_sorted

### create a ranked_features_list using the map function 
### Return a list of the results of applying the function to the items of the argument sequence(s).  
### If more than one sequence is given, the function is called with an argument list consisting of the corresponding
### item of each sequence, substituting None for missing values when not all sequences have the same length. If the function is None, 
### return a list of the items of the sequence (or a list of tuples if more than one sequence).
ranked_features_list = poi_features + list(map(lambda x: x[0], scores_sorted))


### Feature Selection

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Creation of classifiers
nb_clf = GaussianNB()
dt_clf = DecisionTreeClassifier()
svc_clf = SVC(kernel='rbf', class_weight='balanced')
kNN_clf = KNeighborsClassifier()
ab_clf = AdaBoostClassifier()

### Evaluation of classifiers (commented out for graders convenience)
'''
### naive bayes evaluation
evaluate_optimized_classifiers(nb_clf, SelectKBest(), dataset, features_list)

### decision tree evaluation
evaluate_optimized_classifiers(dt_clf, SelectKBest(), dataset, features_list)

### svc evaluation
evaluate_optimized_classifiers(svc_clf, SelectKBest(), dataset, features_list)

### k-nearest neighbors evaluation
evaluate_optimized_classifiers(kNN_clf, SelectKBest(), dataset, features_list)

### adaBoost neighbors evaluation
evaluate_optimized_classifiers(ab_clf, SelectKBest(), dataset, features_list)

'''
### Evaluation of the DecisionTree classifier utilizing the feature_importances feature selection process
### Extract features and labels from dataset for local testing with the dt_most_important_features_list
### (COMMENTED OUT DUE TO POOR PERFORMANCE RESULTS)
'''
data = featureFormat(my_dataset, dt_most_important_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(features, labels, test_size = 0.3, random_state=42)

### classifier creation, training, predicting, and evaluation results
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print
print "Accuracy Score of DecisionTree Classifier using feature_importances feature selection process:", accuracy_score(labels_test, pred)
print "Recall Score of DecisionTree Classifier using feature_importances feature selection process:", recall_score(labels_test, pred)
print "Precision Score of DecisionTree Classifier using feature_importances feature selection process:", precision_score(labels_test, pred)
'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=0.55,
          n_estimators=100, random_state=None)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, dataset, features_list)