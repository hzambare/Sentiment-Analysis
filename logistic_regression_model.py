import time
start = time.time()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict # ,cross_val_score
from sklearn import metrics

test_file = pd.read_csv("nlp_test_use.csv")
reviews = list(test_file.iloc[:,0])[500000:750000]
ratings = list(test_file.iloc[:,1])[500000:750000]
test_reviews = list(test_file.iloc[:,0])[750000:800000]
test_ratings = list(test_file.iloc[:,1])[750000:800000]

x = []
for i in reviews:
    i = i[1:-1]
    i = i.replace(",","")
    i = i.replace("'","")
    i = i.lower()
    x.append(i)
a = []
for u in test_reviews:
    u = u[1:-1]
    u = u.replace(",","")
    u = u.replace("'","")
    u = u.lower()
    a.append(u)

cv = TfidfVectorizer(max_features = 1500)
X = cv.fit_transform(x).toarray()
y = np.array(ratings)

a = cv.fit_transform(a).toarray()
b1 = np.array(test_ratings)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)
#log_reg = LogisticRegression(solver = 'liblinear', multi_class = 'ovr', max_iter = 100, random_state=0)
log_reg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter = 350, random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)
#accuracy_log_reg = accuracy_score(y_test, y_pred)*100

predictions = cross_val_predict(log_reg, X_valid, y_valid, cv=3)
cross_val_accuracy = metrics.accuracy_score(y_valid, predictions)
p = log_reg.predict_proba(a)
#test_accuracy = metrics.accuracy_score(b1, p)
print("Test Set Size:",int(len(reviews)*0.2))
#print("Logistic Regression Accuracy: {}%".format(accuracy_log_reg))
print("Cross Validation Accuracy: {}%".format(cross_val_accuracy*100))
#print("Test Accuracy: {}%".format(test_accuracy*100))
end = time.time()
print("Runtime: {} minutes".format((end - start)/60))