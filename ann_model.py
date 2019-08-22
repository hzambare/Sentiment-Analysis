import time
start = time.time()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import accuracy_score
#from keras.utils import to_categorical
#from keras import backend as Keras

#file = pd.read_csv("reviews2.csv")
#file = pd.read_csv("reviews4.csv")
test_file = pd.read_csv("nlp_test3_use.csv")
reviews = list(test_file.iloc[:,0])[50000:75000]
ratings = list(test_file.iloc[:,1])[50000:75000]
test_reviews = list(test_file.iloc[:,0])[75000:80000]
test_ratings = list(test_file.iloc[:,1])[75000:80000]

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
x = cv.fit_transform(x).toarray()
y = np.array(ratings)
#y = to_categorical(y1)

a = cv.fit_transform(a).toarray()
b1 = np.array(test_ratings)
#b = to_categorical(b1)

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.4)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Activation

classifier = Sequential()


classifier.add(Dense(64, kernel_initializer = 'orthogonal', input_dim = 1500)) # input layer
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())

classifier.add(Dense(34, kernel_initializer = 'orthogonal')) # hidden layer
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())

classifier.add(Dense(18, kernel_initializer = 'orthogonal')) # hidden layer
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())

classifier.add(Dense(3, kernel_initializer = 'orthogonal')) # hidden layer
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())

classifier.add(Dense(3, kernel_initializer = 'orthogonal')) # hidden layer
classifier.add(Activation("relu"))
classifier.add(BatchNormalization())

classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'softmax')) # output layer

#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # creates ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # creates ANN

classifier.fit(x_train, y_train, batch_size = 100, epochs = 50, validation_data=(x_valid, y_valid))

#y_pred = classifier.predict(x_test)
y_pred = classifier.predict(a)
#y_pred = classifier.predict(x_valid)
y_pred = np.argmax(y_pred, axis=1)

accuracy_ann = accuracy_score(b1, y_pred)*100

print("Test Accuracy: {}%".format(accuracy_ann))
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, x_test)

end = time.time()
print("Runtime: {} minutes".format((end - start)/60))