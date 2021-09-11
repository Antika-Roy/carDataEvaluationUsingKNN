import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())

le = preprocessing.LabelEncoder()
#buying,maint,door,persons,lug_boot,safety,class
#returns numpy array
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
#print(cls)


X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features, zip works like list of objects
y = list(cls)  # labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print("accuracy: ", accuracy)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    # This will display the predicted class, our data and the actual class
    # We create a names list so that we can convert our integer predictions into their string representation
    n = model.kneighbors([x_test[x]], 9, True)
    # here [x_test[x]] is used as two dimensional array which is required for kneighbours() function
    print("N: ", n)


