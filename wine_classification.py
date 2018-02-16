import pandas as pd #Pandas is a great library for data manipulation and analysis

#Importing data
data_f = pd.read_csv('wine.csv', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])

#Have a look at the data. Before preprocessing, always have one pass at the data
data_f.head()
data_f.describe().transpose()
data_f.shape

#Assign X and y - X has all the features which will help us predict the target variable y and y is the feature which we want to predict
#Here we drop the last column i.e. Cultivator as it is the target variable.
X = data_f.drop('Cultivator',axis =1)
y= data_f['Cultivator']
# We can use iloc function too - to assign X and y

#Now lets split the data into train and test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

#Now its time to preprocess - normalise the data
#We use the inbuilt StandardScaler to normalise the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Fit the training data
scaler.fit(X_train)
#Now apply the transformation
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Its time to train the model
from sklearn.neural_network import MLPClassifier
#create the instance/object of the mlp model
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13,13), max_iter =5000) #You can play around with different parameters
mlp.fit(X_train,y_train)

#Predict
predictions = mlp.predict(X_test)
print(predictions)

#create the confusion matrix and classification report
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
