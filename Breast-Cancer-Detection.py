#We use data from sklearn library itself
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#Lets do our first pass at the data. We look at different attributes and get basic info about the data.
print(cancer.keys())
print(cancer['data'])


#The DESCR column contains basic description about the data
print(cancer['DESCR'])

print(cancer['data'].shape)
print(cancer['target'])

#Lets split the data into predictor features and target feature
X = cancer['data']
y = cancer['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)


# We will now scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#Its time to train the model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,10,15))
mlp.fit(X_train,y_train)


#Predict
predictions = mlp.predict(X_test)
print(predictions)

# Lets see how we did at the prediction - confusion matrix& classification report
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
