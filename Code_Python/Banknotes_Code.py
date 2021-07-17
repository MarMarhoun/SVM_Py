# =============================================================================
# # Libraries for analysises
# =============================================================================

import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix  

# =============================================================================
# # Libraries for visuals
# =============================================================================

import matplotlib.pyplot as plt  

# =============================================================================
# Load the data
# =============================================================================
bankdata = pd.read_csv("bill_authentication.csv") 

# =============================================================================
# Serparat data into two categorises (class and attributs)
# =============================================================================
X = bankdata.drop('Class', axis=1)  
y = bankdata['Class'] 


# =============================================================================
# Split data into train/test sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

# =============================================================================
# Train the Model.
# =============================================================================

# SVM classifier using linear kernel
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train) 


# SVM classifier using rbf kernel
svclassifier_RBF = SVC(kernel='rbf')  
svclassifier_RBF.fit(X_train, y_train) 

# SVR classifier using these parameters (C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.2, gamma='scale', 
#                                         kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

SVRclassifier = SVR()
SVRclassifier.fit(X_train, y_train) 


# =============================================================================
# Generate predictions from the model
# =============================================================================

y_pred = svclassifier.predict(X_test)

y_pred_RBF = svclassifier_RBF.predict(X_test)

y_pred_SVR = SVRclassifier.predict(X_test)



# =============================================================================
# Convert dataframe into matrix
# =============================================================================
xx=X_test.as_matrix()


# =============================================================================
# Plot some data
# =============================================================================

# Plot SVM Classifier using linear kernel

print("\n\n\n----------------SVM Classifier using linear kernel--------")
plt.scatter(xx[:, 0], xx[:, 3],c=y_test)
plt.title("Real case")
plt.show() 
plt.scatter(xx[:, 0], xx[:, 3],c=y_pred)
plt.title('Predictions')
plt.show() 

print("******* Matrice de confusion *********")
print(confusion_matrix(y_test,y_pred))  
print("\n\n**** Rapport de la classification ***")
print(classification_report(y_test,y_pred))


# Plot SVM Classifier using rbf kernel

print("\n\n\n---------------- SVM Classifier using rbf kernel--------")
plt.scatter(xx[:, 0], xx[:, 3],c=y_test)
plt.title("Real case")
plt.show() 
plt.scatter(xx[:, 0], xx[:, 3],c=y_pred_RBF)
plt.title('Predictions')
plt.show() 

print("******* Matrice de confusion avec RBF kernel *********")
print(confusion_matrix(y_test,y_pred_RBF))  
print("\n\n**** Rapport de la classification avec RBF kernel***")
print(classification_report(y_test,y_pred_RBF))

# Plot SVR  using linear kernel

print("\n\n\n---------------- SVR using rbf kernel --------")
plt.scatter(xx[:, 0], xx[:, 3],c=y_test)
plt.title("Real case")
plt.show() 
plt.scatter(xx[:, 0], xx[:, 3],c=y_pred_SVR)
plt.title('Predictions')
plt.show() 

print("******* Matrice de confusion SVR*********")
print(confusion_matrix(y_test,y_pred_SVR.round()))  
print("\n\n**** Rapport de la classification SVR ***")
print(classification_report(y_test,y_pred_SVR.round()))



## Made by MarMarhoun
