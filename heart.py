import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import scale  
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("heart.csv")
print(df.head())

print(sns.countplot(x='target', hue='sex', data=df))
print(sns.distplot(df['age'], bins=24, color='r'))

class Deep_learning_Model:
    def __init__(self, input_dim,model, X, y):
        self.input_dim = input_dim
        self.model = model
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        self.model.add(tf.keras.layers.Dense(units=128,activation="relu",input_dim=self.input_dim))
        self.model.add(tf.keras.layers.Dropout(rate=0.1))
        self.model.add(tf.keras.layers.Dense(units=128,activation="relu"))
        self.model.add(tf.keras.layers.Dropout(rate=0.1))
        self.model.add(tf.keras.layers.Dense(units=18,activation="relu"))
        self.model.add(tf.keras.layers.Dropout(rate=0.1))
        self.model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
        self.model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
        
        self.model.fit(self.X_train, self.y_train,epochs=100,batch_size=32)
        print(f"{self.model_str()} Model Trained..")
        self.y_pred = np.around(self.model.predict(self.X_test))
        
    def model_str(self):
        return str(self.model.__class__.__name__)
        
    def scores_(self, cv=5):
        print(self.model_str() + "\n" + "="*60)
        
        cv_acc1 = precision_score(self.y_test,
                                   self.y_pred)
        
        cv_acc2 = recall_score(self.y_test,
                               self.y_pred)

        print(f"{self.model_str()} precision score: {cv_acc1}")
        print(f"{self.model_str()} recall score: {cv_acc2}")
                 
    def accuracy(self):
        accuarcy = accuracy_score(self.y_test, self.y_pred)
        print(self.model_str() + " Model " + "Accuracy is: ")
                            
        return accuarcy
        
    def confusionMatrix(self):        
        plt.figure(figsize=(5, 5))
        mat = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(mat.T, square=True, 
                    annot=True, 
                    cbar=False, 
                    xticklabels=["Haven't Disease", "Have Disease"], 
                    yticklabels=["Haven't Disease", "Have Disease"])
        
        plt.title(self.model_str() + " Confusion Matrix")
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()
        
    def classificationReport(self):
        print(self.model_str() + " Classification Report" + "\n" + "="*60)
        print(classification_report(self.y_test, 
                                    self.y_pred, 
                                    target_names=['Non Disease', 'Disease']))
    
    def rocCurve(self):
        fpr, tpr, thr = roc_curve(self.y_test, self.y_pred)
        lw = 2
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 
                 color='darkorange', 
                 lw=lw, 
                 label="Curve Area = %0.3f" % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], color='green', 
                 lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.model_str() + ' Receiver Operating Characteristic Plot')
        plt.legend(loc="lower right")
        plt.show()
        
        
x = scale(np.array(df.drop("target",axis=1)))
y = np.array(df["target"])
model = tf.keras.models.Sequential()
input_dim = 13
model = model
X = x
y = y

model = Deep_learning_Model(input_dim,model,x,y)
model.scores_()
model.classificationReport()
model.confusionMatrix()

def real_accuracy():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=128,activation="relu",input_dim=13))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(units=128,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(units=18,activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    
    return model

X = scale(np.array(df.drop("target",axis=1)))
y = np.array(df["target"])
clf = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=real_accuracy,epochs=100,batch_size=32)
accuracy = cross_val_score(clf,x,y,cv=5,n_jobs=-1).mean()
print("real_acc",accuracy)
print("")

class Machine_learning_model(object):
    def __init__(self, model , X, y):
        self.model = model
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model.fit(self.X_train,self.y_train)
        print(f"{self.model_str()} Model Trained..")
        self.y_pred = self.model.predict(self.X_test)

    def model_str(self):
            return str(self.model.__class__.__name__)
        
    def scores_(self, cv=5):
        print(self.model_str() + "\n" + "="*60)
        
        cv_acc1 = precision_score(self.y_test,
                                   self.y_pred)
        
        cv_acc2 = recall_score(self.y_test,
                               self.y_pred)

        print(f"{self.model_str()} precision score: {cv_acc1}")
        print(f"{self.model_str()} recall score: {cv_acc2}")
    def accuracy(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(self.model_str() + " Model " + "Accuracy is: ")
        print(accuracy)
        return accuracy
        
    def confusionMatrix(self):        
        plt.figure(figsize=(5, 5))
        mat = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(mat.T, square=True, 
                    annot=True, 
                    cbar=False, 
                    xticklabels=["Haven't Disease", "Have Disease"], 
                    yticklabels=["Haven't Disease", "Have Disease"])
        
        plt.title(self.model_str() + " Confusion Matrix")
        plt.xlabel('Predicted Values')
        plt.ylabel('True Values')
        plt.show()
   
 

clf2 = GradientBoostingClassifier()
clf3 = KNeighborsClassifier()
clf4 = GaussianNB()

Machine_learning_model = Machine_learning_model(clf2,x,y)
Machine_learning_model.scores_()
Machine_learning_model.accuracy()
Machine_learning_model.confusionMatrix()









 