import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.tree import DecisionTreeClassifier 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
url = 'https://github.com/plotly/datasets/raw/master/diabetes.csv'
data = pd.read_csv(url)
X = data.drop('Outcome', axis=1) 
y = data['Outcome'] 
print(data.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42) 
clf.fit(X_train, y_train)
predictions = clf.predict(X_test) 
accuracy = accuracy_score(y_test, predictions) 
print(f"Accuracy: {accuracy * 100:.2f}") 
