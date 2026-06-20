import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv("records.csv")
print("Records:",data)

X=data[['HoursStudied','Attendance']]
Y=data[['Result']]

print("\nFeatures:",X)
print("\nTarget:",Y)

x_train,x_test,y_train,y_test=train_test_split(
    X,
    Y,
    test_size=0.5,
    random_state=42

)
print("\nTraining Data:",x_train)
print("\nTesting Data:",x_test)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)
