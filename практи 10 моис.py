import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/seker/OneDrive/Рабочий стол/уроки/diabetes.csv")
X = df[['Glucose', 'Age']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
plt.scatter(X_test['Age'], y_prob)
plt.xlabel("Возраст")
plt.ylabel("Вероятность диабета")
plt.title("Связь между возрастом и вероятностью диабета")
plt.show()
