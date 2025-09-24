import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris_flower_df = pd.read_csv("Iris.csv")

# print(iris_flower_df.head())

# print(iris_flower_df.isnull().sum())

# print(iris_flower_df.info())

iris_flower_df = iris_flower_df.drop(['Id'], axis=1)

iris_flower_df['Species'] = iris_flower_df['Species'].astype(str)

# sns.heatmap(iris_flower_df.select_dtypes(include='number').corr(method='pearson'), annot=True)

"""
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

for col in columns:
    Q1 = np.percentile(iris_flower_df[col], 25, interpolation='midpoint')
    Q3 = np.percentile(iris_flower_df[col], 75, interpolation='midpoint')
    IQR = Q3 - Q1

    upper = np.where(iris_flower_df[col] >= (Q3 + 1.5 * IQR))
    lower = np.where(iris_flower_df[col] <= (Q1 - 1.5 * IQR))

    iris_flower_df.drop(upper[0], inplace=True)
    iris_flower_df.drop(lower[0], inplace=True)

for col in columns:
    sns.boxplot(x=iris_flower_df[col])
    plt.title("Boxplot for " + col)
    plt.show()
"""

X = iris_flower_df[['PetalLengthCm', 'PetalWidthCm']]
y = iris_flower_df['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_preds)
svm_cm = confusion_matrix(y_test, svm_preds)

print("SVM Accuracy:", svm_acc)

print("Confusion Matrix:\n", svm_cm)
print("Classification Report:\n", classification_report(y_test, svm_preds))

while True:
    try:
        petal_length = float(input("Enter Petal Length (cm): "))
        petal_width = float(input("Enter Petal Width (cm): "))
        prediction = svm.predict([[petal_length, petal_width]])[0]
        print("Predicted Species: " + prediction)
        break
    except ValueError:
        print("Please enter valid numeric values for Petal Length and Width.")
