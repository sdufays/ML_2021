# run using kaggle through individual cells 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import plotly.graph_objects as go
import joblib
%matplotlib inline



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv("../input/classification-dataset/247711.csv")
df.head(50)

df.shape

null_pct = df.isnull().sum() / len(df)
null_pct = null_pct[null_pct>0]
null_per = null_pct * 100
null_per

print(df.dtypes)
df = df.drop(["name"],axis=1)

df['Gender'].astype(str)
smap = {'Male':0, 'Female': 1}
df['Sex'] = df['Gender'].map(smap)
df = df.drop(["Gender"],axis=1)
df.head()

for col in df.columns:
    print(col, df[col].nunique())

df.describe()
df.info()
df = df.abs()

prof = ProfileReport(df)
prof.to_file(output_file='output.html')

df.corr()

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)
plt.show()

#graphs
fig = go.Figure(data=go.Scattergl(
    y = df['Sex'],
    x = df['EstimatedSalary'],
    text=df.index,
    mode='markers',
    marker=dict(
        color=df['Purchased'],
        colorscale=["red", "green"],
        line_width=1
    )
))

fig.update_layout(
    title="Classification of buyers (green) and non buyers (red)",
    xaxis_title="Estimated Salary",
    yaxis_title="Sex",
)

fig.show()


fig = go.Figure(data=go.Scattergl(
    y = df['Age'],
    x = df['EstimatedSalary'],
    text=df.index,
    mode='markers',
    marker=dict(
        color=df['Purchased'],
        colorscale=["red", "green"],
        line_width=1
    )
))

fig.update_layout(
    title="Classification of buyers (green) and non buyers (red)",
    xaxis_title="Estimated Salary",
    yaxis_title="Age",
)

fig.show()

import plotly.graph_objects as go

fig = go.Figure(data=go.Scattergl(
    y = df['Sex'],
    x = df['Age'],
    text=df.index,
    mode='markers',
    marker=dict(
        color=df['Purchased'],
        colorscale=["red", "green"],
        line_width=1
    )
))

fig.update_layout(
    title="Classification of buyers (green) and non buyers (red)",
    xaxis_title="Age",
    yaxis_title="Sex",
)

fig.show()

#distributions
sns.set_style("whitegrid");
sns.pairplot(df, hue="Purchased", height=3);
plt.show()

sns.FacetGrid(df,hue='Purchased',height=5).map(sns.distplot,'Sex').add_legend()
sns.FacetGrid(df,hue='Purchased',height=5).map(sns.distplot,'EstimatedSalary').add_legend()
sns.FacetGrid(df,hue='Purchased',height=5).map(sns.distplot,'Age').add_legend()
sns.jointplot(data=df, x="EstimatedSalary", y="Age", hue="Purchased", kind="kde")
sns.jointplot(data=df, x="EstimatedSalary", y="Sex", hue="Purchased", kind="kde")

# split data 
X = df[['EstimatedSalary','Sex','Age']]
Y = df['Purchased']

# models 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# logistic regresison test
logreg = LogisticRegression()

# fit the model with data
logreg.fit(x_train,y_train)
print(logreg.score(x_test,y_test))

y_pred=logreg.predict(x_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# true positive rate against the false positive rate, shows the tradeoff between sensitivity and specificity.

# testing k-Nearest (not most ideal)
X = df[['EstimatedSalary','Sex','Age']]
Y = df['Purchased']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(result), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

# testing random forest 
X = df[['EstimatedSalary','Sex','Age']]
Y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

scaler = StandardScaler()
scaler.fit(x_train)
X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=10)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

clf.score(X,Y)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#result = confusion_matrix(y_test, y_pred)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(result), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

joblib_file = "joblib_kN_Model.pkl"  
joblib.dump(classifier, joblib_file)

joblib_LR_model = joblib.load(joblib_file)
joblib_LR_model

def ml_prediction(salary, gender, age):
    if gender == 0: 
        sex = "male"
    else if gender == 1: 
        sex = "female"
    prediction = classifier.predict([[int(salary), int(gender),int(age)]])
    #print(prediction)
    if prediction == [0]: 
        pred = "would not"
    else if prediction == [1]: 
        pred = "would"
    print("A " + sex + " that is " +  str(age)  + " years old with an estimated salary of " + str(salary) + " " + pred + " purchase a cog")

salary = input("Salary: ")
gender = input("Gender (0 for male and 1 for female) :")
age = input("Age: ")
ml_prediction(int(salary),int(gender),int(age))
