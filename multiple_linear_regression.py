# run using kaggle in individual cells 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv("../input/las-vegas-sold-properties/las_vegas_sold_properties.csv")

df.head()

null_pct = df.isnull().sum() / len(df)
null_pct = null_pct[null_pct>0]
null_per = null_pct * 100
null_per

df = df.drop(["SALE TYPE", "MLS#","LOCATION","INTERESTED","FAVORITE","SOURCE","NEXT OPEN HOUSE START TIME","NEXT OPEN HOUSE END TIME","STATUS","STATE OR PROVINCE","CITY","HOA/MONTH","SOLD DATE","ADDRESS","LOT SIZE","URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)","DAYS ON MARKET"],axis=1)
df.head()

df.shape

print(df.dtypes)

categoricals = ['PROPERTY TYPE']

for col in categoricals:
    df[col] = df[col].astype('category')

for col in df.columns:
    print(col, df[col].nunique())

df.describe()

df.info()

df.corr()

# graphs 

plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True)
plt.show()

plt.figure(figsize=(12,5))
sns.displot(df['PRICE'], bins=40, rug=True)
plt.show()

plt.figure(figsize=(12,5))
sns.displot(df['BEDS'], bins=40, rug=True)
plt.show()

plt.figure(figsize=(12,5))
sns.displot(df['BATHS'], bins=40, rug=True)
plt.show()

sns.distplot(df['YEAR BUILT'], bins=100)

sns.distplot(df['PRICE'], bins=100)

plt.figure(figsize=(10,6))
sns.boxplot(x='BEDS', y='SQUARE FEET', data=df)
plt.title('Comparisons')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='BEDS', y='PRICE', data=df)
plt.title('Comparisons')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='BATHS', y='PRICE', data=df)
plt.title('Comparisons')
plt.show()

sns.regplot(x="SQUARE FEET", y="PRICE", data=df)
plt.ylim(0,)

sns.regplot(x="BEDS", y="PRICE", data=df)
plt.ylim(0,)

sns.regplot(x="BATHS", y="PRICE", data=df)
plt.ylim(0,)

#testing correlations
features =["SQUARE FEET", "BATHS","BEDS" ,"YEAR BUILT"]    

simple_lg_baths_v_price = LinearRegression()
X = df[['BATHS']]
Y = df['PRICE']
print("Bathrooms v Price")
simple_lg_baths_v_price.fit(X,Y)
simple_lg_baths_v_price.score(X,Y)

simple_lg_beds_v_price = LinearRegression()
X = df[['BEDS']]
Y = df['PRICE']
print("Bedrooms v Price")
simple_lg_beds_v_price.fit(X,Y)
simple_lg_beds_v_price.score(X,Y)

simple_lg_sqfeet_v_price = LinearRegression()
X = df[['SQUARE FEET']]
Y = df['PRICE']
print("Square feet v Price")
simple_lg_sqfeet_v_price.fit(X,Y)
simple_lg_sqfeet_v_price.score(X,Y)

# linear regression
X = df[['SQUARE FEET', 'BATHS']]
y = df['PRICE']

regr = linear_model.LinearRegression()
model = regr.fit(X, y)

X = df[['SQUARE FEET', 'BATHS']].values.reshape(-1,2)
Y = df['PRICE']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

x = X[:, 0]
y = X[:, 1]
z = Y

xx_pred = np.linspace(4, 9, 30)  
yy_pred = np.linspace(2, 5, 30)  

xx_pred, yy_pred = np.meshgrid(xx_pred, yy_pred)
model_flattened = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

predicted = model.predict(model_flattened)

r2 = model.score(X, Y)
print(r2*100)

# saving the model
import joblib

joblib_file = "joblib_RL_Model.pkl"  
joblib.dump(model, joblib_file)

joblib_LR_model = joblib.load(joblib_file)

joblib_LR_model

# testing 
def ml_linear_regression(square_footage, bathrooms):
    try:
        print('The price will be ${:.0f} for a house with {} square feet and {} baths'.format(
            model.predict([[float(square_footage), float(bathrooms)]])[0],
            square_footage, 
            bathrooms))
    except ValueError:
        print('Please enter a correct value')

size = input('What is the square footage of the house? \n')
bathrooms = input('How many bathrooms are in the house? \n')

ml_linear_regression(size, bathrooms)
    
# more results 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error

r2=model.score(x_test, y_test)
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
    
mse = mean_squared_error(y_train, y_pred), mean_squared_error(y_test, y_pred)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_pred), mean_squared_error(y_test, y_pred)))
print('RMSE train: %.3f, test: %.3f' % (rmse(y_train, y_pred), rmse(y_test, y_pred)))
print('MAE train: %.3f, test: %.3f' % (mean_absolute_error(y_train, y_pred), mean_absolute_error(y_test, y_pred)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_pred)))
print("R^2: {}".format(r2))
adj_r2 = 1 - (1 - r2 ** 2) * ((x_train.shape[1] - 1) / (x_train.shape[0] - x_train.shape[1] - 1))
print("Adjusted R^2: {}".format(adj_r2))


