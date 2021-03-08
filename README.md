# Winery
source: https://github.com/CodeAcademyBerlin/Data-Science/blob/master/Module%201/Week%202/Task3-email.md

presentation: 8-12 slides (final: ) 3-4 slides (small presentation: 15th Feb.)

consider the customer's deadline than internal team deadline

predict the proper price

follow the guidelines and reply to Mark's email as you progress with the project.
## backgrond - email - Blueberry Winery
a start-up company

is trying to enter the business with a good amount of analytics & research on domain knowledge.

wine-maker (not distributior, not seller)

they want to build a Wine Quality Analytics system which can help them determine the quality of wines produced based on ingredients & composition.

companies should start with preliminary quality research on either market / wine quality.

companies trust data and machine learning to come to a decision instead of expert recommendation / analysis.

From a Sales & Marketing perspective, put a proper price tag for a bottle of wine.

there should be no mismatch between quality and price of the product which is one of the major factors contributing to 'Customer Satisfaction'.

many other factors that contribute to the quality.

The age of a bottle of wine :
time changes the taste of the fruit flavors in a wine as well as
time reduces the acidity and tannin in a wine
As the acidity and tannin are reduced, the wine becomes rounder and smoother.
The analysis should not restrict to the technical specifications but include the business / domain aspects

help BlueBerry Winery with Business Decisions

##  Prepare/clean dataset
load the dataset

understand the composition of red wine and white wine

red wine:
white wine:
combine two dataframes into one "wines"
  # load dataset
import pandas as pd
import numpy as np

red_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep= ';')
white_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep= ';')

red_wine.dtypes
red_wine.astype('int64').dtypes
red_wine.head()

white_wine.dtypes
white_wine.astype('int64').dtypes
white_wine.head()

# add a column 'wine_type' to dataframe 'red_wine' and 'white_wine'

## create a new variable 'wine_type'
red_wine['wine_type'] = 'red'

# bucket wine quality scores into qualitative quality labels
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
if value <= 5 else 'medium'
if value <= 7 else 'high')
red_wine

red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'],
categories=['low', 'medium', 'high'])
red_wine['quality_label'].dtype

# create a new variable 'wine_type'
white_wine['wine_type'] = 'white'

# bucket wine quality scores into qualitative quality labels

white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
if value <= 5 else 'medium'
if value <= 7 else 'high')

white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'],
categories=['low', 'medium', 'high'])

white_wine

# combine two dataframe (red_wine) & (white_wine)
wines = pd.concat([red_wine, white_wine])

# re-shuffle records just to randomize data points
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)
wines

# if wine_type is removed after concat(): wines['wine_type'] = wines['wine_type'].replace(np.nan, 'red')

# wines['wine_type'] = wines['wine_type'].replace(wines['wine_type'] = np.nan, 'red')
# wines.loc[wines.wine_type = np.nan].fillna('red')
# wines.fillna(method='WORD', inplace=True)
# wines.fillna(0, inplace=True)
# wines[wines['wine_type'] != 'white']

# wines['ID'] = wines.index
# wines['wine_type'] = wines[wines.isna().any(axis=1)].replace(np.nan,'red')
# wines['wine_type'] = wines['ID'].map(wines_red.set_index('ID')['wine_type'])
score the quality of wine
Wines with a quality score of 3, 4, and 5 are low quality;
score of 6 and 7 are medium quality;
score of 8 and 9 are high quality wines.
### Data Exploration Analysis
# load library/packages 
import matplotlib as plt
import pandas as pd
import numpy as np

# load dataset 'wines'

# print first, last 10 records
wines.head(10)
wines.tail(10)

# Check wines.info(), wines.shape

# Observe if there is missing values
# https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html

wines.groupby('wine_type')['pH'].mean().plot(kind = 'bar')


# create a table like below to compare the Descriptive Statistics between Red wine and White wine.
# print first 10 records
wines.head(10)

# print last 10 records
wines.tail(10)

# create a table like below to compare the Descriptive Statistics between Red wine and White wine.
subset = ['residual sugar', 'total sulfur dioxide', 'sulphates', 'alcohol', 'volatile acidity', 'quality']

df1 = red_wine[subset].describe()
df2 = white_wine[subset].describe()
df_des_subset = pd.concat([df1, df2],keys=['Red wine statistics', 'White wine statistics'],axis=1)
df_des_subset
try to be more innovative using some other way of representations.
example shows that the mean value of sulfates and volatile acidity seem to be higher in red wine as compared to white wine. Visualize more observations : Come up with few visualizations which compare low, medium and high quality wine.
# display the distribution of the values in a selected column, using histogram
red_wine['total sulfur dioxide'].hist(bins = 100);
red_wine['chlorides'].hist(bins = 100);

import matplotlib.pyplot as plt
# plot bar graphs
ph_wines = wines.groupby('wine_type')['pH'].min().plot(kind = 'bar');

ph_wines2 = wines.groupby('wine_type')['pH'].min()
plt.bar(ph_wines2.index, ph_wines2.values);

wines.dtypes

# wines =wines.astype(np.float)
# to_transform = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide'] 
# wines[to_transform] = pd.to_numeric(wines[to_transform])
# wines[['fixed acidity', 'volatile acidity']] = wines[['fixed acidity', 'volatile acidity']].apply(pd.to_numeric)

fig1, ax = plt.subplots(1,2)
ax[0].boxplot(wines.loc[wines['wine_type'] == 'red','fixed acidity']);
ax[0].set_title('Basic alcohol Plot Red wine')
ax[1].boxplot(wines.loc[wines['wine_type'] == 'white','fixed acidity']);
ax[1].set_title('Basic alcohol  Plot white Wine');#

fig2, ax = plt.subplots(1,2)
ax[0].hist(wines.loc[wines['wine_type'] == 'red','alcohol'], bins = 50, color = 'red');
ax[0].set_title('Basic alcohol Plot Red wine')
ax[1].hist(wines.loc[wines['wine_type'] == 'white','alcohol'], bins = 50, color = 'blue');
ax[1].set_title('Basic alcohol  Plot white Wine');
# volatile acidity by quality

fig3, ax = plt.subplots()

x = wines['alcohol']
y = wines['quality']
colors = {'high':'red', 'medium':'green', 'low':'blue'}

plt.scatter(x, y, s=100*wines['quality'], alpha=0.3,
            c= wines['quality_label'].map(colors),
            cmap='viridis')
plt.colorbar();

fig4, ax = plt.subplots()
fig4.suptitle('Quality x volatile acidity')

colors = {'high':'red', 'medium':'green', 'low':'blue'}
#wine_type

grouped = wines.groupby('quality_label')
for key, group in grouped:
  group.plot(ax=ax, kind='scatter', x='quality', y='volatile acidity', label=key, color=colors[key])
  #group.plot(ax=ax, kind='scatter', x='quality', y='fixed acidity', label=key, color=colors[key])

plt.show()

boxplot = wines.boxplot(column=['fixed acidity' ], by = 'wine_type')
boxplot

fig4, axes = plt.subplots(nrows=1, ncols=3) # create 1x2 array of subplots
fig4.suptitle('Acidity by Wine Quality')

wines.boxplot(column='volatile acidity', by='quality_label', ax=axes[0]) # add boxplot to 1st subplot
wines.boxplot(column='fixed acidity', by='quality_label', ax=axes[1]) # add boxplot to 2nd subplot
wines.boxplot(column='citric acid', by='quality_label', ax=axes[2]) # add boxplot to 2nd subplot

plt.show()
# example shows that the mean value of sulfates and volatile acidity seem to be higher in red wine as compared to white wine. Visualize more observations 
# Come up with few visualizations which compare low, medium and high quality wine.

import seaborn as sns
data = wines
sns.boxplot(data = data , x = 'wine_type' , y = 'volatile acidity' , hue = 'quality_label' , notch = True )

fig5, ax = plt.subplots()
plt.title("Quality - Voltaile acidity", fontsize=14)

ax.scatter(x=wines[wines['wine_type'] == 'red']['quality'], y=wines[wines['wine_type'] == 'red']['volatile acidity'])
ax.scatter(x=wines[wines['wine_type'] == 'white']['quality'], y=wines[wines['wine_type'] == 'white']['volatile acidity'])
ax.legend(labels=['Red','White'], loc='lower left', fontsize=12)

# defining labels 
quality = ['high', 'medium', 'low'] 

# portion covered by each label 
wines_q = wines[wines['quality_label']=='high']['quality'].mean()
wines_m = wines[wines['quality_label']=='medium']['quality'].mean()
wines_l = wines[wines['quality_label']=='low']['quality'].mean()

quality_mean = [wines_q, wines_m, wines_l]

# color for each label 
colors = ['red', 'green', 'b'] 
  
# plotting the pie chart 
plt.pie(quality_mean, labels = quality, colors=colors,  
        startangle=90, shadow = True, autopct = '%1.1f%%') 
  
# plotting legend 
plt.legend() 
  
# showing the plot 
plt.show();