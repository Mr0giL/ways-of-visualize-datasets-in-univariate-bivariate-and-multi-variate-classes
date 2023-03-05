#--------------! Library import !--------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA , IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import prince
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
    ##statistical test libraries
from scipy import stats
import datetime
dt = datetime.datetime(9999,12,31,23,59,59)
sb.set()
#--------------! dataset import !--------------#
df = pd.read_csv('Car Price (2).csv')
df = df.drop('ID',axis=1,inplace=False) 
    #dataframe with no ID column

#--------------! basic descriptive statistics import !--------------#
print(df.info())
print(df.head())
print(df.describe())
print('company uniques : ', df['Company'].unique())
print('model uniques is :', df['Model'].unique())
print('detail uniques is :', df['Detail'].unique())

    ## ProductionYear have unnormal mean
    ## Milage have negative values
    ## most of body conditions is 1
    ## we have lots of 0 in Price
    ## we have nan (NaN) and Nan in Company 
    ## we have some uncapitalized models such m-10 , m-11 and a N-11 in Models
    ## we have some uncapitalized details such d-3 in Details

#--------------! merge some mistakes such as capitalize all characters and merge some columns !--------------#
df = df.replace('x-4','C-4') #change this classes with few counts, This assumed as mistyping. 'x' & 'c' and 'm' & 'n' are nearby in keyboars
df = df.replace('N-11','M-11')
df = df.replace('c-4','C-4')
df = df.replace('d-3','D-3')
df = df.replace('m-11','M-11')
df = df.replace('m-10','M-10')

#--------------! dealing with missing values !--------------#
    ## missing values is only in 'Model' 
df = df.replace('Nan', np.nan)
print('number of nulls is :',df['Company'].isna().sum())

    #number of missing values in less than 5% of dataset so we delete it's rows
df = df.dropna()

#--------------! create dummy variables for non-numeric values !--------------#
obj_df = df.select_dtypes(include='object').copy()
obj_df.columns = (['Date_encoded','Company_encoded','Model_encoded','Detail_encoded'])
obj_df = obj_df.apply(LabelEncoder().fit_transform).astype('int64')
df = df.join(obj_df)


#--------------! create dummy variables for non-numeric values !--------------#
dates = df['Date']
df = df.drop('Date',axis=1,inplace=False) #Deal with time later

# #--------------! visualise dataset !--------------#
#     #none graphical analysis
#         ###variaance and SD of numeric values
print('variance is :', df.var())
print('std is :',df.std())

        ###cross tabluation of non-numerics
print(pd.crosstab(df['Company'],df['Model']))
print(pd.crosstab(df['Company'],df['Detail']))
print(pd.crosstab(df['Detail'],df['Model']))
        ###value counts of each non-numerical columns
print(df['Company'].value_counts())
print(df['Detail'].value_counts())
print(df['Model'].value_counts())

print(df.info())
    # independent test 
rvs1 = stats.norm.rvs(loc = 5, )
    #graphical analysis - univariate categorical
fig = plt.figure()
ax1 = plt.subplot2grid((2,3),(0,0))
plt.pie(data = df,x=df['Company'].value_counts(),labels=df['Company'].unique())
plt.title('univariate - Company')
ax1 = plt.subplot2grid((2,3),(0,1))
plt.pie(data=df, x=df['Model'].value_counts(),labels=df['Model'].unique())
plt.title('univariate - Model')
ax1 = plt.subplot2grid((2,3),(0,2))
plt.pie(data=df, x=df['Detail'].value_counts(),labels=df['Detail'].unique())
plt.title('univariate - Detail')
ax1 = plt.subplot2grid((2,3),(1,0))
plt.bar(data = df,x=df['Company'].unique(),height=df['Company'].value_counts())
plt.title('univariate - Company')
ax1 = plt.subplot2grid((2,3),(1,1))
plt.bar(data = df,x=df['Model'].unique(),height=df['Model'].value_counts())
plt.title('univariate - Model')
ax1 = plt.subplot2grid((2,3),(1,2))
plt.bar(data = df,x=df['Detail'].unique(),height=df['Detail'].value_counts())
plt.title('univariate - Detail')

#  #graphical analysis - univariate nominal
fig = plt.figure()
ax2 = plt.subplot2grid((2,4),(0,0))
plt.boxplot(data= df, x=df['ProductionYear'], vert=False , whis= 5)
plt.title('univariate - ProdYear')
ax2 = plt.subplot2grid((2,4),(0,1))
plt.boxplot(data= df, x=df['Mileage'], vert=False)
plt.title('univariate - Mileage')
ax2 = plt.subplot2grid((2,4),(0,2))
plt.boxplot(data= df, x=df['BodyCondition'], vert=False)
plt.title('univariate - BodyCondition')
ax2 = plt.subplot2grid((2,4),(0,3))
plt.boxplot(data= df, x=df['Price'], vert=False)
plt.title('univariate - Price')
ax2 = plt.subplot2grid((2,4),(1,1))
sb.histplot(data=df, x= 'Mileage',kde=True)
plt.title('univariate - Mileage')
ax2 = plt.subplot2grid((2,4),(1,2))
sb.histplot(data=df, x= 'BodyCondition',kde=True)
plt.title('univariate - BodyCondition')
ax2 = plt.subplot2grid((2,4),(1,3))
sb.histplot(data=df, x= 'Price',kde=True)
plt.title('univariate - Price')

#  #graphical analysis - univariate encoded-non-nominal
ax3 = plt.subplot2grid((2,3),(0,0))
plt.boxplot(data= df, x=df['Company_encoded'], vert=False , whis= 5)
plt.title('univariate - Company_encoded')
ax3 = plt.subplot2grid((2,3),(0,1))
plt.boxplot(data= df, x=df['Model_encoded'], vert=False)
plt.title('univariate - Model_encoded')
ax3 = plt.subplot2grid((2,3),(0,2))
plt.boxplot(data= df, x=df['Detail_encoded'], vert=False)
plt.title('univariate - Detail_encoded')
ax3 = plt.subplot2grid((2,3),(1,0))
sb.histplot(data=df, x= 'Company_encoded',kde=True)
plt.title('univariate - Company_encoded')
ax3 = plt.subplot2grid((2,3),(1,1))
sb.histplot(data=df, x= 'Model_encoded',kde=True)
plt.title('univariate - Model_encoded')
ax3 = plt.subplot2grid((2,3),(1,2))
sb.histplot(data=df, x= 'Detail_encoded',kde=True)
plt.title('univariate - Detail_encoded')

#  #graphical analysis - bivariate nominal
fig = plt.figure()
ax4 = plt.subplot2grid((2,3),(0,0))
sb.scatterplot(df,x='ProductionYear',y='Mileage')
plt.title('bivariate - year-mile')
ax4 = plt.subplot2grid((2,3),(0,1))
sb.scatterplot(df,x='ProductionYear',y='BodyCondition')
plt.title('bivariate - year-body')
ax4 = plt.subplot2grid((2,3),(0,2))
sb.scatterplot(df,x='ProductionYear',y='Price')
plt.title('bivariate - year-price')
ax4 = plt.subplot2grid((2,3),(1,0))
sb.scatterplot(df,x='Mileage',y='BodyCondition')
plt.title('bivariate - mile-body')
ax4 = plt.subplot2grid((2,3),(1,1))
sb.scatterplot(df,x='Mileage',y='Price')
plt.title('bivariate - mile-price')
ax4 = plt.subplot2grid((2,3),(1,2))
sb.scatterplot(df,x='ProductionYear',y='Price')
plt.title('bivariate - body-price')

#  #graphical analysis - miltuvariate nominal
plt.style.use('ggplot')

        ###standarding dataframe whit z-score standardization
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df.select_dtypes(include='int64').copy())
   
    #PCA test   ---> only numerical
pca_7 = PCA(n_components=8,random_state=16,whiten=True)
ipca_7 = IncrementalPCA(n_components=8,whiten=True)
pca_7.fit_transform(x_scaled)
ipca_7.fit_transform(x_scaled)
np.savetxt('PCA.txt', pca_7.explained_variance_ratio_*100,delimiter=',' , fmt='%10.5f')
np.savetxt('IPCA.txt', ipca_7.explained_variance_ratio_*100,delimiter=',' , fmt='%10.5f')
print('pca is :',pca_7.explained_variance_ratio_*100)
print('ipca is :',ipca_7.explained_variance_ratio_*100)
    #LDA test
# LDA = LinearDiscriminantAnalysis(store_covariance=True)
# LDA.fit_transform(x_scaled[:,0:4], None) ### It gives error and need debug
# np.savetxt('LDA.txt',LDA.explained_variance_ratio_, delimiter=',', fmt='%10.5f')

    #BADA test

    #MCA test  -----> only categorical
mca = prince.MCA(check_input = True, random_state = 16 , engine = 'auto')
mca.fit(df.select_dtypes(include='object').copy())
ax = mca.plot_coordinates(
    df.select_dtypes(include='object').copy(),
    ax = None
)
print('MCA eigenvalues : ', mca.eigenvalues_ )
print('MCA total inertia : ', mca.total_inertia_ )
print('MCA explained inertia : ', mca.explained_inertia_ )

        ### creating dummy variables
ohe = OneHotEncoder(drop='first')
obj_df = df.select_dtypes(include='object') 
obj_df = pd.get_dummies(obj_df,drop_first=True).astype('int64')
dfs = df.join(obj_df)
dfs= df.drop(['Company','Model','Detail','Date_encoded','Company_encoded','Model_encoded','Detail_encoded'],axis=1,inplace=False)

ca = prince.CA(check_input = True, random_state = 16 , engine = 'auto') #remove negative values
ca.fit(dfs[dfs.select_dtypes(include=[np.number]).ge(0).all(1)])
# axx = ca.plot_coordinates(
#     df,
#     ax = None
# )
print('CA eigenvalues : ', ca.eigenvalues_ )
print('CA total inertia : ', ca.total_inertia_ )
print('CA explained inertia : ', ca.explained_inertia_ )

    #MFA test
# # groups = {
# #         'Expert #{}'.format(no+1): [c for c in df.columns if c.startswith('E{}'.format(no+1))]
# #         for no in range(43)
# #     }
# mfa = prince.MFA(
#     groups = '',
#     n_components = 7,
#     copy = True,
#     check_input=True,
#     engine='auto',
#     random_state=16
# )
# mfa = mfa.fit(df.values).toarray() ## it gives error
# print('mfa.row_coordinates is :' , mfa.row_coordinates(df))
# print('mfa.partial_row_coordinates is :' , mfa.partial_row_coordinates(df))
# print('mfa.eigenvalues is :' ,mfa.eigenvalues_)
# print('mfa.total_inertia is :' , mfa.total_inertia_)
# print('mfa.row_contributions is :' , mfa.row_contributions(df))
# print('mfa.column_correlations' , mfa.column_correlations(df))
# axxx = mfa.plot_row_coordinates(
#     df,
#     ax=None,
#     show_points = True
# )
# ax.get_figure()
# plt.show()
# # #--------------! preprocess dataset !--------------#
df['Date'] = dates
del dates
    # lets set ProductionYear wich has two types of dates
df = df[(df.ProductionYear >= 1361) & (df.ProductionYear <= 2008)]
df = df[(df.ProductionYear <= 1391) | (df.ProductionYear >= 1997)]
df['is_miladi'] = np.where(df.ProductionYear >= 1997,1,0)
df.ProductionYear = np.where(df.ProductionYear >= 1997, df.ProductionYear, df.ProductionYear + 621)

    #delete mileage negative numbers
df = df[df.Mileage >=0]
    # delete 0 prices  =====> About 25 percent of dataset has zero price but as we can's predict them or find it's valid values we have to remove them 
df = df[df.columns.drop(list(df.filter(regex='_encoded')))]
df = df[df.Price > 0]
    #remove duplicate data
df = df.drop_duplicates(keep= 'first') #in all dataframe including dates has total duplicates
df = df.drop_duplicates(subset=df.columns.difference(['Date']),keep='first') # somtimes It might be same data in different dates wich might cause skewness and reduce data accuracy and validation
    #lets add some variables that might increase data accuracy
df['Date'] = pd.to_datetime(df.Date,dayfirst= True)
print(df.info())
        ## create a column that shows difference between the date of sell or buy and the cars production year and we assume the cars produced
        ## at the first day of a production year and each year is 365 days
df['UsedYears'] = (df['Date'].dt.year - df.ProductionYear)
df['UsedDays'] = (df['Date'].dt.day + (df.UsedYears - 1)*365)
df['MilePerYear'] = (df.Mileage / df.UsedYears)
df['MilePerDay'] = (df.Mileage / df.UsedDays)

dfs['UsedYears'] = df['UsedYears']
dfs['UsedDays'] = df['UsedDays']
dfs['MilePerYear'] = df['MilePerYear']
dfs['MilePerDay'] = df['MilePerDay']

ax5 = sb.heatmap(df.corr(), annot = True)
dfs = df.join(obj_df)
fig = plt.figure(figsize=(200,200))
ax6 = sb.heatmap(dfs.corr(),annot= True)
plt.show()

print(df)
df.to_csv('Somehow Preproccessed data.csv')
