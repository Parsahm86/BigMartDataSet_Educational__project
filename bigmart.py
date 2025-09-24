# add packages
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression,SGDRegressor, Ridge, ElasticNet
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor
# ---------------------------
print('hello world')
# -----------------------------------> download & load the DataSet

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
# Set the path to the file you'd like to load
file_path = "bigmart.csv"
path = kagglehub.dataset_download("yasserh/bigmartsalesdataset")
# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "yasserh/bigmartsalesdataset",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)
# -----------------------------------> fix the DataSet

# ---------> fill null values with mode, mean

df['Item_Weight'] = df.groupby('Item_Type')['Item_Weight'].transform(
    lambda x: x.fillna(x.mean())
)
# تابع کمکی برای پر کردن با mode
def fill_mode(series):
    return series.fillna(series.mode()[0])

df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(fill_mode)

# ---------> clean item_fat_Content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'LF':'Low Fat', 
    'low fat':'Low Fat', 
    'reg':'Regular'
})

# ---------> clean Item_Visibility replace 0 with mean
df['Item_Visibility'] = df['Item_Visibility'].replace(0, df['Item_Visibility'].mean())

# ---------> creat new fiture for Outlet_Years
df['Outlet_Years'] = 2025 - df['Outlet_Establishment_Year']

# ---------> drop fiture of item_Identifier, Outlet_Identifier
df.drop('Item_Identifier', axis=1, inplace=True)
df.drop('Outlet_Identifier', axis=1, inplace=True)

# --------->  one_hot_Encoder 
OHEC = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# ---------> item_type
item_type_encoded = OHEC.fit_transform(df[['Item_Type']])
item_type_df = pd.DataFrame(item_type_encoded,
                            columns=OHEC.get_feature_names_out(['Item_Type']))
df = pd.concat([df.drop('Item_Type', axis=1), item_type_df], axis=1)

# ---------> Outlet_type
outlet_type_encoded = OHEC.fit_transform(df[['Outlet_Type']])
outlet_type_df = pd.DataFrame(outlet_type_encoded,
                              columns=OHEC.get_feature_names_out(['Outlet_Type']))
df = pd.concat([df.drop('Outlet_Type',axis=1), outlet_type_df], axis=1)

# ---------> encoder_ordinal
encoder_ordinal = OrdinalEncoder() # convert to (0,1)

df['Item_Fat_Content'] = encoder_ordinal.fit_transform(df[['Item_Fat_Content']])

df['Outlet_Location_Type'] = encoder_ordinal.fit_transform(df[['Outlet_Location_Type']])

df['Outlet_Size'] = encoder_ordinal.fit_transform(df[['Outlet_Size']])

# -----------------------------------> model

# ---------> Creat X,y

X = df.drop(columns=['Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']

# # ---------> creat train-test X,y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

# # ---------> scale
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

# ---------> Comparison between model regression # -> the Best = XGB_reg

# models = {
#     'lin_reg':LinearRegression(),
#     'SGD_reg':SGDRegressor(),
#     'random_F_reg':RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
#     'XGB_reg':XGBRFRegressor(n_estimators=100, random_state=42),
#     'Ridge':Ridge(alpha=1),
#     'ElasticNet':ElasticNet(alpha=0.1, l1_ratio=0.5)
#     }

# results = []
# for name, model in models.items():
#     print(f"\n--------->{name}<---------\n")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     print(f"r2_score {name}: {r2_score(y_test, y_pred)}")
#     print(f"MSE {name}: {root_mean_squared_error(y_test, y_pred)}")
#     r2_sc = r2_score(y_test, y_pred)
#     RMSE = root_mean_squared_error(y_test, y_pred)
#     results.append({
#         'model':name,
#         'r2_score':r2_sc,
#         'RMSE':RMSE
#         })
    
# df_results = pd.DataFrame(results)

# df_results = df_results.sort_values('r2_score', ascending=False)

rcParams['font.family'] = 'DejaVu Sans'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# # # رسم نمودار
# plt.figure(figsize=(12,6))
# sns.barplot(data=df_results, x='r2_score', y='model', palette='Blues_d')
# plt.xticks(rotation=20)
# plt.yticks(fontsize=8)
# plt.title('R2 Score Comparison')
# plt.show()

# plt.figure(figsize=(10,6))
# sns.barplot(data=df_results, x='RMSE', y='model', palette='Reds_d')
# plt.xticks(rotation=20)
# plt.yticks(fontsize=8)
# plt.title('RMSE Comparison')
# plt.show()

# --------------------------> run the model
xgb_reg = XGBRFRegressor()

# ------------> random search for the Best hyperParm

param_dist = {
    "n_estimators": randint(100, 1000),
    "max_depth": randint(3, 15),
    "learning_rate": uniform(0.01, 0.2),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4)
}
scorer = make_scorer(r2_score)

random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_dist,
    n_iter=45,   # چند ترکیب تست کن
    scoring=scorer,
    cv=3,
    verbose=0,
    n_jobs=-1,
    random_state=42
)

# -------------> train the model
random_search.fit(X_train, y_train)

# cv_results = pd.DataFrame(random_search.cv_results_)
# best_results = cv_results.sort_values(by="mean_test_score", ascending=False)
# best_params = best_results[['mean_test_score','std_test_score','params']].head(10)

xgb_reg__optimal = random_search.best_estimator_


y_pred = xgb_reg__optimal.predict(X_test)

print(f"r2_score: {r2_score(y_test, y_pred)}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred)}")















