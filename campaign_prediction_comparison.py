import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics

# Read in data
data = pd.read_csv('marketing_campaign.csv', sep=';')

# Convert sign-up dates to days since customer signed up
upload_date = '2014-08-01'
data['Customer_Since'] = (datetime.strptime(upload_date, "%Y-%m-%d") - pd.to_datetime(data['Dt_Customer'])).dt.days

# Drop unnecessary columns
data = data.drop(columns=['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4',
                          'AcceptedCmp5', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'ID'])

# Convert appropriate columns to category
data['Education'] = data.Education.astype('category')
data['Marital_Status'] = data.Marital_Status.astype('category')
data['Complain'] = data.Complain.astype('category')

# Checking for missing values
np.sum(data.isna())

# Fill missing values
med_income = data.groupby('Education')['Income'].median()

data['Income'] = data.apply(
    lambda row: med_income[row['Education']] if np.isnan(row['Income']) else row['Income'],
    axis=1
)

# Split data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Response']), data['Response'],
                                                    random_state=7, test_size=0.25)

# Define categorical feature indices
cat_features_index = [1, 2, 18]


# Define evaluation function
def auc(m, train, test):
    """
    Automatically calculates AUC scores for both the training and test set.
    :param m: model or GridSearchCV object
    :param train: Input features of training set (X_train)
    :param test: Input features of test set (X_test)
    :return: Train AUC, Test AUC
    """
    return (metrics.roc_auc_score(y_train, m.predict_proba(train)[:, 1]),
            metrics.roc_auc_score(y_test, m.predict_proba(test)[:, 1]))


# Define hyperparameter grid
hyper_params = {
    'depth': [7, 10],
    'learning_rate': [0.03, 0.1],
    'l2_leaf_reg': [0, 1, 4],
    'iterations': [100, 250, 400]
}

# Initiate CatBoost estimator
catboost_estimator = cb.CatBoostClassifier()

# Create and fit a GridSearch object
cb_model = GridSearchCV(catboost_estimator, hyper_params, scoring='roc_auc', cv=3, n_jobs=-1)
cb_model.fit(X_train, y_train, cat_features=cat_features_index, verbose=False)



# Instantiate LightGBM model
lgbm_estimator = lgb.LGBMClassifier(silent=False)

# Define hyperparameter grid
param_dist = {"max_depth": [20, 25, 50],
              "learning_rate": [0.05, 0.1],
              "num_leaves": [200, 300, 500],
              "n_estimators": [200, 400]}

# Create GridSearch object for lightGBM
lgbm_model = GridSearchCV(lgbm_estimator, n_jobs=-1, param_grid=param_dist, cv=3, scoring="roc_auc", verbose=1)
lgbm_model.fit(X_train, y_train)

# One-hot encode the categorical features
categoricals = ['Education', 'Marital_Status', 'Complain']
full_pipeline = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), categoricals)],
                                  remainder='passthrough')

encoder = full_pipeline.fit(X_train)
X_train_cat = encoder.transform(X_train)
X_test_cat = encoder.transform(X_test)


# Instantiate XGBoost model
xgb_estimator = xgb.XGBClassifier()

# Define hyperparmeter grid
xgb_param_dist = {"max_depth": [10, 25, 45],
                  "min_child_weight": [1, 3, 6],
                  "n_estimators": [200, 400],
                  "learning_rate": [0.05, 0.1]}


# Create GridSearch object for XGBoost
xgb_model = GridSearchCV(xgb_estimator, n_jobs=-1, param_grid=xgb_param_dist, cv=3, scoring="roc_auc", verbose=2)
xgb_model.fit(X_train_cat, y_train)


# Print best parameters
print("Best CatBoost hyperparameters are:{params}".format(params=cb_model.best_params_))
print("Best LightGBM hyperparameters are:{params}".format(params=lgbm_model.best_params_))
print("Best XGBoost hyperparameters are:{params}".format(params=xgb_model.best_params_))

# Calculate train and test AUC
print("CatBoost AUC score is: {auc}".format(auc=auc(cb_model, X_train, X_test)))
print("LightGBM AUC score is: {auc}".format(auc=auc(lgbm_model, X_train, X_test)))
print("XGBoost AUC score is: {auc}".format(auc=auc(xgb_model, X_train_cat, X_test_cat)))


# Calculate false positive rate and true positive rate
y_pred = cb_model.predict_proba(X_test)[::, 1]
cfpr, ctpr, _ = metrics.roc_curve(y_test, y_pred)

y_pred_lgbm = lgbm_model.predict_proba(X_test)[::, 1]
lfpr, ltpr, _ = metrics.roc_curve(y_test, y_pred_lgbm)

y_pred_xgb = xgb_model.predict_proba(X_test_cat)[::, 1]
xfpr, xtpr, _ = metrics.roc_curve(y_test, y_pred_xgb)

# Plot ROC curve
fig, ax = plt.subplots()
plt.grid(visible=True)
plt.plot(cfpr, ctpr)
plt.plot(lfpr, ltpr)
plt.plot(xfpr, xtpr)
plt.legend(['CatBoost', 'LightGBM', 'XGBoost'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='grey')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC curve")
plt.savefig('roc_curve_comparison.png')
plt.show()
