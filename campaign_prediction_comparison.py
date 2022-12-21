import pandas as pd
import numpy as np
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn import metrics
from sklearn.impute import KNNImputer
import warnings

# Silence future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

# Split data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Response']), data['Response'],
                                                    random_state=7, test_size=0.25)

# Preprocess data
# Define numeric and categorical features
categoricals = ['Marital_Status', 'Complain']
ordinals = ['Education']
numerics = list(np.setdiff1d(X_train.columns, (categoricals + ordinals)))

# Create pipeline for numeric transformations
numeric_transformer = Pipeline(
    steps=[("imputer", KNNImputer()), ("scaler", StandardScaler())]
)

# Create one-hot encoder
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Create ordinal encoder
ordinal_transformer = OrdinalEncoder()

# Assemble together in a preprocessor
preproccessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerics),
        ("cat", categorical_transformer, categoricals),
        ("ord", ordinal_transformer, ordinals)
    ]
)


# Fit the preprocessor to the training data
X_train_prepped = preproccessor.fit_transform(X_train)
X_test_prepped = preproccessor.transform(X_test)


# Define evaluation function
def auc(m, train, test):
    """
    Automatically calculates AUC scores for both the training and test set.
    :param m: model or GridSearchCV object
    :param train: Input features of training set (X_train)
    :param test: Input features of test set (X_test)
    :return: Train AUC, Test AUC
    """
    return (np.round(metrics.roc_auc_score(y_train, m.predict_proba(train)[:, 1]), 4),
            np.round(metrics.roc_auc_score(y_test, m.predict_proba(test)[:, 1]), 4))


# Define hyperparameter grid
cb_param_dist = {
    'depth': [7, 10],
    'learning_rate': [0.03, 0.1],
    'l2_leaf_reg': [0, 1, 4],
    'iterations': [100, 250, 400]
}

# Instantiate CatBoost estimator
catboost_estimator = cb.CatBoostClassifier(random_seed=7)

# Create and fit a GridSearch object
print("Start Gridsearch CatBoost\n")
cb_model = GridSearchCV(catboost_estimator, param_grid=cb_param_dist, scoring='roc_auc', cv=5, n_jobs=-1, verbose=0)
cb_model.fit(X_train_prepped, y_train, verbose=False)


# Instantiate LightGBM estimator
lgbm_estimator = lgb.LGBMClassifier(random_state=7)

# Define hyperparameter grid
lgbm_param_dist = {"max_depth": [9, 12, 18],
                   "learning_rate": [0.03, 0.05, 0.1],
                   "num_leaves": [100, 200],
                   "n_estimators": [200, 500, 700]}

# Create GridSearch object for lightGBM
print("Start Gridsearch LightGBM\n")
lgbm_model = GridSearchCV(lgbm_estimator, n_jobs=-1, param_grid=lgbm_param_dist, cv=5, scoring="roc_auc", verbose=0)
lgbm_model.fit(X_train_prepped, y_train)


# Instantiate XGBoost model
xgb_estimator = xgb.XGBClassifier(verbosity=0, seed=7)

# Define hyperparmeter grid
xgb_param_dist = {"max_depth": [15, 25, 35],
                  "min_child_weight": [3, 6, 9],
                  "n_estimators": [200, 400],
                  "learning_rate": [0.05, 0.1]}


# Create GridSearch object for XGBoost
print("Start Gridsearch XGBoost\n")
xgb_model = GridSearchCV(xgb_estimator, n_jobs=-1, param_grid=xgb_param_dist, cv=5, scoring="roc_auc", verbose=0)
xgb_model.fit(X_train_prepped, y_train)


# Instantiate RandomForest Classifier
rf_estimator = RandomForestClassifier(random_state=7)

# Define hyperparamter grid
rf_param_dist = {"n_estimators": [100, 250],
                 "max_depth": [15, None],
                 "max_features": ["sqrt", "log2", 0.5]}

# Create GridSearch object for XGBoost
print("Start Gridsearch Random Forest\n")
rf_model = GridSearchCV(rf_estimator, n_jobs=-1, param_grid=rf_param_dist, cv=5, scoring="roc_auc", verbose=0)
rf_model.fit(X_train_prepped, y_train)

# Instantiate Voting Classifier
vc_model = VotingClassifier(estimators=[('cb', cb.CatBoostClassifier(**cb_model.best_params_, random_seed=7)),
                                        ('lgbm', lgb.LGBMClassifier(**lgbm_model.best_params_, random_state=7)),
                                        ('xgb', xgb.XGBClassifier(**xgb_model.best_params_, seed=7)),
                                        ('rf', RandomForestClassifier(**rf_model.best_params_, random_state=7))],
                            voting='soft', verbose=0)

# Fit voting classifier to the data
print("Fitting Voting Classifier")
vc_model.fit(X_train_prepped, y_train)

# Print best parameters
print("Best CatBoost hyperparameters are:{params}".format(params=cb_model.best_params_))
print("Best LightGBM hyperparameters are:{params}".format(params=lgbm_model.best_params_))
print("Best XGBoost hyperparameters are:{params}".format(params=xgb_model.best_params_))
print("Best Random Forest hyperparameters are:{params}\n".format(params=rf_model.best_params_))


# Calculate train and test AUC
print("CatBoost AUC score is: {auc}".format(auc=auc(cb_model, X_train_prepped, X_test_prepped)))
print("LightGBM AUC score is: {auc}".format(auc=auc(lgbm_model, X_train_prepped, X_test_prepped)))
print("XGBoost AUC score is: {auc}".format(auc=auc(xgb_model, X_train_prepped, X_test_prepped)))
print("Random Forest AUC score is: {auc}".format(auc=auc(rf_model, X_train_prepped, X_test_prepped)))
print("Voting Classifier AUC score is: {auc}".format(auc=auc(vc_model, X_train_prepped, X_test_prepped)))


# Calculate false positive rate and true positive rate
y_pred_cat = cb_model.predict_proba(X_test_prepped)[::, 1]
cfpr, ctpr, _ = metrics.roc_curve(y_test, y_pred_cat)

y_pred_lgbm = lgbm_model.predict_proba(X_test_prepped)[::, 1]
lfpr, ltpr, _ = metrics.roc_curve(y_test, y_pred_lgbm)

y_pred_xgb = xgb_model.predict_proba(X_test_prepped)[::, 1]
xfpr, xtpr, _ = metrics.roc_curve(y_test, y_pred_xgb)

y_pred_rf = rf_model.predict_proba(X_test_prepped)[::, 1]
rfpr, rtpr, _ = metrics.roc_curve(y_test, y_pred_rf)

y_pred_vc = vc_model.predict_proba(X_test_prepped)[::, 1]
vfpr, vtpr, _ = metrics.roc_curve(y_test, y_pred_vc)


# Plot ROC curve
fig, ax = plt.subplots()
plt.grid(visible=True)
plt.plot(cfpr, ctpr, alpha=0.3)
plt.plot(lfpr, ltpr, alpha=0.3)
plt.plot(xfpr, xtpr, alpha=0.3)
plt.plot(rfpr, rtpr, alpha=0.3)
plt.plot(vfpr, vtpr, alpha=0.3)
plt.legend(['CatBoost', 'LightGBM', 'XGBoost', 'Random Forest', 'Combined'])
ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--', color='grey')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title("ROC curve")
plt.savefig('roc_curve_comparison.png')
plt.show()
