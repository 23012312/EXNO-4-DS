# EXNO:4-DS
### Name: K ABHINESWAR REDDY
### Reg No: 212223040084
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
### Feature Scaling:
```py
# Standard Scaler
standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X)
print("Standard Scaled Data Sample:\n", pd.DataFrame(X_standard, columns=X.columns).head())

# Min-Max Scaler
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
print("\nMin-Max Scaled Data Sample:\n", pd.DataFrame(X_minmax, columns=X.columns).head())

# Max Absolute Scaler
maxabs_scaler = MaxAbsScaler()
X_maxabs = maxabs_scaler.fit_transform(X)
print("\nMax-Abs Scaled Data Sample:\n", pd.DataFrame(X_maxabs, columns=X.columns).head())

# Robust Scaler
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
print("\nRobust Scaled Data Sample:\n", pd.DataFrame(X_robust, columns=X.columns).head())
```
### Output:
![image](https://github.com/user-attachments/assets/fe41968d-e86d-4935-811f-19f0abf9a98e)
![image](https://github.com/user-attachments/assets/76c07254-dbeb-4864-bf76-3f069a5e01ed)
### Feature Selection:
```py
# Filter Method - Mutual Information
mi_scores = mutual_info_regression(X, y)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("\nMutual Information Scores:\n", mi_series)

# Filter Method - Chi-Square
X_chi2 = MinMaxScaler().fit_transform(X)  # chi2 requires non-negative values
chi_scores = chi2(X_chi2, y)[0]
chi_series = pd.Series(chi_scores, index=X.columns).sort_values(ascending=False)
print("\nChi-Square Scores:\n", chi_series)

# Embedded Method - Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print("\nLasso Coefficients:\n", pd.Series(lasso.coef_, index=X.columns))

# Embedded Method - Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("\nRidge Coefficients:\n", pd.Series(ridge.coef_, index=X.columns))

# Embedded Method - Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:\n", rf_series)

# Wrapper Method - Forward Selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
fs_forward = SFS(LinearRegression(), k_features=5, forward=True, floating=False, scoring='r2', cv=5)
fs_forward.fit(X_train, y_train)
print("\nForward Selection Features:", fs_forward.k_feature_names_)

# Wrapper Method - Backward Selection
fs_backward = SFS(LinearRegression(), k_features=5, forward=False, floating=False, scoring='r2', cv=5)
fs_backward.fit(X_train, y_train)
print("\nBackward Selection Features:", fs_backward.k_feature_names_)

# Wrapper Method - Recursive Feature Elimination
from sklearn.feature_selection import RFE
rfe = RFE(LinearRegression(), n_features_to_select=5)
rfe.fit(X, y)
print("\nRFE Selected Features:", X.columns[rfe.support_].tolist())
```
### Output:
![image](https://github.com/user-attachments/assets/4174d15d-d447-4ac6-bcca-eec1dfd221e9)
![image](https://github.com/user-attachments/assets/dd292a72-745a-4191-8a7e-d7d3e06ae928)
![image](https://github.com/user-attachments/assets/a829ded1-f453-4791-aa20-beff126e31e9)

# RESULT:
Thus the feature scaling and selection techniques are applied successfully.
