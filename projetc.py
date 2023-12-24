import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('PropertiesData.csv')

# Separate features and target variable
features = ['ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
target = 'CRIM'

# Drop rows with NaN values in the target variable
data = data.dropna(subset=[target])

# Impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Normalize the data
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# Perform PCA
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data[features])

# Train the model
model = LinearRegression()
model.fit(data_pca, data[target])

# Make predictions
predictions = model.predict(data_pca)

# Evaluate the model
mse_original = mean_squared_error(data[target], predictions)
print('MSE of the original model:', mse_original)

# Inverse transform the PCA predictions to compare with the original data
inverse_transformed_predictions = pca.inverse_transform(data_pca)
mse_pca = mean_squared_error(data[features], inverse_transformed_predictions)
print('MSE of the PCA model:', mse_pca)



















# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# # Load the dataset
# df = pd.read_csv('PropertiesData.csv')

# # Separate features (X) and target variable (y)
# X = df.drop(columns=['CRIM'])
# y = df['CRIM']

# # Check for missing values in the target variable
# if y.isnull().any():
#     # Impute missing values in the target variable using mean imputation
#     y_imputer = SimpleImputer(strategy='mean')
#     y = y_imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Impute missing values in features
# imputer = SimpleImputer(strategy='mean')
# X_train_imputed = imputer.fit_transform(X_train)
# X_test_imputed = imputer.transform(X_test)

# # Normalize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_imputed)
# X_test_scaled = scaler.transform(X_test_imputed)

# # Apply PCA
# pca = PCA(n_components=0.95)  # Retain 95% of variance
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

# # Train linear regression model without PCA
# model_without_pca = LinearRegression()
# model_without_pca.fit(X_train_scaled, y_train)
# y_pred_without_pca = model_without_pca.predict(X_test_scaled)

# # Train linear regression model with PCA
# model_with_pca = LinearRegression()
# model_with_pca.fit(X_train_pca, y_train)
# y_pred_with_pca = model_with_pca.predict(X_test_pca)

# # Evaluate mean squared error without PCA
# mse_without_pca = mean_squared_error(y_test, y_pred_without_pca)

# # Evaluate mean squared error with PCA
# mse_with_pca = mean_squared_error(y_test, y_pred_with_pca)

# # Print results
# print(f'MSE without PCA: {mse_without_pca}')
# print(f'MSE with PCA: {mse_with_pca}')

# # Summarize findings and conclusions
# print("\nSummary:")
# print("1. Regression without PCA:")
# print("   - Mean Squared Error (MSE):", mse_without_pca)

# print("\n2. Regression with PCA:")
# print("   - Mean Squared Error (MSE):", mse_with_pca)
# print("   - Number of Principal Components Retained:", pca.n_components_)

# print("\n3. Conclusion:")
# print("   - PCA helps in reducing dimensionality while retaining most of the variance.")
# print("   - The impact on MSE depends on the dataset and the amount of retained variance.")

# prompt: The features describe various parameters of properties Here's a brief description of each feature: • CRIM: Crime rate per capita. • ZN: Proportion of residential land zoned for large lots. • INDUS: Proportion of non-retail business acres. • CHAS: Proximity to Charles River (binary variable). • NOX: Nitrogen oxides concentration. • RM: Average number of rooms per dwelling. • AGE: Proportion of owner-occupied units built before 1940. • DIS: Weighted distances to five employment centers. • RAD: Accessibility to radial highways. • TAX: Property tax rate. • PTRATIO: Pupil-teacher ratio. • B: Proportion of African American residents. • LSTAT: Lower status population percentage. • MEDV: Median value of owner-occupied homes (target variable for regression). These features are used in the dataset to predict property prices. Apply regression algorithms to predict CRIM: Crime rate per capita. using the provided dataset (PropertiesData.csv). Implement imputation, normalization, and PCA. Assess whether PCA improves the mean squared error (MSE). Summarize your findings and conclusions.
