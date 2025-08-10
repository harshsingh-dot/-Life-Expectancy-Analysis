import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import os
import warnings
warnings.filterwarnings('ignore')

# === Set absolute path to your dataset ===
data_path = 'E:/life_expectancy_analysis_project/data/Life Expectancy Data.csv'

# Load data
df = pd.read_csv(data_path)

# Fix column names: remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Print actual column names (debug tip)
print("Available Columns:")
print(df.columns.tolist())

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = imputer.fit_transform(df[[col]])

# Handle outliers (correct column names used)
outlier_cols = [
    'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
    'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio',
    'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
    'thinness  1-19 years',  # NOTE: two spaces
    'thinness 5-9 years',
    'Income composition of resources', 'Schooling'
]

for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[col] = np.where((df[col] > upper_bound) | (df[col] < lower_bound), np.mean(df[col]), df[col])

# Encode categorical columns
le = LabelEncoder()
df['Country'] = le.fit_transform(df['Country'])
df['Status'] = le.fit_transform(df['Status'])

# Split features/target
x = df.drop(columns='Life expectancy')
y = df['Life expectancy']

# Scale features
scaler = StandardScaler()
x[x.columns] = scaler.fit_transform(x[x.columns])

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# Define models
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Extra Trees Regressor': ExtraTreesRegressor(random_state=42),
    'GradientBoost Regressor': GradientBoostingRegressor(random_state=42),
    'XGB Regressor': XGBRegressor()
}

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    r2 = r2_score(y_test, preds)
    results.append({'Model': name, 'RMSE': rmse, 'R2 Score': r2})

results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False)
print("\nModel Performance Comparison:")
print(results_df)

# Cross-validation on best model
best_model = XGBRegressor()
kf = KFold(n_splits=20, shuffle=True, random_state=42)
scores = cross_val_score(best_model, x, y, cv=kf, scoring='r2')
print("\nCross-Validation R2 Mean:", scores.mean())
print("Cross-Validation R2 Std Dev:", scores.std())

# Save results plot
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(8, 6))
sns.pointplot(data=results_df, x='Model', y='R2 Score')
plt.title('Model Comparison: R2 Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/model_comparison_r2.png')
plt.show()

print("Plot saved as 'plots/model_comparison_r2.png'")
