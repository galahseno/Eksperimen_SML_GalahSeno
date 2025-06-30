import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

def load_data(path):
    return pd.read_csv(path, sep=';')

def handle_missing_values(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    for col in numerical_cols:
        if df[col].isnull().sum():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().sum():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def engineer_features(df):
    if 'age' in df.columns and df['age'].mean() > 365:
        df['age_years'] = (df['age'] / 365.25).round(0)
        df['age_group'] = pd.cut(df['age_years'], [0, 40, 50, 60, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])

    if 'height' in df.columns and 'weight' in df.columns:
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['bmi_category'] = pd.cut(df['bmi'], [0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        def bp_category(row):
            s, d = row['ap_hi'], row['ap_lo']
            if s < 120 and d < 80:
                return 'Normal'
            elif s < 130 and d < 80:
                return 'Elevated'
            elif (130 <= s < 140) or (80 <= d < 90):
                return 'Stage 1 Hypertension'
            elif s >= 140 or d >= 90:
                return 'Stage 2 Hypertension'
            return 'Other'
        df['bp_category'] = df.apply(bp_category, axis=1)

    return df

def handle_outliers(df):
    num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    for col in [c for c in num_cols if c in df.columns]:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
        df[col] = df[col].clip(lower, upper)

    if 'height' in df.columns:
        df = df[(df['height'] >= 100) & (df['height'] <= 250)]
    if 'weight' in df.columns:
        df = df[(df['weight'] >= 20) & (df['weight'] <= 300)]
    if 'ap_hi' in df.columns:
        df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 250)]
    if 'ap_lo' in df.columns:
        df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
    if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
        df = df[df['ap_hi'] > df['ap_lo']]

    return df

def encode_categoricals(df):
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != 'cardio']
    for col in binary_cols:
        df[f'{col}_encoded'] = df[col].map({df[col].unique()[0]: 0, df[col].unique()[1]: 1})

    multi_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() > 2]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)
    return df

def scale_features(df):
    df_norm = df.copy()
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.int64, np.float64] and col != 'cardio']
    numeric_cols = [col for col in numeric_cols if df[col].nunique() > 10]
    df_norm[numeric_cols] = MinMaxScaler().fit_transform(df_norm[numeric_cols])
    return df_norm

def drop_id(df):
    if 'id' in df.columns:
        df = df.drop(columns='id')
    return df

def export(df, path):
    df.to_csv(path, index=False)
    print(f"✅ Exported preprocessed dataset to {path}")

if __name__ == "__main__":
    print("🚀 Starting Preprocessing Pipeline")
    df = load_data("cardiovascular_disease_raw.csv")
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = engineer_features(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)
    df = scale_features(df)
    df = drop_id(df)
    export(df, "preprocessing/cardiovascular_disease_preprocessing.csv")
