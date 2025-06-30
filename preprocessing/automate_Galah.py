"""
Automated Cardiovascular Disease Dataset Preprocessing Pipeline
================================================================

This module provides automated preprocessing functions for cardiovascular disease datasets.
It converts raw data into clean, ML-ready datasets with multiple preprocessing options.

Author: [Your Name]
Date: June 2025
Version: 1.0

Usage:
    from automate_cardiovascular_preprocessing import CardiovascularPreprocessor
    
    preprocessor = CardiovascularPreprocessor()
    processed_data = preprocessor.preprocess_data(df, target_column='cardio')
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CardiovascularPreprocessor:
    """
    A comprehensive preprocessing pipeline for cardiovascular disease datasets.
    
    This class provides automated data cleaning, feature engineering, encoding,
    scaling, and validation specifically designed for cardiovascular datasets.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print detailed processing information
        """
        self.verbose = verbose
        self.scalers = {}
        self.encoders = {}
        self.feature_info = {}
        self.processing_log = []
        
    def log_step(self, message):
        """Log processing steps with optional verbose output."""
        self.processing_log.append(message)
        if self.verbose:
            print(message)
    
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with missing values handled
        """
        self.log_step("\n🔍 STEP 1: Handling Missing Values")
        
        df_clean = df.copy()
        missing_summary = df_clean.isnull().sum()
        
        if missing_summary.sum() == 0:
            self.log_step("✅ No missing values found!")
            return df_clean
        
        # Handle numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                self.log_step(f"  ✓ Filled {col} missing values with median: {median_val}")
        
        # Handle categorical columns
        categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                self.log_step(f"  ✓ Filled {col} missing values with mode: {mode_val}")
        
        return df_clean
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows from the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with duplicates removed
        """
        self.log_step("\n🔄 STEP 2: Removing Duplicates")
        
        n_duplicates = df.duplicated().sum()
        
        if n_duplicates > 0:
            df_clean = df.drop_duplicates()
            self.log_step(f"✅ Removed {n_duplicates} duplicate rows")
            self.log_step(f"New dataset shape: {df_clean.shape}")
            return df_clean
        else:
            self.log_step("✅ No duplicates found!")
            return df.copy()
    
    def engineer_features(self, df):
        """
        Create new features from existing ones.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with engineered features
        """
        self.log_step("\n🔧 STEP 3: Feature Engineering")
        
        df_engineered = df.copy()
        
        # Convert age from days to years if needed
        if 'age' in df_engineered.columns and df_engineered['age'].mean() > 365:
            df_engineered['age_years'] = (df_engineered['age'] / 365.25).round(0)
            self.log_step("✅ Created 'age_years' feature (converted from days)")
            
            # Create age groups
            df_engineered['age_group'] = pd.cut(df_engineered['age_years'], 
                                               bins=[0, 40, 50, 60, 100], 
                                               labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
            self.log_step("✅ Created 'age_group' feature")
        
        # Calculate BMI if height and weight are available
        if 'height' in df_engineered.columns and 'weight' in df_engineered.columns:
            df_engineered['bmi'] = df_engineered['weight'] / ((df_engineered['height'] / 100) ** 2)
            self.log_step("✅ Created 'bmi' feature")
            
            # BMI categories
            df_engineered['bmi_category'] = pd.cut(df_engineered['bmi'], 
                                                  bins=[0, 18.5, 25, 30, 100], 
                                                  labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            self.log_step("✅ Created 'bmi_category' feature")
        
        # Blood pressure categories
        if 'ap_hi' in df_engineered.columns and 'ap_lo' in df_engineered.columns:
            def bp_category(row):
                systolic = row['ap_hi']
                diastolic = row['ap_lo']
                
                if systolic < 120 and diastolic < 80:
                    return 'Normal'
                elif systolic < 130 and diastolic < 80:
                    return 'Elevated'
                elif (systolic >= 130 and systolic < 140) or (diastolic >= 80 and diastolic < 90):
                    return 'Stage 1 Hypertension'
                elif systolic >= 140 or diastolic >= 90:
                    return 'Stage 2 Hypertension'
                else:
                    return 'Hypertensive Crisis'
            
            df_engineered['bp_category'] = df_engineered.apply(bp_category, axis=1)
            self.log_step("✅ Created 'bp_category' feature")
        
        return df_engineered
    
    def handle_outliers(self, df, target_column=None):
        """
        Detect and handle outliers in the dataset.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str, optional
            Name of target column to exclude from outlier handling
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with outliers handled
        """
        self.log_step("\n🚨 STEP 4: Handling Outliers")
        
        df_clean = df.copy()
        
        # Define numerical columns for outlier detection
        numerical_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
        available_numerical = [col for col in numerical_cols if col in df_clean.columns]
        
        # Remove target column from outlier handling
        if target_column and target_column in available_numerical:
            available_numerical.remove(target_column)
        
        # Remove physiologically impossible values
        impossible_removed = 0
        
        # Height: reasonable range 100-250 cm
        if 'height' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[(df_clean['height'] >= 100) & (df_clean['height'] <= 250)]
            removed = before - len(df_clean)
            impossible_removed += removed
            if removed > 0:
                self.log_step(f"  ✓ Removed {removed} rows with impossible height values")
        
        # Weight: reasonable range 20-300 kg
        if 'weight' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[(df_clean['weight'] >= 20) & (df_clean['weight'] <= 300)]
            removed = before - len(df_clean)
            impossible_removed += removed
            if removed > 0:
                self.log_step(f"  ✓ Removed {removed} rows with impossible weight values")
        
        # Blood pressure: reasonable ranges
        if 'ap_hi' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[(df_clean['ap_hi'] >= 70) & (df_clean['ap_hi'] <= 250)]
            removed = before - len(df_clean)
            impossible_removed += removed
            if removed > 0:
                self.log_step(f"  ✓ Removed {removed} rows with impossible systolic BP values")
        
        if 'ap_lo' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[(df_clean['ap_lo'] >= 40) & (df_clean['ap_lo'] <= 150)]
            removed = before - len(df_clean)
            impossible_removed += removed
            if removed > 0:
                self.log_step(f"  ✓ Removed {removed} rows with impossible diastolic BP values")
        
        # Logical check: systolic should be higher than diastolic
        if 'ap_hi' in df_clean.columns and 'ap_lo' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['ap_hi'] > df_clean['ap_lo']]
            removed = before - len(df_clean)
            impossible_removed += removed
            if removed > 0:
                self.log_step(f"  ✓ Removed {removed} rows where systolic <= diastolic BP")
        
        # Cap extreme outliers using IQR method
        for col in available_numerical:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            extreme_lower = Q1 - 3 * IQR
            extreme_upper = Q3 + 3 * IQR
            
            # Count extreme outliers before capping
            extreme_outliers = ((df_clean[col] < extreme_lower) | (df_clean[col] > extreme_upper)).sum()
            
            if extreme_outliers > 0:
                df_clean[col] = df_clean[col].clip(lower=extreme_lower, upper=extreme_upper)
                self.log_step(f"  ✓ Capped {extreme_outliers} extreme outliers in {col}")
        
        self.log_step(f"Total impossible values removed: {impossible_removed}")
        return df_clean
    
    def encode_categorical_variables(self, df, target_column=None):
        """
        Encode categorical variables for machine learning.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str, optional
            Name of target column to exclude from encoding
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with encoded categorical variables
        """
        self.log_step("\n🔤 STEP 5: Encoding Categorical Variables")
        
        df_encoded = df.copy()
        
        # Identify categorical columns
        categorical_columns = []
        binary_columns = []
        
        for col in df_encoded.columns:
            if col != target_column and (df_encoded[col].dtype == 'object' or df_encoded[col].nunique() <= 10):
                unique_vals = df_encoded[col].nunique()
                if unique_vals == 2:
                    binary_columns.append(col)
                elif unique_vals > 2:
                    categorical_columns.append(col)
        
        # Binary encoding (0/1)
        for col in binary_columns:
            unique_vals = df_encoded[col].unique()
            if len(unique_vals) == 2:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                df_encoded[f'{col}_encoded'] = df_encoded[col].map(mapping)
                self.encoders[col] = mapping
                self.log_step(f"  ✓ Binary encoded {col}: {mapping}")
        
        # One-hot encoding for multi-class categorical variables
        if categorical_columns:
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, prefix=categorical_columns)
            self.log_step(f"  ✓ One-hot encoded {len(categorical_columns)} columns")
        
        return df_encoded
    
    def scale_features(self, df, target_column=None, scaling_method='standard'):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str, optional
            Name of target column to exclude from scaling
        scaling_method : str, default='standard'
            Scaling method: 'standard', 'minmax', or 'both'
            
        Returns:
        --------
        pandas.DataFrame or dict
            Scaled dataframe or dictionary of scaled dataframes
        """
        self.log_step(f"\n📏 STEP 6: Feature Scaling ({scaling_method})")
        
        # Identify numerical columns for scaling
        numerical_for_scaling = []
        for col in df.columns:
            if col != target_column and df[col].dtype in ['int64', 'float64']:
                if df[col].nunique() > 10:  # Skip binary/categorical encoded features
                    numerical_for_scaling.append(col)
        
        if not numerical_for_scaling:
            self.log_step("No numerical columns found for scaling")
            return df.copy()
        
        self.log_step(f"Columns to be scaled: {numerical_for_scaling}")
        
        if scaling_method == 'standard':
            df_scaled = df.copy()
            scaler = StandardScaler()
            df_scaled[numerical_for_scaling] = scaler.fit_transform(df[numerical_for_scaling])
            self.scalers['standard'] = scaler
            self.log_step("✅ Applied StandardScaler (z-score normalization)")
            return df_scaled
            
        elif scaling_method == 'minmax':
            df_scaled = df.copy()
            scaler = MinMaxScaler()
            df_scaled[numerical_for_scaling] = scaler.fit_transform(df[numerical_for_scaling])
            self.scalers['minmax'] = scaler
            self.log_step("✅ Applied MinMaxScaler (0-1 normalization)")
            return df_scaled
            
        elif scaling_method == 'both':
            # Return both scaled versions
            df_standard = df.copy()
            df_minmax = df.copy()
            
            scaler_standard = StandardScaler()
            df_standard[numerical_for_scaling] = scaler_standard.fit_transform(df[numerical_for_scaling])
            self.scalers['standard'] = scaler_standard
            
            scaler_minmax = MinMaxScaler()
            df_minmax[numerical_for_scaling] = scaler_minmax.fit_transform(df[numerical_for_scaling])
            self.scalers['minmax'] = scaler_minmax
            
            self.log_step("✅ Applied both StandardScaler and MinMaxScaler")
            return {'standard': df_standard, 'minmax': df_minmax}
    
    def create_bins(self, df):
        """
        Create binned versions of continuous variables.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with binned features
        """
        self.log_step("\n📊 STEP 7: Creating Bins")
        
        df_binned = df.copy()
        
        # Age binning
        if 'age_years' in df_binned.columns:
            if 'age_group' not in df_binned.columns:
                df_binned['age_group'] = pd.cut(df_binned['age_years'], 
                                               bins=[0, 40, 50, 60, 100], 
                                               labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
                self.log_step("✅ Created age groups")
        
        # BMI binning
        if 'bmi' in df_binned.columns:
            if 'bmi_category' not in df_binned.columns:
                df_binned['bmi_category'] = pd.cut(df_binned['bmi'], 
                                                  bins=[0, 18.5, 25, 30, 100], 
                                                  labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
                self.log_step("✅ Created BMI categories")
        
        # Blood pressure binning
        if 'ap_hi' in df_binned.columns:
            df_binned['systolic_category'] = pd.cut(df_binned['ap_hi'], 
                                                   bins=[0, 120, 130, 140, 300], 
                                                   labels=['Normal', 'Elevated', 'Stage1_HTN', 'Stage2_HTN'])
            self.log_step("✅ Created systolic BP categories")
        
        if 'ap_lo' in df_binned.columns:
            df_binned['diastolic_category'] = pd.cut(df_binned['ap_lo'], 
                                                    bins=[0, 80, 85, 90, 200], 
                                                    labels=['Normal', 'Elevated', 'Stage1_HTN', 'Stage2_HTN'])
            self.log_step("✅ Created diastolic BP categories")
        
        return df_binned
    
    def validate_data(self, df, target_column=None):
        """
        Validate the processed data quality.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str, optional
            Name of target column
            
        Returns:
        --------
        dict
            Validation report
        """
        validation_report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        if target_column and target_column in df.columns:
            validation_report['target_distribution'] = df[target_column].value_counts(normalize=True).to_dict()
        
        return validation_report
    
    def preprocess_data(self, df, target_column='cardio', scaling_method='both', return_multiple=True):
        """
        Main preprocessing pipeline that orchestrates all preprocessing steps.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw input dataframe
        target_column : str, default='cardio'
            Name of target column
        scaling_method : str, default='both'
            Scaling method: 'standard', 'minmax', or 'both'
        return_multiple : bool, default=True
            Whether to return multiple preprocessed versions
            
        Returns:
        --------
        dict
            Dictionary containing different preprocessed versions of the data
        """
        self.log_step("=" * 60)
        self.log_step("AUTOMATED CARDIOVASCULAR PREPROCESSING PIPELINE")
        self.log_step("=" * 60)
        
        # Store original data
        df_original = df.copy()
        
        # Step 1: Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Step 2: Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # Step 3: Feature engineering
        df_engineered = self.engineer_features(df_clean)
        
        # Step 4: Handle outliers
        df_outliers_handled = self.handle_outliers(df_engineered, target_column)
        
        # Step 5: Encode categorical variables
        df_encoded = self.encode_categorical_variables(df_outliers_handled, target_column)
        
        # Step 6: Scale features
        if scaling_method == 'both':
            scaled_results = self.scale_features(df_encoded, target_column, 'both')
            df_standardized = scaled_results['standard']
            df_normalized = scaled_results['minmax']
        else:
            df_scaled = self.scale_features(df_encoded, target_column, scaling_method)
            if scaling_method == 'standard':
                df_standardized = df_scaled
                df_normalized = None
            else:
                df_normalized = df_scaled
                df_standardized = None
        
        # Step 7: Create bins
        df_binned = self.create_bins(df_encoded)
        
        # Validation
        self.log_step("\n✅ STEP 8: Data Validation")
        validation_report = self.validate_data(df_encoded, target_column)
        
        # Prepare results
        results = {
            'original': df_original,
            'clean': df_outliers_handled,
            'encoded': df_encoded,
            'binned': df_binned,
            'validation_report': validation_report,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'processing_log': self.processing_log
        }
        
        if df_standardized is not None:
            results['standardized'] = df_standardized
        if df_normalized is not None:
            results['normalized'] = df_normalized
        
        # Summary
        self.log_step("\n" + "=" * 60)
        self.log_step("PREPROCESSING COMPLETED SUCCESSFULLY! 🎉")
        self.log_step("=" * 60)
        self.log_step(f"Original shape: {df_original.shape}")
        self.log_step(f"Final shape: {df_encoded.shape}")
        self.log_step(f"Available datasets: {list(results.keys())}")
        
        # 💾 EXPORT DATA
        for name, df in results.items():
            if 'id' in df.columns:
                results[name] = df.drop(columns=['id'])

        if return_multiple:
            return results
        else:
            # Return the most commonly used version
            return df_normalized


def get_ml_ready_data(df, target_column='cardio'):
    """
    Get data preprocessed specifically for different ML model types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw input dataframe
    target_column : str, default='cardio'
        Name of target column
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe optimized for the specific model type
    """
    preprocessor = CardiovascularPreprocessor(verbose=False)
    results = preprocessor.preprocess_data(df, target_column, 'both', return_multiple=True)
    
    return  results['normalized']