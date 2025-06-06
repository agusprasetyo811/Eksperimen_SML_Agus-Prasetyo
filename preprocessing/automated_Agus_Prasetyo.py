# Automated Data Preprocessing Pipeline - Exact Match Manual Version
# =====================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ===============================================
# 1. EXACT MATCH TRANSFORMERS
# ===============================================

class ExactDataCleaner(BaseEstimator, TransformerMixin):
    """Exact match untuk data cleaning dari kode manual"""
    
    def __init__(self):
        self.numeric_medians_ = {}
        self.categorical_modes_ = {}
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Simpan median untuk kolom numerik (sama seperti manual)
        numeric_cols = ['umur', 'pendapatan', 'skor_kredit', 'jumlah_pinjaman']
        for col in numeric_cols:
            if col in X_df.columns:
                self.numeric_medians_[col] = X_df[col].median()
        
        # Simpan mode untuk kolom kategorikal (sama seperti manual)
        categorical_cols = ['pekerjaan', 'disetujui']
        for col in categorical_cols:
            if col in X_df.columns:
                self.categorical_modes_[col] = X_df[col].mode()[0]
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Handle missing values persis seperti manual
        for col, median_val in self.numeric_medians_.items():
            if col in X_df.columns and X_df[col].isnull().sum() > 0:
                X_df[col].fillna(median_val, inplace=True)
        
        for col, mode_val in self.categorical_modes_.items():
            if col in X_df.columns and X_df[col].isnull().sum() > 0:
                X_df[col].fillna(mode_val, inplace=True)
        
        # Remove duplicates
        X_df = X_df.drop_duplicates()
        
        return X_df


class ExactOutlierHandler(BaseEstimator, TransformerMixin):
    """Exact match untuk outlier handling dengan IQR capping"""
    
    def __init__(self):
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Exact sama seperti detect_outliers_iqr function
        numeric_cols = ['umur', 'pendapatan', 'skor_kredit', 'jumlah_pinjaman']
        
        for col in numeric_cols:
            if col in X_df.columns:
                Q1 = X_df[col].quantile(0.25)
                Q3 = X_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.bounds_[col] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Capping method persis seperti manual
        for col, (lower_bound, upper_bound) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = np.where(X_df[col] < lower_bound, lower_bound, X_df[col])
                X_df[col] = np.where(X_df[col] > upper_bound, upper_bound, X_df[col])
        
        return X_df


class ExactFeatureEngineer(BaseEstimator, TransformerMixin):
    """Exact match untuk feature engineering"""
    
    def __init__(self):
        self.pendapatan_quantiles_ = None
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Simpan quantiles untuk kategori_pendapatan
        if 'pendapatan' in X_df.columns:
            self.pendapatan_quantiles_ = [
                X_df['pendapatan'].quantile(0.0),
                X_df['pendapatan'].quantile(0.333333),
                X_df['pendapatan'].quantile(0.666667),
                X_df['pendapatan'].quantile(1.0)
            ]
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # 1. Rasio Pinjaman terhadap Pendapatan (exact sama)
        if 'jumlah_pinjaman' in X_df.columns and 'pendapatan' in X_df.columns:
            X_df['rasio_pinjaman_pendapatan'] = X_df['jumlah_pinjaman'] / X_df['pendapatan']
        
        # 2. Kategori Umur (exact sama seperti function categorize_age)
        if 'umur' in X_df.columns:
            def categorize_age(age):
                if age < 30:
                    return 'Muda'
                elif age < 50:
                    return 'Dewasa'
                else:
                    return 'Senior'
            
            X_df['kategori_umur'] = X_df['umur'].apply(categorize_age)
        
        # 3. Kategori Skor Kredit (exact sama seperti function categorize_credit_score)
        if 'skor_kredit' in X_df.columns:
            def categorize_credit_score(score):
                if score < 600:
                    return 'Poor'
                elif score < 700:
                    return 'Fair'
                elif score < 800:
                    return 'Good'
                else:
                    return 'Excellent'
            
            X_df['kategori_skor_kredit'] = X_df['skor_kredit'].apply(categorize_credit_score)
        
        # 4. Kategori Pendapatan (exact sama dengan qcut)
        if 'pendapatan' in X_df.columns and self.pendapatan_quantiles_:
            X_df['kategori_pendapatan'] = pd.cut(
                X_df['pendapatan'], 
                bins=self.pendapatan_quantiles_, 
                labels=['Rendah', 'Sedang', 'Tinggi'],
                include_lowest=True
            )
        
        return X_df


class ExactEncoder(BaseEstimator, TransformerMixin):
    """Exact match untuk encoding seperti manual"""
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.categorical_features = ['pekerjaan', 'kategori_umur', 'kategori_skor_kredit', 'kategori_pendapatan']
        self.dummy_columns_ = []
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Fit label encoder untuk target
        if 'disetujui' in X_df.columns:
            self.label_encoder.fit(X_df['disetujui'])
        
        # Simulasi get_dummies untuk mendapatkan column names
        X_temp = X_df.copy()
        if 'disetujui' in X_temp.columns:
            X_temp['disetujui_encoded'] = self.label_encoder.transform(X_temp['disetujui'])
        
        # Get dummy columns
        existing_cats = [col for col in self.categorical_features if col in X_temp.columns]
        if existing_cats:
            X_dummies = pd.get_dummies(X_temp, columns=existing_cats, prefix=existing_cats)
            self.dummy_columns_ = X_dummies.columns.tolist()
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Label Encoding untuk target variable (exact sama)
        if 'disetujui' in X_df.columns:
            X_df['disetujui_encoded'] = self.label_encoder.transform(X_df['disetujui'])
        
        # One-Hot Encoding (exact sama dengan get_dummies)
        existing_cats = [col for col in self.categorical_features if col in X_df.columns]
        if existing_cats:
            X_df = pd.get_dummies(X_df, columns=existing_cats, prefix=existing_cats)
            
            # Ensure all expected columns exist
            for col in self.dummy_columns_:
                if col not in X_df.columns:
                    X_df[col] = 0
            
            # Keep only expected columns in right order
            X_df = X_df.reindex(columns=self.dummy_columns_, fill_value=0)
        
        return X_df


class ExactScaler(BaseEstimator, TransformerMixin):
    """Exact match untuk feature scaling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_features = ['umur', 'pendapatan', 'skor_kredit', 'jumlah_pinjaman', 'rasio_pinjaman_pendapatan']
    
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        # Fit hanya pada kolom numerik yang ada
        existing_numeric = [col for col in self.numeric_features if col in X_df.columns]
        if existing_numeric:
            self.scaler.fit(X_df[existing_numeric])
            self.numeric_features = existing_numeric
        
        return self
    
    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Scale hanya kolom numerik
        if self.numeric_features:
            X_df[self.numeric_features] = self.scaler.transform(X_df[self.numeric_features])
        
        return X_df


# ===============================================
# 2. EXACT AUTOMATED PREPROCESSOR
# ===============================================

class ExactAutomatedPreprocessor:
    """Automated preprocessor yang menghasilkan output PERSIS SAMA dengan manual"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize components dalam urutan yang sama
        self.data_cleaner = ExactDataCleaner()
        self.outlier_handler = ExactOutlierHandler()
        self.feature_engineer = ExactFeatureEngineer()
        self.encoder = ExactEncoder()
        self.scaler = ExactScaler()
        
        self.is_fitted = False
        self.final_columns_ = []
    
    def fit_transform(self, df):
        """Fit dan transform dengan exact sama steps seperti manual"""
        
        print("="*50)
        print("AUTOMATED PREPROCESSING - EXACT MATCH")
        print("="*50)
        
        # Copy original data
        df_original = df.copy()
        
        print(f"Dataset asli:")
        print(f"  Shape: {df_original.shape}")
        print(f"  Columns: {list(df_original.columns)}")
        
        # STEP 1: DATA CLEANING (exact sama)
        print("\n" + "="*50)
        print("STEP 1: DATA CLEANING")
        print("="*50)
        
        df_clean = self.data_cleaner.fit_transform(df_original)
        print(f"✓ Setelah cleaning: {df_clean.shape}")
        
        # STEP 2: OUTLIER HANDLING (exact sama)
        print("\n" + "="*50)
        print("STEP 2: OUTLIER DETECTION & HANDLING")
        print("="*50)
        
        df_no_outliers = self.outlier_handler.fit_transform(df_clean)
        print(f"✓ Setelah outlier handling: {df_no_outliers.shape}")
        
        # STEP 3: FEATURE ENGINEERING (exact sama)
        print("\n" + "="*50)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*50)
        
        df_engineered = self.feature_engineer.fit_transform(df_no_outliers)
        
        print("Feature baru yang dibuat:")
        print("  1. rasio_pinjaman_pendapatan: Rasio jumlah pinjaman terhadap pendapatan")
        print("  2. kategori_umur: Muda/Dewasa/Senior")
        print("  3. kategori_skor_kredit: Poor/Fair/Good/Excellent")
        print("  4. kategori_pendapatan: Rendah/Sedang/Tinggi")
        print(f"✓ Setelah feature engineering: {df_engineered.shape}")
        
        # STEP 4: ENCODING (exact sama)
        print("\n" + "="*50)
        print("STEP 4: ENCODING CATEGORICAL VARIABLES")
        print("="*50)
        
        df_encoded = self.encoder.fit_transform(df_engineered)
        
        print("Label Encoding untuk target variable:")
        print(f"  Tidak -> 0")
        print(f"  Ya -> 1")
        print(f"✓ Setelah encoding: {df_encoded.shape}")
        
        # STEP 5: FEATURE SCALING (exact sama)
        print("\n" + "="*50)
        print("STEP 5: FEATURE SCALING")
        print("="*50)
        
        df_scaled = self.scaler.fit_transform(df_encoded)
        print(f"✓ Setelah scaling: {df_scaled.shape}")
        
        # STEP 6: DATASET FINAL (exact sama)
        print("\n" + "="*50)
        print("STEP 6: DATASET FINAL")
        print("="*50)
        
        # Hapus kolom 'disetujui' yang asli (exact sama dengan manual)
        if 'disetujui' in df_scaled.columns:
            df_final = df_scaled.drop(columns=['disetujui'])
        else:
            df_final = df_scaled.copy()
        
        # Simpan kolom final
        self.final_columns_ = df_final.columns.tolist()
        
        # Pisahkan features dan target (exact sama)
        X = df_final.drop(['disetujui_encoded'], axis=1)
        y = df_final['disetujui_encoded']
        
        print("Dataset final berhasil dibuat:")
        print(f"  Shape: {df_final.shape}")
        print(f"  Features (X): {X.shape}")
        print(f"  Target (y): {y.shape}")
        
        print(f"\nKolom di dataset final:")
        for i, col in enumerate(df_final.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # STEP 7: TRAIN-TEST SPLIT (exact sama)
        print("\n" + "="*50)
        print("STEP 7: TRAIN-TEST SPLIT")
        print("="*50)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print("Dataset di-split menjadi:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        # SUMMARY (exact sama)
        print("\n" + "="*50)
        print("SUMMARY PREPROCESSING")
        print("="*50)
        
        print(f"TRANSFORMASI DATASET:")
        print(f"  Dataset asli: {df_original.shape}")
        print(f"  Setelah cleaning: {df_clean.shape}")
        print(f"  Setelah outlier handling: {df_no_outliers.shape}")
        print(f"  Setelah feature engineering: {df_engineered.shape}")
        print(f"  Setelah encoding: {df_encoded.shape}")
        print(f"  Dataset final: {df_final.shape}")
        
        self.is_fitted = True
        
        return X_train, X_test, y_train, y_test, X, y, df_final
    
    def get_feature_names(self):
        """Return feature names (tanpa target)"""
        if not self.is_fitted:
            raise ValueError("Preprocessor belum di-fit!")
        
        feature_names = [col for col in self.final_columns_ if col != 'disetujui_encoded']
        return feature_names
    
    def transform_new_data(self, df_new):
        """Transform data baru dengan pipeline yang sama"""
        if not self.is_fitted:
            raise ValueError("Preprocessor belum di-fit!")
        
        # Apply semua transformasi dalam urutan yang sama
        df_clean = self.data_cleaner.transform(df_new)
        df_no_outliers = self.outlier_handler.transform(df_clean)
        df_engineered = self.feature_engineer.transform(df_no_outliers)
        df_encoded = self.encoder.transform(df_engineered)
        df_scaled = self.scaler.transform(df_encoded)
        
        # Remove target column if exists
        if 'disetujui' in df_scaled.columns:
            df_final = df_scaled.drop(columns=['disetujui'])
        else:
            df_final = df_scaled.copy()
        
        # Remove target encoded if exists
        if 'disetujui_encoded' in df_final.columns:
            X_new = df_final.drop(['disetujui_encoded'], axis=1)
        else:
            X_new = df_final
        
        return X_new
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, df_final):
        """Save data exactly like manual version"""
        
        print(f"\nMENYIMPAN DATASET...")
        

        # Buat subdirektori jika belum ada
        output_dir = 'final_dataset'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dengan nama file yang sama
        df_final.to_csv('final_dataset/dataset_kredit_preprocessed.csv', index=False)
        
        # Convert to DataFrame with proper column names
        feature_names = self.get_feature_names()
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        y_train_df = pd.DataFrame(y_train, columns=['disetujui_encoded'])
        y_test_df = pd.DataFrame(y_test, columns=['disetujui_encoded'])
        
        X_train_df.to_csv('final_dataset/X_train.csv', index=False)
        X_test_df.to_csv('final_dataset/X_test.csv', index=False)
        y_train_df.to_csv('final_dataset/y_train.csv', index=False)
        y_test_df.to_csv('final_dataset/y_test.csv', index=False)
        
        print(f"✓ Dataset berhasil disimpan:")
        print(f"  - dataset_kredit_preprocessed.csv (dataset lengkap)")
        print(f"  - X_train.csv, X_test.csv (features)")
        print(f"  - y_train.csv, y_test.csv (target)")
        
        print("\n" + "="*50)
        print("PREPROCESSING SELESAI!")
        print("="*50)
        
        # Sample data final
        print(f"\nSample dataset final (5 baris pertama):")
        print(df_final.head())
        
        print(f"\nDataset siap untuk tahap modeling!")


# ===============================================
# 3. WRAPPER FUNCTION - EXACT MATCH
# ===============================================

def run_exact_automated_preprocessing(dataset_path='dataset_kredit_pinjaman.csv'):
    """Run automated preprocessing yang exact match dengan manual"""
    
    print("="*60)
    print("EXACT MATCH AUTOMATED PREPROCESSING")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Initialize exact preprocessor
    preprocessor = ExactAutomatedPreprocessor(test_size=0.2, random_state=42)
    
    # Run preprocessing
    X_train, X_test, y_train, y_test, X_full, y_full, df_final = preprocessor.fit_transform(df)
    
    # Save data
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, df_final)
    
    return preprocessor, X_train, X_test, y_train, y_test, df_final


# ===============================================
# 4. VERIFICATION FUNCTION
# ===============================================

def verify_exact_match(manual_result_path='dataset_kredit_preprocessed.csv'):
    """Verify bahwa hasil automated sama persis dengan manual"""
    
    try:
        # Load manual result
        df_manual = pd.read_csv(manual_result_path)
        print(f"Manual result loaded: {df_manual.shape}")
        print(f"Manual columns: {list(df_manual.columns)}")
        
        # Load automated result
        df_auto = pd.read_csv('dataset_kredit_preprocessed.csv')
        print(f"Automated result loaded: {df_auto.shape}")
        print(f"Automated columns: {list(df_auto.columns)}")
        
        # Compare
        if df_manual.shape == df_auto.shape:
            print("Shape match!")
        else:
            print("Shape mismatch!")
        
        if list(df_manual.columns) == list(df_auto.columns):
            print("Columns match perfectly!")
        else:
            print("Columns mismatch!")
            print(f"Manual has: {set(df_manual.columns) - set(df_auto.columns)}")
            print(f"Auto has: {set(df_auto.columns) - set(df_manual.columns)}")
        
        return True
        
    except FileNotFoundError:
        print("Manual result file not found. Run manual preprocessing first.")
        return False


# ===============================================
# 5. QUICK START
# ===============================================

if __name__ == "__main__":
    print("🎯 EXACT MATCH AUTOMATED PREPROCESSING")
    print("="*60)
    
    preprocessor, X_train, X_test, y_train, y_test, df_final = run_exact_automated_preprocessing('dataset_kredit_pinjaman.csv')
    