"""
Advanced Python Logging for ML Pipelines

Original Lab: Basic Python logging tutorial
My Modifications: Applied to complete ML pipeline with custom features
"""

import logging
import sys
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================
# CUSTOM LOGGING SETUP (Beyond Basic Lab)
# ============================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors - MODIFICATION from original lab"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname:8}"
                f"{self.RESET}"
            )
        return super().format(record)


def setup_logging(name="MLPipeline", log_file="ml_pipeline.log"):
    """
    Setup comprehensive logging - EXTENDS original lab's basic config
    Original: logging.basicConfig()
    Mine: Multi-handler with custom formatters
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Console Handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File Handler (ALL levels including DEBUG)
    file_handler = logging.FileHandler(f'logs/{log_file}')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Initialize logger
logger = setup_logging()

logger.info("="*70)
logger.info("ML PIPELINE WITH COMPREHENSIVE LOGGING")
logger.info("="*70)


# ============================================
# TIMING DECORATOR (CUSTOM ADDITION)
# ============================================

def log_execution_time(func):
    """Decorator to log execution time - NOT in original lab"""
    def wrapper(*args, **kwargs):
        logger.info(f"Starting: {func.__name__}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed: {func.__name__} in {execution_time:.2f}s")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {str(e)}")
            raise
    
    return wrapper


# ============================================
# DATA LOADING WITH LOGGING
# ============================================

@log_execution_time
def create_sample_dataset():
    """
    Create a sample dataset if no data file exists
    Demonstrates: logger.debug(), logger.info(), logger.warning()
    """
    logger.debug("Creating synthetic heart disease dataset...")
    
    np.random.seed(42)
    n_samples = 300
    
    # Create synthetic features
    age = np.random.randint(30, 80, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)
    trestbps = np.random.randint(90, 200, n_samples)
    chol = np.random.randint(120, 400, n_samples)
    fbs = np.random.randint(0, 2, n_samples)
    restecg = np.random.randint(0, 3, n_samples)
    thalach = np.random.randint(70, 200, n_samples)
    exang = np.random.randint(0, 2, n_samples)
    oldpeak = np.random.uniform(0, 6, n_samples)
    slope = np.random.randint(0, 3, n_samples)
    ca = np.random.randint(0, 4, n_samples)
    thal = np.random.randint(0, 4, n_samples)
    
    # Create target with some logic
    target = ((age > 55) & (chol > 250) | (cp > 2) | (thalach < 120)).astype(int)
    
    df = pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal, 'target': target
    })
    
    # Save to data folder
    df.to_csv('data/heart_disease.csv', index=False)
    logger.info(" Sample dataset created: data/heart_disease.csv")
    
    return df


@log_execution_time
def load_dataset(filepath='data/heart_disease.csv'):
    """Load dataset with comprehensive logging"""
    logger.debug(f"Attempting to load file: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded: {filepath}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Check data quality
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
        else:
            logger.info("No missing values")
        
        return df
        
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}")
        logger.info("Creating sample dataset...")
        return create_sample_dataset()
    
    except Exception as e:
        logger.exception(f"Error loading dataset")
        raise


@log_execution_time
def preprocess_data(df, target_col='target'):
    """Preprocess data with detailed logging"""
    logger.info("Starting preprocessing...")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    logger.info(f"Features: {X.shape[1]}, Samples: {len(df)}")
    logger.debug(f"Target distribution:\n{y.value_counts()}")
    
    # Check class balance
    class_ratio = y.value_counts().min() / y.value_counts().max()
    if class_ratio < 0.5:
        logger.warning(f"Class imbalance detected! Ratio: {class_ratio:.2f}")
    else:
        logger.info(f"Classes balanced (ratio: {class_ratio:.2f})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("Preprocessing completed")
    return X_train_scaled, X_test_scaled, y_train, y_test


# ============================================
# MODEL TRAINING WITH LOGGING
# ============================================

@log_execution_time
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model"""
    logger.info("-" * 50)
    logger.info(f"Training: {model_name}")
    logger.info("-" * 50)
    
    try:
        # Training
        model.fit(X_train, y_train)
        logger.info(f"Training completed")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"Results for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric:12}: {value:.4f}")
        
        # Critical check
        if metrics['accuracy'] < 0.6:
            logger.critical(f"CRITICAL: Low accuracy ({metrics['accuracy']:.2f})")
        
        return metrics
        
    except Exception as e:
        logger.exception(f"Training failed for {model_name}")
        return None


def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train and compare multiple models"""
    logger.info("="*70)
    logger.info("TRAINING MULTIPLE MODELS")
    logger.info("="*70)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        metrics = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        if metrics:
            results[name] = metrics
    
    return results


def compare_models(results):
    """Compare all models and find the best"""
    logger.info("="*70)
    logger.info("MODEL COMPARISON")
    logger.info("="*70)
    
    if not results:
        logger.error("No models to compare!")
        return None
    
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    
    logger.info(f"{'Model':<25} {'Accuracy':>10} {'F1-Score':>10}")
    logger.info("-" * 70)
    
    for model_name, metrics in results.items():
        marker = "  " if model_name == best_model else "  "
        logger.info(
            f"{marker} {model_name:<23} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['f1']:>10.4f}"
        )
    
    logger.info("-" * 70)
    logger.info(f"BEST MODEL: {best_model} ({results[best_model]['accuracy']:.4f})")
    
    return best_model


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main pipeline with comprehensive error handling"""
    try:
        # Load data
        logger.info("\n" + "="*70)
        logger.info("STEP 1: DATA LOADING")
        logger.info("="*70)
        df = load_dataset()
        
        # Preprocess
        logger.info("\n" + "="*70)
        logger.info("STEP 2: PREPROCESSING")
        logger.info("="*70)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Train models
        logger.info("\n" + "="*70)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*70)
        results = train_multiple_models(X_train, X_test, y_train, y_test)
        
        # Compare
        logger.info("\n" + "="*70)
        logger.info("STEP 4: COMPARISON")
        logger.info("="*70)
        best = compare_models(results)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Check detailed logs: logs/ml_pipeline.log")
        
    except Exception as e:
        logger.critical("FATAL ERROR: Pipeline failed!")
        logger.exception("Full traceback:")
    
    finally:
        logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    main()