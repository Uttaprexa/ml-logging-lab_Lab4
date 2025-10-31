# ML Logging Lab - Advanced Python Logging for ML Pipeline

## Executive Summary

This lab **extends** the basic Python logging tutorial (`Starter.ipynb`) by implementing comprehensive logging in a production-ready ML pipeline for heart disease classification.
**Result**: Successfully trained and compared 4 ML models with complete logging coverage across all pipeline stages.

---

## Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 85.00% | 85.28% | 85.00% | 85.11% | 0.04s |
| Random Forest | 96.67% | 96.82% | 96.67% | 96.61% | 0.16s |
| **Gradient Boosting** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **0.15s** |
| SVM | 85.00% | 84.67% | 85.00% | 84.58% | 0.02s |

**Best Model**: Gradient Boosting (100% accuracy on test set)

---

## What Changed from Original Lab?

### Original Lab (`Starter.ipynb`)

The original lab taught **basic Python logging concepts**:

1. **Import**: `import logging`
2. **Basic Config**: `logging.basicConfig(level=logging.DEBUG)`
3. **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
4. **Custom Loggers**: `logging.getLogger("my_module")`
5. **Exception Logging**: `logging.exception("error message")`
6. **File Handlers**: `logging.FileHandler('app.log')`
7. **Console Handlers**: `logging.StreamHandler()`

**Scope**: Simple code examples demonstrating each concept

---

### My Implementation (main.py)

I **applied** these concepts to a **real-world ML problem**:

#### **1. Custom Colored Formatter** 
```python
class ColoredFormatter(logging.Formatter):
    """ANSI color codes for visual log distinction"""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
```
**Why**: Makes console logs easier to scan and understand

#### **2. Multi-Handler Architecture** 
```python
# Console: INFO and above (for monitoring)
console_handler.setLevel(logging.INFO)

# File: DEBUG and above (for debugging)
file_handler.setLevel(logging.DEBUG)
```
**Why**: Different detail levels for different audiences

#### **3. Real ML Application** 
Instead of toy examples, built complete pipeline:
- Data loading with validation
- Preprocessing with quality checks
- Training 4 different models
- Performance evaluation
- Model comparison

**Why**: Demonstrates logging in production context

#### **4. Execution Time Tracking** 
```python
@log_execution_time
def train_and_evaluate(...):
    # Automatically logs start, end, duration
```
**Why**: Performance monitoring is critical in MLOps

#### **5. Comprehensive Error Handling** 
- Try-except blocks throughout
- Full traceback logging with `logger.exception()`
- Graceful failure handling

**Why**: Production systems need robust error handling

#### **6. Structured Logging Strategy** 
Every function logs:
- **DEBUG**: Parameters, intermediate values
- **INFO**: Major steps, successful operations
- **WARNING**: Data quality issues
- **ERROR**: Operation failures
- **CRITICAL**: Pipeline-breaking problems

---

##  Project Structure
```
ml-logging-lab/
├── main.py                    # Complete ML pipeline 
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   └── heart_disease.csv     # Synthetic dataset (300 samples, 14 features)
├── logs/
│  └── ml_pipeline.log       # Detailed execution logs

```

---

## How to Implement

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Uttaprexa/ml-logging-lab_Lab4.git
cd ml-logging-lab

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline
```bash
# Execute full pipeline
python main.py

# View console output (INFO level)
# - See major steps and results

# View detailed logs (DEBUG level)
# Windows:
type logs\ml_pipeline.log
# Mac/Linux:
cat logs/ml_pipeline.log
```

---

## Learning Outcomes

### From Original Lab (Starter.ipynb)
- Understood Python logging module architecture
- Learned 5 log levels and appropriate usage
- Practiced custom logger creation
- Implemented file and console handlers
- Used exception logging with tracebacks

### My Extensions
- Applied logging to complete ML workflow
- Created production-ready logging setup
- Implemented execution time tracking
- Built comprehensive error handling
- Demonstrated MLOps logging best practices
- Compared multiple ML models with logged metrics

---

## Logging Strategy

| Log Level | When Used | Example from Code |
|-----------|-----------|-------------------|
| **DEBUG** | Detailed info for debugging | `Target distribution: 0: 211, 1: 89` |
| **INFO** | Major pipeline steps, results | `Training: Random Forest` |
| **WARNING** | Potential issues | `Class imbalance detected!` |
| **ERROR** | Operation failures | `Training failed for model_name` |
| **CRITICAL** | Pipeline-breaking problems | `Low accuracy (0.45)` |

---

## Code Comparison

### Original Lab Approach
```python
# Simple example
logging.basicConfig(level=logging.DEBUG)
logging.debug("This is a debug message")
logging.info("This is an info message")

try:
    result = 10 / 0
except ZeroDivisionError:
    logging.exception("Division by zero error")
```

### My Implementation
```python
# Production-ready setup
logger = setup_logging("MLPipeline", "ml_pipeline.log")

@log_execution_time
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    logger.info(f"Training: {model_name}")
    
    try:
        model.fit(X_train, y_train)
        metrics = calculate_metrics(...)
        
        logger.info(f"Results for {model_name}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        if metrics['accuracy'] < 0.6:
            logger.critical(f"CRITICAL: Low accuracy!")
            
    except Exception as e:
        logger.exception(f"Training failed for {model_name}")
```

---

## Dataset Details

### Synthetic Heart Disease Dataset
- **Source**: Programmatically generated
- **Samples**: 300 (240 train, 60 test)
- **Features**: 13 clinical parameters
  - Age, Sex, Chest Pain Type
  - Blood Pressure, Cholesterol
  - Heart Rate, etc.
- **Target**: Binary classification (disease/no disease)
- **Class Distribution**: 211 negative, 89 positive

---

## Technical Implementation

### Key Technologies
- **Python 3.11**: Core language
- **scikit-learn 1.5+**: ML algorithms
- **pandas 2.0+**: Data manipulation
- **numpy 1.24+**: Numerical operations

### Models Trained
1. **Logistic Regression**: Baseline linear classifier
2. **Random Forest**: Ensemble of 100 decision trees
3. **Gradient Boosting**: Sequential ensemble (100 estimators)
4. **SVM**: Support Vector Machine with RBF kernel

### Preprocessing
- **Scaling**: StandardScaler (zero mean, unit variance)
- **Splitting**: 80/20 train-test, stratified
- **Validation**: Cross-validation in training

---

## Sample Outputs

### Console Output (Colored, INFO level)
```
18:57:28 | INFO     | ML PIPELINE WITH COMPREHENSIVE LOGGING
18:57:28 | INFO     | STEP 1: DATA LOADING
18:57:28 | INFO     | Dataset loaded: data/heart_disease.csv
18:57:28 | INFO     | Shape: (300, 14)
18:57:28 | INFO     | STEP 2: PREPROCESSING
18:57:28 | INFO     | Train: 240 samples, Test: 60 samples
18:57:28 | INFO     | STEP 3: MODEL TRAINING
18:57:28 | INFO     | Training: Logistic Regression
18:57:28 | INFO     | Results for Logistic Regression:
18:57:28 | INFO     |   accuracy    : 0.8500
18:57:28 | INFO     | Training: Random Forest
18:57:28 | INFO     |   accuracy    : 0.9667
```

### Log File (Detailed, DEBUG level)
```
2025-10-30 18:57:28 | MLPipeline | DEBUG | preprocess_data | Target distribution:
target
1    211
0     89
Name: count, dtype: int64
2025-10-30 18:57:28 | MLPipeline | INFO | preprocess_data | Train: 240 samples, Test: 60 samples
```

---

## Comparison Table

| Aspect | Original Lab | My Implementation |
|--------|--------------|-------------------|
| **Scope** | Tutorial examples | Production pipeline |
| **Application** | Print statements | Heart disease classification |
| **Formatters** | Basic | Custom colored |
| **Error Handling** | 1 try-except | Throughout pipeline |
| **Timing** | None | Every function |
| **Dataset** | None | 300 samples |
| **Models** | None | 4 algorithms |
| **Metrics** | None | Accuracy, Precision, Recall, F1 |

```

---
