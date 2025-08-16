import os
import re
import sys
import datetime
import joblib
import logging
import pandas as pd

from main import INPUT_DATA_FILE
from utils import normalize_headers
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

INPUT_DATA_FILE = "mbusa_all_claims.xlsx"
MIN_CLASS_COUNT = 10

# Set up logging
log_filename = f"logs/model_training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Add logger to modules
for module in ['train_test_split', 'TfidfVectorizer', 'LogisticRegression']:
    logging.getLogger(module).setLevel(logging.WARNING)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# List of target columns to train models for
target_columns = [
    'ai04g__issue_presentation',
    'ai04h__issue_type', 
    'ai04m__repair_costs_handling',
    'ai04s__does_repair_fall_under_warranty', 
    'ai04i__issue_verified',
    'ai04r__oem_engineering_services_involved', 
    'ai04j__repair_performed',
    'ai04k___of_repairs_performed_for_this_issue',
    'ai04n__not_repaired_reason',
    'ai04l__is_this_issue_the_primary_issue_driving_the_days_down',
    'ai04o__days_out_reason','ai04q__outside_influences'
]

def main():
    # ===== 1. LOAD DATA =====
    data = pd.read_excel(INPUT_DATA_FILE)

    # Normalize headers
    data = normalize_headers(data)

    logging.info(data.columns)
    
    for target_col in target_columns:
        # Copy the data to avoid modifying the original
        df = data.copy()

        # Convert both columns to string
        df['sf01c__issue_description'] = df['sf01c__issue_description'].astype(str)
        df['sf01d__repair_detail'] = df['sf01d__repair_detail'].astype(str)

        # Create a new column that combines issue description with repair detail
        df['issue_repair_combined_desc'] = df['sf01c__issue_description'] + " " + df['sf01d__repair_detail']

        # Drop rows with missing values
        df = df.dropna(subset=['issue_repair_combined_desc', target_col])

        # Remove rows with value "Unclear"
        if target_col in ['ai04i__issue_verified', 'ai04j__repair_performed', 
                        'ai04n__not_repaired_reason', 'ai04l__is_this_issue_the_primary_issue_driving_the_days_down']:
            df = df[df[target_col] != 'Unclear']

        # Remove rows with less than 10 examples of a class
        value_counts = df[target_col].value_counts()
        valid_values = value_counts[value_counts > MIN_CLASS_COUNT].index
        df = df[df[target_col].isin(valid_values)]

        logging.info(f"Total observations after filtering: {len(df)}")

        # Log value counts for each target column
        logging.info(f"\nValue counts for {target_col}:")
        logging.info(df[target_col].value_counts().to_string())

        # ===== 2. TEXT CLEANING =====
        def clean_text(text):
            text = text.lower()
            text = re.sub(r'\n+', ' ', text)  # Remove newlines
            text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        df['clean_description'] = df['issue_repair_combined_desc'].apply(clean_text)

            
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            df['clean_description'],
            df[target_col],
            test_size=0.2,
            stratify=df[target_col],
            random_state=42
        )

        # ===== 4. TF-IDF VECTORIZATION =====
        vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # ===== 5. TRAIN LOGISTIC REGRESSION =====
        model = LogisticRegression(max_iter=2000, class_weight='balanced')
        model.fit(X_train_tfidf, y_train)

        # ===== 6. EVALUATE =====
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_test, y_pred))

        # ===== 7. SAVE MODEL & VECTORIZER =====
        model_filename = f"models/{target_col}_classifier.pkl"
        vectorizer_filename = f"models/tfidf_vectorizer_{target_col}.pkl"
        
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, model_filename)
        joblib.dump(vectorizer, vectorizer_filename)
        
        logging.info(f"\nModel and vectorizer saved to disk for {target_col}")
        logging.info(f"Model: {model_filename}")
        logging.info(f"Vectorizer: {vectorizer_filename}")

if __name__ == "__main__":
    main()