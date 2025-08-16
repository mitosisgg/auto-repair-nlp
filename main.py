from utils import normalize_headers
from summarize import summarize_descriptions

import pandas as pd
import spacy
import joblib
import time
import pytextrank
import argparse
import sys
import logging
import os
from datetime import datetime

def setup_logging():
    """Set up logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'vehicle_repair_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process vehicle repair data')
    parser.add_argument('--input', type=str, default='data/sample_input.xlsx',
                      help='Path to the input Excel file')
    parser.add_argument('--output', type=str, default='data/sample_output.xlsx',
                      help='Path to save the output Excel file')
    parser.add_argument('--llm-host', type=str, default='localhost',
                      help='Hostname or IP of the LLM API server')
    parser.add_argument('--llm-port', type=int, default=12434,
                      help='Port of the LLM API server')
    parser.add_argument('--llm-model', type=str, default='gemma3',
                      help='Name of the LLM model to use')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Set the logging level')
    return parser.parse_args()

# ---- GLOBAL VARIABLES ----
args = parse_arguments()
logger = setup_logging()
logger.setLevel(args.log_level)

INPUT_DATA_FILE = args.input
OUTPUT_DATA_FILE = args.output
START_TIME = time.time()  # for execution time calculation

logger.info(f"Starting processing with input: {INPUT_DATA_FILE}")
logger.info(f"Output will be saved to: {OUTPUT_DATA_FILE}")
SENTENCE_LIMIT = 3 # limits keyphrase summarization length

# List of tagging columns
target_columns = [
    'ai04g__issue_presentation','ai04h__issue_type', 
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

def log_section_time(start_time, section_name):
    """Helper function to log section execution time"""
    elapsed = time.time() - start_time
    logger.info(f"{section_name} completed in {elapsed:.2f} seconds")
    return time.time()

def main():

    # --- Step 1: Load and Process Input Data ---
    try:
        df = pd.read_excel(INPUT_DATA_FILE)
        df = normalize_headers(df)
        df = df.astype(object)
        logger.info(f"Loaded {len(df)} rows from {INPUT_DATA_FILE}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    except FileNotFoundError as e:
        logger.error(f"Could not find input file: {INPUT_DATA_FILE}")
        logger.debug(f"Error details: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading input file: {str(e)}")
        logger.debug(f"Error details: {str(e)}")
        sys.exit(1)

    # -- Step 2: Summarize issue and repair descriptions --
    llm_start = time.time()
    df = summarize_records(
        df,
        llm_host=args.llm_host,
        llm_port=args.llm_port,
        llm_model=args.llm_model
    )
    llm_start = log_section_time(llm_start, "LLM summarization")
    logger.info("="*50)

    # --- Step 3: Predict enumerable columns ---
    logger.info("Starting prediction of enumerable fields...")
    predict_start = time.time()
    
    # Load the appropriate model for each target column
    for target_col in target_columns:
        logger.debug(f"Processing column: {target_col}")
        col_start = time.time()
        model = joblib.load(f"models/{target_col}_classifier.pkl")
        vectorizer = joblib.load(f"models/tfidf_vectorizer_{target_col}.pkl")

        # Convert both columns to string
        df['sf01c__issue_description'] = df['sf01c__issue_description'].astype(str)
        df['sf01d__repair_detail'] = df['sf01d__repair_detail'].astype(str)

        # Create a new column that combines issue description with repair detail
        df['issue_repair_combined_desc'] = df['sf01c__issue_description'] + " " + df['sf01d__repair_detail']

        # predict value for each row and fill in the missing values
        df[target_col] = model.predict(vectorizer.transform(df['issue_repair_combined_desc']))

        # Remove temporary column
        df = df.drop(columns=['issue_repair_combined_desc'])
        log_section_time(col_start, f"{target_col} prediction")

    predict_start = log_section_time(predict_start, "All predictions completed")
    logger.info("="*50)

    # --- Step 4: Save the Results ---
    save_start = time.time()
    try:
        df.to_excel(OUTPUT_DATA_FILE, index=False)
        logger.info(f"Results successfully saved to {OUTPUT_DATA_FILE}")
        logger.debug(f"Output file size: {os.path.getsize(OUTPUT_DATA_FILE) / (1024*1024):.2f} MB")
    except Exception as e:
        logger.error(f"Failed to save results to {OUTPUT_DATA_FILE}")
        logger.debug(f"Error details: {str(e)}")
        sys.exit(1)
    
    save_start = log_section_time(save_start, "Saving results")
    
    # Log total execution time
    total_time = time.time() - START_TIME
    logger.info("="*50)
    logger.info(f"Total execution time: {total_time/60:.2f} minutes")
    logger.info("Processing completed successfully")

def summarize_records(df, llm_host='localhost', llm_port=12434, llm_model='gemma3'):
    logger.info("Starting record summarization using local LLM")
    logger.info(f"LLM Server: {llm_host}:{llm_port}, Model: {llm_model}")
    total_records = len(df)
    start_time = time.time()
    success_count = 0
    
    for index, row in df.iterrows():
        record_start = time.time()
        row_id = row.get('record_id', index + 1)  # Fallback to index if 'record_id' doesn't exist
        issue_description = row['sf01c__issue_description']
        repair_description = row['sf01d__repair_detail']
        
        logger.info(f"Processing record {index + 1}/{total_records} (ID: {row_id})")
        
        try:
            result = summarize_descriptions(
                issue_description=issue_description,
                repair_description=repair_description,
                llm_host=llm_host,
                llm_port=llm_port,
                model_name=llm_model
            )
            for k, v in result.items():
                df.loc[index, k] = v
            
            success_count += 1
            record_time = time.time() - record_start
            
            # Log detailed timing info at debug level
            logger.debug(f"Record {row_id} processed in {record_time:.2f} seconds")
            
            # Log progress every 10 records or every minute, whichever comes first
            if (index + 1) % 10 == 0 or record_time > 60:
                avg_time = (time.time() - start_time) / (index + 1)
                remaining = (total_records - index - 1) * avg_time
                logger.info(
                    f"Progress: {index + 1}/{total_records} records "
                    f"({(index + 1)/total_records:.1%}) - "
                    f"Avg: {avg_time:.2f}s/record - "
                    f"ETA: {remaining/60:.1f} min remaining"
                )
                
        except Exception as e:
            logger.error(f"Error processing record {row_id}: {str(e)}")
            logger.debug(f"Record {row_id} failed after {time.time() - record_start:.2f} seconds")
    
    total_time = time.time() - start_time
    success_rate = (success_count / total_records) * 100
    
    logger.info("="*50)
    logger.info(f"Summarization completed. Processed {total_records} records in {total_time/60:.1f} minutes")
    logger.info(f"Successfully processed: {success_count}/{total_records} records ({success_rate:.1f}%)")
    logger.info(f"Average processing time: {total_time/total_records:.2f} seconds/record")
    
    if success_count < total_records:
        logger.warning(f"{total_records - success_count} records failed processing")
    
    return df

def main_wrapper():
    try:
        logger.info("="*50)
        logger.info("Starting vehicle repair summarization")
        logger.info(f"Command line arguments: {' '.join(sys.argv)}")
        
        main()
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        print("\nVehicle Repair Summarization Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  python main.py [options]\n")
        print("Options:")
        print("  --input <file>      Path to input Excel file")
        print("  --output <file>     Path for output Excel file")
        print("  --log-level <level> Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
        print("  --help, -h          Show this help message\n")
        print("Example:")
        print("  python main.py --input data/input.xlsx --output results/output.xlsx --log-level DEBUG\n")
        sys.exit(0)
    
    main_wrapper()