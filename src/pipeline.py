from data_processing import load_data,impute_embarked,impute_age,preprocess_features


def run_pipeline():
    raw_data_path='../data/train.csv'
    df_raw = load_data(raw_data_path)
    print("Step 1: Data loaded successfully.")
    df=impute_embarked(df_raw)
    df=impute_age(df)
    df=preprocess_features(df)
    print("\n--- Processed Data Info ---")
    df.info()
    print("\n--- First 5 Rows of Processed Data ---")
    print(df.head())
    
    return df

if __name__ == "__main__":
    processed_data = run_pipeline()