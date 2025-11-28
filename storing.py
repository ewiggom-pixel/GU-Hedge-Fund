import pandas as pd
import numpy as np

def load_individual_stock_sensitivities(file_path):
    try:
        df_sensitivities = pd.read_csv(file_path, index_col='Stock')
        
        if 'Alpha' not in df_sensitivities.columns:
             raise ValueError("The input file must contain a column labeled 'Alpha'.")
             
        cols = ['Alpha'] + [col for col in df_sensitivities.columns if col != 'Alpha']
        df_sensitivities = df_sensitivities[cols]
        
        print("successfully loaded")
        return df_sensitivities
        
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
        # Return an empty DataFrame in case of error
        return pd.DataFrame()
    except ValueError as e:
        print(f"Data Error: {e}")
        return pd.DataFrame()


# NEED TO Replace 'path/to/your/data.csv' with the actual file path.
file_path_example = 'path/to/your/data.csv' 
# df_sensitivities = load_individual_stock_sensitivities(file_path_example)

# blank data frame here for now 
df_sensitivities = pd.DataFrame({
    'Alpha': [], 'Mkt-RF': [], 'SMB': [], 'HML': []
}, index=[])


if not df_sensitivities.empty:
    # convert the DataFrame into a NumPy array
    sensitivity_array = df_sensitivities.to_numpy()

    # print(sensitivity_array) 
    print(f"Shape: (N Stocks, {len(df_sensitivities.columns)} Factors)")
    print(f"Column Order: {df_sensitivities.columns.tolist()}")

    # accessing Alpha and Beta from the Array
    # all_alphas = sensitivity_array[:, 0]
    # all_market_betas = sensitivity_array[:, 1]
    
    # print(f"\nAll Alpha values (NumPy): {all_alphas}")
    
else:
    print("\n⚠️ No data loaded. Please provide a valid CSV file path.")