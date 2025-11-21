import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# indivdiual beta datas 
import pandas as pd
import numpy as np

def load_individual_stock_betas(file_path=None):
    # temorary data/placeholder to test code
    data = {
        'Mkt-RF': [1.15, 1.05, 1.20, 1.45, 1.80, 1.60],
        'SMB': [-0.10, -0.05, 0.02, 0.20, 0.45, 0.30],
        'HML': [-0.40, -0.30, -0.25, -0.15, -0.55, -0.50],
        'RMW': [0.25, 0.35, 0.15, 0.05, -0.10, 0.00],
        'CMA': [0.10, 0.08, 0.05, 0.02, 0.15, 0.20]
    }
    
    index_labels = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA']
    df_betas = pd.DataFrame(data, index=index_labels)
    
    # for when u have files
    # df_betas = pd.read_csv(file_path, index_col='Stock')
    
    print("DataFrame successfully loaded.")
    return df_betas

# load the data
df_betas_from_regression = load_individual_stock_betas()
# plot individual stock 
import matplotlib.pyplot as plt

def plot_stock_factor_sensitivities(df_betas):
    
    # visualization, no MKT for clearer visualization of other factors 
    factors_to_plot = ['SMB', 'HML', 'RMW', 'CMA'] 
    df_plot = df_betas[factors_to_plot].copy()
    
    # calculate the total non-market factor exposure 
    df_plot['Total_Exposure'] = df_plot.sum(axis=1)
    df_plot = df_plot.sort_values('Total_Exposure', ascending=False)
    
    # prep data for plotting
    stocks = df_plot.index
    
    # setup
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # colors
    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
    
    # crete bar plots
    bottoms = np.zeros(len(stocks)) 
    
    for i, factor in enumerate(factors_to_plot):
        factor_values = df_plot[factor].values
        ax.bar(stocks, factor_values, bottom=bottoms, label=factor, color=colors[i])
        bottoms += factor_values # Update the bottom for the next factor
    
    # labels
    ax.set_title('Factor Sensitivities (SMB, HML, RMW, CMA) for Individual Stocks', fontsize=16)
    ax.set_ylabel('Beta Value', fontsize=12)
    ax.set_xlabel('Stock Ticker', fontsize=12)
    ax.legend(title='FF Factors')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    ax.axhline(0, color='grey', linewidth=0.8)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

plot_stock_factor_sensitivities(df_betas_from_regression)