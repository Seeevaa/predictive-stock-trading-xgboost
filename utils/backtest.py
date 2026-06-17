import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def execute_backtest(test_results):
    """
    Simulate algorithmic trading strategy performance against Buy & Hold benchmark.
    """
    df_bt = pd.DataFrame({
        'Aktual_Return_%': test_results['actual_returns'],
        'Sinyal_Trading': test_results['predictions']
    }, index=test_results['timestamps'])
    
    # Calculate performance metrics
    df_bt['Return_Strategi_%'] = df_bt['Sinyal_Trading'] * df_bt['Aktual_Return_%']
    df_bt['Cum_Buy_Hold'] = (1 + df_bt['Aktual_Return_%'] / 100).cumprod()
    df_bt['Cum_Strategy'] = (1 + df_bt['Return_Strategi_%'] / 100).cumprod()
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    ax1.plot(df_bt['Cum_Buy_Hold'], label='Benchmark: Buy & Hold', color='#1f77b4', linewidth=1.5)
    ax1.plot(df_bt['Cum_Strategy'], label='Quant Strategy (XGBoost)', color='#2ca02c', linewidth=2)
    
    # Plot exit signals
    exit_signals = df_bt[df_bt['Sinyal_Trading'] == 0]
    ax1.scatter(exit_signals.index, df_bt.loc[exit_signals.index, 'Cum_Strategy'], 
                color='red', marker='v', s=40, label='Exit / Cash Out Sinyal', zorder=5)
    
    ax1.set_title('Equity Curve Simulation (Fold 5 Performance)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (Normalized base = 1.0)')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    return df_bt, fig, ax2