import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from utils.data_pipeline import fetch_market_data, transform_quant_features
from utils.model import run_walk_forward_classifier
from utils.backtest import execute_backtest

def main():
    print("="*60)
    print("      QUANTITATIVE TRADING ENGINE EXECUTION INTERFACE")
    print("="*60)
    
    # 1. Pipeline Ingestion & Transformation
    df_raw = fetch_market_data(ticker="BBCA.JK", currency_pair="IDR=X")
    df_clean = transform_quant_features(df_raw)
    
    # 2. Split Features and Targets
    features = ['Return_Asset_%', 'Perubahan_Volume_%', 'Return_USD_%']
    X = df_clean[features]
    y = df_clean['Target_Arah_Besok']
    
    # 3. Model Training via Forward Validation
    model, test_results = run_walk_forward_classifier(X, y, df_clean)
    
    # 4. Strategy Backtesting
    df_bt, fig, ax2 = execute_backtest(test_results)
    
    # 5. Inference / Real-time Prediction for Tomorrow
    latest_data = df_clean.iloc[[-1]][features]
    prediction = model.predict(latest_data)
    probabilities = model.predict_proba(latest_data)
    
    prob_down = probabilities[0][0] * 100
    prob_up = probabilities[0][1] * 100
    
    # Print Analytical Dashboard Terminal
    print("\n" + "="*50)
    print("      ALGORITHMIC PROJECTION FOR THE NEXT SESSION")
    print("="*50)
    print(f"Latest Asset Return : {latest_data['Return_Asset_%'].values[0]:.2f}%")
    print(f"Latest Vol Change   : {latest_data['Perubahan_Volume_%'].values[0]:.2f}%")
    print(f"Latest FX Return    : {latest_data['Return_USD_%'].values[0]:.2f}%")
    print("-"*50)
    
    if prediction[0] == 1:
        signal = "BULLISH / LONG POSITION"
        confidence = prob_up
        bar_colors = ['#ff9999', '#99ff99']
    else:
        signal = "BEARISH / CASH OUT (EXIT)"
        confidence = prob_down
        bar_colors = ['#ff9999', '#66b3ff']
        
    print(f"DIRECTIONAL SIGNAL : {signal}")
    print(f"CONFIDENCE METRIC  : {confidence:.2f}%")
    print("="*50)
    
    # Complete the visual plot dynamically
    bars = ax2.bar(['Prob Bearish', 'Prob Bullish'], [prob_down, prob_up], color=bar_colors, edgecolor='black')
    ax2.set_title('Model Confidence Level')
    ax2.set_ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('strategy_performance.png')
    print("[SUCCESS] Analytical chart exported successfully as 'strategy_performance.png'")

if __name__ == "__main__":
    main()