import yfinance as yf
import pandas as pd
import numpy as np

def fetch_market_data(ticker="BBCA.JK", currency_pair="IDR=X", period="2y"):
    """
    Ingest raw data from Yahoo Finance API and merge into a single dataframe.
    """
    try:
        print(f"[INFO] Fetching historical data for {ticker}...")
        df_asset = yf.download(ticker, period=period)[['Close', 'Volume']]
        df_asset.rename(columns={'Close': 'Harga_Asset', 'Volume': 'Volume_Asset'}, inplace=True)
        
        print(f"[INFO] Fetching forex data for {currency_pair}...")
        df_fx = yf.download(currency_pair, period=period)[['Close']]
        df_fx.rename(columns={'Close': 'Kurs_USD'}, inplace=True)
        
        # Inner join to synchronize trading days
        df_merged = pd.merge(df_asset, df_fx, left_index=True, right_index=True, how='inner')
        return df_merged
    except Exception as e:
        print(f"[-] Error during data ingestion: {e}")
        return pd.DataFrame()

def transform_quant_features(df):
    """
    Execute feature engineering and sanitize structural anomalies (Infinities/NaNs).
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Pipeline transformation aborted.")
        
    df_features = pd.DataFrame(index=df.index)
    
    # 1. Feature Engineering (Predictors)
    df_features['Return_Asset_%'] = df['Harga_Asset'].pct_change() * 100
    df_features['Perubahan_Volume_%'] = df['Volume_Asset'].pct_change() * 100
    df_features['Return_USD_%'] = df['Kurs_USD'].pct_change() * 100
    df_features['Diatas_MA5'] = np.where(df['Harga_Asset'] > df['Harga_Asset'].rolling(5).mean(), 1, 0)
    
    # 2. Target Generation
    df_features['Target_Return_Besok_%'] = df_features['Return_Asset_%'].shift(-1)
    df_features['Target_Arah_Besok'] = np.where(df_features['Target_Return_Besok_%'] > 0, 1, 0)
    
    # 3. Data Cleansing & Sanitization
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df_features.dropna()
    
    return df_clean