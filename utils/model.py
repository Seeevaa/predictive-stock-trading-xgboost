import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy as np

def run_walk_forward_classifier(X, y, df_source):
    """
    Execute walk-forward validation using TimeSeriesSplit to eliminate look-ahead bias.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    
    print("\n[INFO] Initializing Walk-Forward Validation...")
    fold = 1
    
    # Placeholders for the final fold metrics
    test_results = {}
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X_cls.iloc[test_index] if 'X_cls' in locals() else X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        prediksi = model.predict(X_test)
        akurasi = accuracy_score(y_test, prediksi)
        
        print(f" -> Fold {fold} | Train Size: {len(X_train)} | Test Size: {len(X_test)} | Accuracy: {akurasi*100:.2f}%")
        
        if fold == 5:
            test_results['predictions'] = prediksi
            test_results['actual_returns'] = df_source['Target_Return_Besok_%'].iloc[test_index].values
            test_results['timestamps'] = y_test.index
            
        fold += 1
        
    return model, test_results