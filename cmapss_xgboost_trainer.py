import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Universal Column Headers for NASA C-MAPSS
COLUMNS = ['engine_id', 'cycle', 'os_1', 'os_2', 'os_3'] + [f'sensor_{i}' for i in range(1, 22)]

def load_train_data(fd_number, data_dir="."):
    file_path = os.path.join(data_dir, f"train_FD00{fd_number}.txt")
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMNS)
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.rename(columns={'cycle': 'max_cycle'}, inplace=True)
    df = df.merge(max_cycles, on='engine_id', how='left')
    
    # Calculate RUL and Cap at 125
    df['RUL'] = df['max_cycle'] - df['cycle']
    df['RUL'] = df['RUL'].clip(upper=125)
    
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def load_test_data(fd_number, data_dir="."):
    test_path = os.path.join(data_dir, f"test_FD00{fd_number}.txt")
    rul_path = os.path.join(data_dir, f"RUL_FD00{fd_number}.txt")
    df_test = pd.read_csv(test_path, sep=r'\s+', header=None, names=COLUMNS)
    true_rul = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['True_RUL'])
    true_rul['engine_id'] = true_rul.index + 1 
    max_cycles = df_test.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.rename(columns={'cycle': 'last_recorded_cycle'}, inplace=True)
    rul_data = max_cycles.merge(true_rul, on='engine_id')
    df_test = df_test.merge(rul_data, on='engine_id', how='left')
    
    # Back-calculate RUL and Cap at 125
    df_test['RUL'] = df_test['True_RUL'] + (df_test['last_recorded_cycle'] - df_test['cycle'])
    df_test['RUL'] = df_test['RUL'].clip(upper=125)
    
    df_test.drop(['last_recorded_cycle', 'True_RUL'], axis=1, inplace=True)
    return df_test

def drop_flatline_sensors(train_df, test_df):
    feature_cols = [c for c in train_df.columns if c not in ['engine_id', 'cycle', 'RUL']]
    cols_to_keep = ['engine_id', 'cycle', 'RUL']
    dropped_cols = []
    for col in feature_cols:
        if train_df[col].std() > 0.0001:
            cols_to_keep.append(col)
        else:
            dropped_cols.append(col)
    return train_df[cols_to_keep], test_df[cols_to_keep], [c for c in cols_to_keep if c not in ['engine_id', 'cycle', 'RUL']]

def normalize_by_operating_condition(train_df, test_df, features):
    train_norm = train_df.copy()
    test_norm = test_df.copy()
    os_cols = ['os_1', 'os_2', 'os_3']
    
    if train_df['os_1'].std() < 0.01:
        for col in features:
            mean = train_norm[col].mean()
            std = train_norm[col].std()
            if std > 0:
                train_norm[col] = (train_norm[col] - mean) / std
                test_norm[col] = (test_norm[col] - mean) / std
    else:
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        train_norm['condition'] = kmeans.fit_predict(train_norm[os_cols])
        test_norm['condition'] = kmeans.predict(test_norm[os_cols])
        
        for col in features:
            condition_stats = train_norm.groupby('condition')[col].agg(['mean', 'std']).reset_index()
            train_norm = train_norm.merge(condition_stats, on='condition', how='left')
            train_norm[col] = (train_norm[col] - train_norm['mean']) / (train_norm['std'] + 1e-8)
            train_norm.drop(['mean', 'std'], axis=1, inplace=True)
            
            test_norm = test_norm.merge(condition_stats, on='condition', how='left')
            test_norm[col] = (test_norm[col] - test_norm['mean']) / (test_norm['std'] + 1e-8)
            test_norm.drop(['mean', 'std'], axis=1, inplace=True)
            
        train_norm.drop('condition', axis=1, inplace=True)
        test_norm.drop('condition', axis=1, inplace=True)
        
    return train_norm, test_norm

def add_rolling_features(df, features, window=5):
    df_rolled = df.copy()
    new_feature_names = []
    for col in features:
        mean_col = f"{col}_rolling_mean"
        df_rolled[mean_col] = df_rolled.groupby('engine_id')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        std_col = f"{col}_rolling_std"
        df_rolled[std_col] = df_rolled.groupby('engine_id')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        new_feature_names.extend([mean_col, std_col])
    return df_rolled, new_feature_names

def train_and_evaluate(fd_number, data_dir="."):
    print(f"\n{'='*50}")
    print(f"PROCESSING DATASET FD00{fd_number} (XGBOOST)")
    print(f"{'='*50}")
    
    train_df = load_train_data(fd_number, data_dir)
    test_df = load_test_data(fd_number, data_dir)
    train_df, test_df, base_features = drop_flatline_sensors(train_df, test_df)
    train_df, test_df = normalize_by_operating_condition(train_df, test_df, base_features)
    train_df, rolling_features = add_rolling_features(train_df, base_features, window=5)
    test_df, _ = add_rolling_features(test_df, base_features, window=5)
    
    all_features = base_features + rolling_features
    
    print(f"   -> Training XGBoost Regressor on {len(all_features)} engineered features...")
    model = XGBRegressor(
        n_estimators=150, 
        max_depth=6, 
        learning_rate=0.05, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        n_jobs=-1
    )
    
    X_train = train_df[all_features]
    y_train = train_df['RUL']
    model.fit(X_train, y_train)
    
    test_last_cycle = test_df.groupby('engine_id').last().reset_index()
    X_test = test_last_cycle[all_features]
    y_true = test_last_cycle['RUL']
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"   -> [RESULT] RMSE for FD00{fd_number}: {rmse:.2f} cycles")
    
    return model, rmse

if __name__ == "__main__":
    DATA_DIRECTORY = "." 
    try:
        results = {}
        for i in range(1, 5):
            model, rmse = train_and_evaluate(i, DATA_DIRECTORY)
            results[f"FD00{i}"] = rmse
            
        print("\n" + "*"*40)
        print("Root Mean Squared Error")
        print("*"*40)
        for dataset, error in results.items():
            print(f"{dataset}: Off by ~{error:.2f} cycles on average")
            
    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing files. Ensure all train, test, and RUL files are in the directory.")
        print(e)