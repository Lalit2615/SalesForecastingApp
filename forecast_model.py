import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime

def simulate_data():
    np.random.seed(42)
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
    n = len(dates)
    
    # Base sales
    base_sales = 500
    # Trend: slow increase
    trend = np.linspace(0, 200, n)
    # Seasonality: Weekly (higher on weekends) + Yearly (higher in holiday season / summer)
    weekly_seasonality = np.where(dates.dayofweek >= 5, 150, 0) # Weekends
    yearly_seasonality = np.sin((dates.dayofyear / 365) * 2 * np.pi - np.pi/2) * 100 
    
    # Add some spikes for holidays
    holidays = pd.to_datetime(['2021-11-26', '2021-12-24', '2022-11-25', '2022-12-24', '2023-11-24', '2023-12-24'])
    holiday_effect = np.isin(dates, holidays) * 400
    
    noise = np.random.normal(0, 50, n)
    sales = base_sales + trend + weekly_seasonality + yearly_seasonality + holiday_effect + noise
    sales = np.maximum(sales, 0) # no negative sales
    
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    df['Is_Holiday'] = np.isin(df['Date'], holidays).astype(int)
    
    # Random promotions
    df['Promotion'] = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    
    # Apply promotion effect
    promotion_mask = df['Promotion'] == 1
    df.loc[promotion_mask, 'Sales'] += np.random.uniform(50, 150, size=promotion_mask.sum())
    
    print("Saving simulated data to sales_data.csv...")
    df.to_csv('sales_data.csv', index=False)
    return df

def feature_engineering(df):
    print("Engineering features...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    return df

def train_and_evaluate(df):
    features = ['Year', 'Month', 'DayOfWeek', 'DayOfYear', 'Is_Weekend', 'Is_Holiday', 'Promotion']
    X = df[features]
    y = df['Sales']
    
    # Split: train on 2021-2022, test on 2023
    train_mask = df['Year'] < 2023
    test_mask = df['Year'] == 2023
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Model Evaluation (Test Data - 2023):")
    print(f" - Mean Absolute Error (MAE): {mae:.2f} units")
    print(f" - Root Mean Squared Error (RMSE): {rmse:.2f} units")
    
    return model, features, test_mask, y_pred

def forecast_future(model, features, days=30):
    print(f"Forecasting future {days} days...")
    future_dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Year'] = future_df['Date'].dt.year
    future_df['Month'] = future_df['Date'].dt.month
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
    future_df['Is_Weekend'] = (future_df['DayOfWeek'] >= 5).astype(int)
    
    # Assume 2024-01-01 is a holiday, and assume a promotion on the second weekend
    future_df['Is_Holiday'] = (future_df['Date'] == '2024-01-01').astype(int)
    future_df['Promotion'] = 0
    # Add a mock promotion
    future_df.loc[(future_df['Date'] >= '2024-01-12') & (future_df['Date'] <= '2024-01-14'), 'Promotion'] = 1
    
    X_future = future_df[features]
    future_df['Predicted_Sales'] = model.predict(X_future)
    return future_df

def create_visualizations(df, test_mask, y_pred, future_df, model, features):
    print("Creating visualizations...")
    # sns.set_theme(style="whitegrid")
    plt.style.use('ggplot')
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Sales'], label='Actual Sales (Historical)', color='#1f77b4', alpha=0.6)
    plt.plot(df.loc[test_mask, 'Date'], y_pred, label='Model Predictions (2023)', color='#d62728', alpha=0.8)
    plt.title('Business Demand: Historical Sales vs. Model Predictions', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales Volume (Units)', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('historical_vs_predicted.png', dpi=300)
    plt.close()
    
    # Plot 2: Future Forecast Zoomed In
    plt.figure(figsize=(12, 5))
    # Last 2 months of 2023 for context
    context_df = df[df['Date'] >= '2023-11-01']
    plt.plot(context_df['Date'], context_df['Sales'], label='Recent Actual Sales', color='#1f77b4', marker='.')
    plt.plot(future_df['Date'], future_df['Predicted_Sales'], label='Next 30 Days Forecast', color='#2ca02c', linestyle='--', marker='o')
    
    # Highlight promotion periods in the future
    promo_dates = future_df[future_df['Promotion'] == 1]['Date']
    if not promo_dates.empty:
        plt.scatter(promo_dates, future_df.loc[future_df['Promotion'] == 1, 'Predicted_Sales'], color='gold', s=100, zorder=5, label='Planned Promos')
    
    plt.title('Short-Term Demand Forecast (Next 30 Days)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Expected Sales Volume', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('future_forecast.png', dpi=300)
    plt.close()
    
    # Plot 3: Feature Importance
    plt.figure(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.barh(range(len(indices)), importances[indices], color='#ff7f0e')
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=12)
    plt.title('What Drives Sales? (Factor Importance)', fontsize=16, fontweight='bold')
    plt.xlabel('Relative Impact on Sales', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()

def main():
    print("Starting Sales & Demand Forecasting Pipeline...")
    df = simulate_data()
    df = feature_engineering(df)
    model, features, test_mask, y_pred = train_and_evaluate(df)
    future_df = forecast_future(model, features, days=30)
    create_visualizations(df, test_mask, y_pred, future_df, model, features)
    
    future_df.to_csv('future_predictions.csv', index=False)
    print("Successfully generated predictions and visualizations!")

if __name__ == "__main__":
    main()
