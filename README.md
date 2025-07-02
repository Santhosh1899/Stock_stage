Stock Stage prediction

This project focuses on predicting market stages and generating trading signals using a combination of data preprocessing, custom reinforcement learning environments,
and Deep Q-Networks (DQN). It begins with technical and engineered stock features, which are standardized and reduced using PCA for dimensionality reduction. 
A custom OpenAI Gym environment simulates trading actions based on rewards derived from price movements. A DQN is then trained across multiple stocks to learn optimal 
buy/sell/hold strategies. The model outputs discrete actions which are mapped to real-world trading signals (BUY, SELL, HOLD, etc.). Profits and losses are tracked for
every entry and exit, enabling a detailed performance evaluation. Finally, the pipeline generates annotated plots for each stock and stores a CSV summarizing trade 
performance metrics like average profit, win rate, and total return.

📌 Key Highlights (Bullet Points)

🔄 Data Processing: Applies StandardScaler and PCA on engineered stock market features.

🧠 Custom Gym Environment: Simulates trading with 5 discrete actions, including long and short trades.

🤖 Reinforcement Learning: Trains a Deep Q-Network (DQN) model to predict optimal trading actions.

💡 Signal Generation: Converts predicted stages into BUY, EXIT_BUY, SHORT_SELL, EXIT_SELL, and HOLD signals.

📈 Profit Calculation: Tracks percentage profit or loss for every completed trade.

📊 Performance Metrics:

Total profit per stock

Average profit per BUY/SHORT trade

Win rate for BUY/SHORT signals

Number of trades executed

📉 Visualization: Annotated plots with signals and profit markers overlayed on price charts.

💾 Model Saving: Scaler, PCA, and model state are saved using joblib for reuse.

📁 Evaluation Output: CSV file (model_performance_by_stock.csv) summarizing all results per stock.
