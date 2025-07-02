**Stock Stage prediction**

This project implements a comprehensive pipeline for unsupervised market stage prediction and trading signal generation using Deep Reinforcement Learning. It begins with an extensive set of technical indicators derived from stock market data. These include trend, momentum, volatility, volume, and price-based features such as RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic Oscillator, OBV, ROC, MFI, CCI, and many more. The features are standardized and reduced in dimensionality using PCA. A custom OpenAI Gym environment simulates the trading process using a 5-action space (HOLD, BUY, EXIT_BUY, SHORT_SELL, EXIT_SELL). A Deep Q-Network (DQN) is trained across multiple stocks to learn optimal trading decisions based on state representations. The model outputs discrete market stages that are converted into actionable trading signals. The pipeline tracks trade-level profits and evaluates performance per stock using metrics like total return, average profit, and win rate. Additionally, annotated visualizations of trading stages and signals are generated, providing insights into the model’s effectiveness.

📌 Key Features & Highlights

🔍 Technical Indicators Used:

Trend: EMA (short/medium/long), ADX, AROON, HT TrendMode

Momentum: RSI, MACD Histogram, ROC (1–20), CCI, Ultimate Oscillator, Stochastic Oscillator

Volatility: Bollinger Bands, ATR, Keltner Channels, Standard Deviation

Volume-based: OBV, OBV Change, VPT, CMF, Relative Volume

Price action: AVWAP, Fib Levels, Pivot High/Low, SAR, Price Spread

Advanced stats: Divergence indicators, HT Sine/Cycle/Phase, Relative Price Position, Market Stage Label

🧠 Model: Deep Q-Network (DQN) with experience replay, trained across multiple stock environments.

🏗️ Custom Gym Environment: Simulates market trading with rewards tied to price movements and actions.

💡 Signal Mapping: Translates predicted market stages into real-world actions: BUY, SHORT, EXIT, HOLD.

📈 Performance Metrics:

Total Profit %

Average Profit per BUY/SHORT

Win Rate for BUY and SHORT

Number of Trades per Stock

📉 Visualizations: Price charts overlaid with predicted stages, trade signals, and annotated profits.

💾 Model Saving: Scaler, PCA model, and trained policy are serialized with joblib.

📊 Evaluation Output: model_performance_by_stock.csv contains summary metrics for each stock evaluated.
