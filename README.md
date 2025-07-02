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


![image](https://github.com/user-attachments/assets/c6965bf8-44a3-4c2e-a875-45f2d48ddfe7)

The stock stage prediction model demonstrated strong performance across a wide range of stocks, particularly in generating profitable BUY signals. Notably, ADANIPORTS, AUROPHARMA, ABB, and ADANIENT stood out with total profit percentages exceeding 90%, with ADANIPORTS achieving the highest overall return of 107.83%. This was driven by a balanced contribution from both BUY and SHORT trades, highlighting the model’s adaptability in both bullish and bearish conditions.

In terms of average profit per BUY trade, ABB, ADANIENT, ABFRL, and AARTIIND reported impressive values, often exceeding 2.5% per trade, indicating the model's ability to enter trades with strong reward potential. Similarly, ALKEM, ASTRAL, and ADANIPORTS also produced significant average profits on SHORT trades, showing the model's capability to capture downside movements effectively when conditions were favorable.

The model also achieved high win rates for BUY trades, with several stocks like ABCAPITAL, APOLLOTYRE, APOLLOHOSP, and AUROPHARMA recording success rates of over 75%, reflecting consistent accuracy in timing long entries and exits. SHORT trade win rates were slightly lower and more variable, with AUROPHARMA achieving the highest at 83.33%, followed by ALKEM and ADANIPORTS, both above 70%.

However, some stocks such as ATUL, APOLLOHOSP, and APOLLOTYRE experienced negative or marginal returns from SHORT trades, indicating the model occasionally misjudged bearish momentum. These cases suggest room for improvement, potentially through enhanced reward design, more balanced training data, or strategy constraints that better reflect real-world short selling risks.

In summary, the model is highly effective in generating BUY signals with strong profitability and reliability, while SHORT strategies show promise but may require further refinement. The approach proves scalable across multiple stocks, offering a solid foundation for AI-driven trading systems.



