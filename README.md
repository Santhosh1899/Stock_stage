## Stock Stage prediction and trading

This project implements a comprehensive pipeline for unsupervised market stage prediction and trading signal generation using **Deep Reinforcement Learning**. It begins with an extensive set of technical indicators derived from stock market data. These include trend, momentum, volatility, volume, and price-based features such as RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic Oscillator, OBV, ROC, MFI, CCI, and many more. The features are standardized and reduced in dimensionality using PCA. A custom  Gym environment simulates the trading process using a 5-action space (HOLD, BUY, EXIT_BUY, SHORT_SELL, EXIT_SELL). A Deep Q-Network (DQN) is trained across multiple stocks to learn optimal trading decisions based on state representations. The model outputs discrete market stages that are converted into actionable trading signals. The pipeline tracks trade-level profits and evaluates performance per stock using metrics like total return, average profit, and win rate. Additionally, annotated visualizations of trading stages and signals are generated, providing insights into the model’s effectiveness.

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



![WhatsApp Image 2025-06-14 at 21 05 29](https://github.com/user-attachments/assets/5ec47053-ea92-4b39-a893-c289a0c53996)

![WhatsApp Image 2025-06-14 at 21 00 53](https://github.com/user-attachments/assets/e4896001-88b9-431e-8c31-8443248cc21c)

![WhatsApp Image 2025-06-14 at 21 01 06](https://github.com/user-attachments/assets/3b19017f-6ebd-48c1-8366-86bd5839e17a)

![WhatsApp Image 2025-06-14 at 21 01 32](https://github.com/user-attachments/assets/7dac215b-58c8-48d4-9d0e-adbf4b0255f8)

![WhatsApp Image 2025-06-14 at 21 01 45](https://github.com/user-attachments/assets/2d6898ab-bd3d-490f-ab82-b124b6506b9f)

![WhatsApp Image 2025-06-14 at 21 02 03](https://github.com/user-attachments/assets/403d7c34-0705-4e07-897b-c75a6a9390b5)

![WhatsApp Image 2025-06-14 at 21 02 32](https://github.com/user-attachments/assets/37d52d8d-6f5d-4e03-842e-0ba4f8f58319)

![WhatsApp Image 2025-06-14 at 21 02 57](https://github.com/user-attachments/assets/0e236cf8-e612-4946-9565-9701b643879a)

![WhatsApp Image 2025-06-14 at 21 03 30](https://github.com/user-attachments/assets/ef70a9b9-9ccd-4caf-a9bd-9b988f2dd1d8)

![WhatsApp Image 2025-06-14 at 21 03 48](https://github.com/user-attachments/assets/46114df8-83bc-4ecd-a7d6-cdc8683a32e9)

![WhatsApp Image 2025-06-14 at 21 04 00](https://github.com/user-attachments/assets/cb37d628-9271-4c6d-a3b8-d06bc58d61b2)

![WhatsApp Image 2025-06-14 at 21 04 10](https://github.com/user-attachments/assets/241b9169-2a5c-4f87-bcd3-dec79ab8788c)

![WhatsApp Image 2025-06-14 at 21 04 24](https://github.com/user-attachments/assets/a3159771-1407-4bea-b4dc-980527c1feeb)

![WhatsApp Image 2025-06-14 at 21 04 40](https://github.com/user-attachments/assets/f63027b3-eb0d-47d9-801a-816760793a5e)

![WhatsApp Image 2025-06-14 at 21 04 52](https://github.com/user-attachments/assets/e129889b-e8d9-4946-9603-08d7e263c454)

![WhatsApp Image 2025-06-14 at 21 05 03](https://github.com/user-attachments/assets/ad9fad30-7acb-41d6-a819-0426d7b29bea)


###  **Trading summary from 2024 to 2025**

![image](https://github.com/user-attachments/assets/c6965bf8-44a3-4c2e-a875-45f2d48ddfe7)


The stock stage prediction model demonstrated strong performance across a wide range of stocks, particularly in generating profitable BUY signals. Notably, ADANIPORTS, AUROPHARMA, ABB, and ADANIENT stood out with total profit percentages exceeding 90%, with ADANIPORTS achieving the highest overall return of 107.83%. This was driven by a balanced contribution from both BUY and SHORT trades, highlighting the model’s adaptability in both bullish and bearish conditions.

In terms of average profit per BUY trade, ABB, ADANIENT, ABFRL, and AARTIIND reported impressive values, often exceeding 2.5% per trade, indicating the model's ability to enter trades with strong reward potential. Similarly, ALKEM, ASTRAL, and ADANIPORTS also produced significant average profits on SHORT trades, showing the model's capability to capture downside movements effectively when conditions were favorable.

The model also achieved high win rates for BUY trades, with several stocks like ABCAPITAL, APOLLOTYRE, APOLLOHOSP, and AUROPHARMA recording success rates of over 75%, reflecting consistent accuracy in timing long entries and exits. SHORT trade win rates were slightly lower and more variable, with AUROPHARMA achieving the highest at 83.33%, followed by ALKEM and ADANIPORTS, both above 70%.

However, some stocks such as ATUL, APOLLOHOSP, and APOLLOTYRE experienced negative or marginal returns from SHORT trades, indicating the model occasionally misjudged bearish momentum. These cases suggest room for improvement, potentially through enhanced reward design, more balanced training data, or strategy constraints that better reflect real-world short selling risks.

In summary, the model is highly effective in generating BUY signals with strong profitability and reliability, while SHORT strategies show promise but may require further refinement. The approach proves scalable across multiple stocks, offering a solid foundation for AI-driven trading systems.



