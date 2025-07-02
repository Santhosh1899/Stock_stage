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

TRADE_SYMBOL	Total_BUY_Profit(%)	Total_SHORT_Profit(%)	Total_Profit(%)	BUY_Trades	SHORT_Trades	Avg_BUY_Profit(%)	Avg_SHORT_Profit(%)	WinRate_BUY(%)	WinRate_SHORT(%)
ABBOTINDIA	52.07652772	2.043727928	54.12025565	29	12	1.795742335	0.170310661	72.4137931	75
ABCAPITAL	77.98095226	5.82086579	83.80181805	28	14	2.785034009	0.415776128	89.28571429	64.28571429
ABFRL	80.66117225	5.342762213	86.00393446	27	12	2.987450824	0.445230184	81.48148148	75
ACC	63.63073757	21.00029992	84.63103748	30	15	2.121024586	1.400019994	70	53.33333333
ADANIENT	81.03317983	13.83945822	94.87263804	24	13	3.376382493	1.064573709	87.5	69.23076923
ADANIPORTS	60.84323176	46.98486561	107.8280974	30	15	2.028107725	3.132324374	70	73.33333333
ALKEM	66.38356946	14.06617966	80.44974912	28	5	2.370841766	2.813235932	75	80
AMBUJACEM	52.3352278	34.72834556	87.06357336	27	13	1.93834177	2.671411197	70.37037037	53.84615385
APOLLOHOSP	43.69476571	-5.419491979	38.27527373	30	6	1.45649219	-0.903248663	83.33333333	66.66666667
APOLLOTYRE	65.82218654	-6.179785564	59.64240098	26	19	2.531622559	-0.325251872	84.61538462	57.89473684
ASHOKLEY	48.71314658	7.149963596	55.86311017	28	27	1.739755235	0.264813467	75	55.55555556
ASIANPAINT	12.26112666	11.46186281	23.72298948	19	16	0.645322456	0.716366426	68.42105263	62.5
ASTRAL	46.82960129	33.51420609	80.34380737	23	12	2.036069621	2.792850507	78.26086957	75
ATUL	29.51873559	-6.029940601	23.48879499	36	9	0.819964878	-0.6699934	66.66666667	44.44444444
AUBANK	30.18078345	6.710390107	36.89117355	21	22	1.437180164	0.305017732	66.66666667	45.45454545
AUROPHARMA	86.24745841	9.30766825	95.55512666	33	6	2.613559346	1.551278042	78.78787879	83.33333333
AARTIIND	79.98894684	9.02284768	89.01179452	31	16	2.580288608	0.56392798	77.41935484	62.5
ABB	95.81554339	3.995968621	99.81151201	30	6	3.193851446	0.66599477	73.33333333	66.66666667
![image](https://github.com/user-attachments/assets/c6965bf8-44a3-4c2e-a875-45f2d48ddfe7)





