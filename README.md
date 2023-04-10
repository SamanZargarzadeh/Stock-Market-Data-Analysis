# Stock Market Data Analysis
Comparing various trading strategies in a Python based program for stock market data from Y!Finance library, visualizing S&amp;P 500 data using Tableau, and predicting stock prices using the LSTM artificial neural network model.



## Data
For exploring stock market data, we use Python libraries, and we select S&P 500 companies for our sample for analysis.
### Y!Finance Package
Rather than relying on a single dataset, we are using Y!Finance in this study. Ran Aroussi's Y!finance library is widely used because it provides an easy way to retrieve the up-to-date financial data hosted on Yahoo Finance. Through the use of this library, we are able to get information such as S&P 500 stock market data and historical OHLCV (Open, High, Low, Close, Volume) data. Listed below are the benefits and drawbacks of Y!finance. [4] 
Pros	Cons
Quick and easy to set yourself up	Some methods are fragile
Free of charge	Unofficial / not necessarily maintained
High granularity of data	Can get yourself rate limited/blacklisted
Returns data directly in pandas dataframes	Lacks specialized features
### S&P 500
The Standard & Poor's 500 Index tracks the performance of 500 of the largest publicly traded corporations in the United States. The S&P 500 Index's ticker symbol is ^GSPC. It simply displays the movement in the S&P 500 stock price index. Thus, to make the dataset for our trading strategies’ exploration, we simply download the historical data from Y!Finance with the help of the code below.
 
To have more detail in our dataset, we used a dataset that had all the tickers and their sectors, as well as the pandas_datareader library to get the market cap of each stock.
 	 
After running a for loop, all of the S&P 500 companies and their attributes are added to a new dataset for a time period of more than six years. (The code will be uploaded)
## Exploratory Method
Due to the purpose of this project, which is to make money on the stock market, we define three steps: exploring and familiarizing ourselves with the stock market, predicting, and building the ability to trade with strategy. First of all, we explore the data and visualize it with the help of Tableau and Python to give us a better view of different sectors and stocks' market caps. As a second step, we applied three different trading strategies and compared them. At the end, we utilized machine learning methods and predicted the 2023 ^GSPC movement.


### Trading Strategy
The goal of all the strategies is the same. We want to find the buy and sell signal, but in every strategy, we go with a different approach. The programming for this section was done in Python. For each method function defined, based on the buy and sell signals, a position column is created, which is 1 in the case that the user buys the stock and possesses it, and -1 in the case of selling.
#### Simple Moving Average 
The moving average is one of the simplest and most common ways of predicting  stock market movement. Our method here would be to define a function to calculate the average closing price for different rolling periods of time. Then, to identify the sell and buy signals, we can compare the moving averages for two different periods of time one and while the SMA for the shorter period crosses and goes above the longer period, it is a signal to buy; if the shorter period SMA goes below the longer one, it is a sign to sell.
 

#### Mean Reversion 
The concept behind the mean reversion theory is buying low and selling high, which is evident in the diagram. For calculating it, we first make a ratio parameter that indicates the closing price divided by the simple moving average amount. By looking at the description of the ratio, the 15th and 85th percentiles are added to the graph, and it helps us to mark the sell and buy signals, which are presented below. A sell signal would be the time that the ratio goes above the red line, which is the time that our position parameter will change to 1, and a buy signal would be the time that the ratio goes below the 15th percentile (the green line). A modification was made here so that we can also consider the return of the ratio line below the red line to be a buy signal. 




#### Stochastic RSI 
Technical analysis uses the Stochastic oscillator formula to calculate the Stochastic RSI (Relative Strength Index), which ranges from zero to one. Using this formula, traders can determine whether an asset is overbought or oversold. The formula is presented below: [5] 

StochRSI =  (RSI - Min(RSI))/(Max(RSI) - Min(RSI)) 
where: RSI = 100 -[100/(1 + (Average Gain)/(Average Loss))]
For calculating average gain and loss, an exponential moving average is defined with a 14-day period. In the next step, the RSI and StochRSI are calculated, which are shown in the graph above. When the graph goes above the red line, it is an indicator of overbought, and when it comes below the green line, it is time to buy as it is oversold. The stock is marked with buy and sell signals.



### Prediction with LSTM Model
Long short-term memory (LSTM) networks are an advanced kind of recurrent neural networks (RNNs). Looped RNNs utilize prior data to produce a final output. RNNs can only connect information from the recent past. LSTMs, a sort of RNN that remembers information over time, are superior at stock price prediction. [3]
For making the LSTM model these steps should be followed:
	Scaling the Data 
Because the LSTM model requires input in the range of 0 to 1, we use the sklearn.prepocessing library to convert our data to this range.
 
	Splitting the Dataset to test and train
Our dataset consists of the 10-year close price of the ^GSPC, split into training and test datasets. The first 80% of our data is referred to as the train dataset, while the remainder is used for testing.
	Timestepping Train Data
As previously stated, neural networks require prior data to predict the following day's data. As a result, the closing price should be divided into x and y. Our first X would be the closing price from the first to the 60th day, and the Y would be the price on the 61st day. The second X corresponds to days 2 to 61, and the Y corresponds to days 62 and so on. By doing so, we covered the entire train dataset and prepared it for the model.
 
	LSTM Model Creation
In this project, the keras library is used to build the LSTM model. Using the library functions, we can create the LSTM model based on the train dataset.
 
	Test Dataset prediction
The test dataset, like the train dataset, should be time stepped. The LSTM model is then run on the test dataset, and the result is our prediction of the closing price.
 	 
## Analysis
### Stock Market Data Exploration
 
For better insight into the performance of different sectors and their market caps, two figures are presented. The figure below shows the growth of a one dollar investment (invested on 1 Jan 2022). By looking at the different sector results, we can conclude that energy industries were the most valuable sector of the 2022 stock market, which might be due to oil sanctions. 
 
Moreover, by getting the average percentage of growth over a 30 day (month) and 365 day (year) period, we calculate the monthly and yearly performance of each company and present them by their market cap. Based on the graph below, we can conclude that leading companies generate profits ranging from 0-5% monthly and 0-100% annually. If you are looking to make more profit, you should go into the trade of smaller companies and take the risk. After choosing a stock for trading based on the graphs, it is time to compare the trading strategies.


### Comparing Trading Strategies
Based on the method described, we can apply the buy and sell signals to the ^GSPC ticker data to compare the functions of various strategies. To gain a better understanding of the data, we created a case study in which traders begin trading with one dollar and their investment’s growth is visualized. By adding a new column to our dataset, we mark the buy and sell signals, and based on that, we can find out how long the trader owned the stock. The graph below shows the trader’s investment with these three strategies and also with the buy-and-hold strategy.

 
### LSTM Model Results
Based on the method described in the previous chapter, an LSTM model was built in this project and tested on 20% of our dataset; the results are shown below. The blue line represents the actual closing price of the ^GSPC ticker, while the orange line represents the model's price prediction. The output is rescaled from one to zero to the real range in dollars for this graph.

 

Two types of errors are calculated for the evaluation of our model, as shown below:
 

The LSTM model was created to predict the next day's closing price for traders. So, for our dataset, which ends on November 25th, our goal would be to predict the closing price on November 26th. This objective was achieved by running the model on the last 60 days of our dataset, and the amount is reported below:
 

## Conclusion
This project covers a variety of subjects, including data exploration, trading strategies, and the LSTM model, all of which are highlighted in this chapter. First of all, data exploration is done which provides us with a better understanding of the sectors, market cap, and returns that each sector generated for the prior year. It has been determined that the energy sector was the most successful one for investments over the past year. In addition, the comparison of market cap and returns reveals that the leaders are the most secure places to invest, despite having low percentages of return on their investments.
Second, there is a discussion of three distinct trading strategies. By examining the growth graphs of these strategies, we can see that in the year that the market fell, you would have had less than 0.8 dollars if you bought one dollar in these days last year. However, if you use our Python program, you can make even 1.3 dollars from that one dollar, which is a high return. The Stochastic Relative Strength Index (RSI) and Mean Reversion Strategy is unquestionably performing significantly better than the simple moving average. However, it can be seen that the best strategy would change depending on the ticker and the timing of the trade. Nevertheless, one thing that is evident is that utilizing these strategies is unquestionably superior to the buy-and-hold strategy.
Last but not least, the LSTM model assists us in predicting the price for the following day, which is something that traders find to be of great value. Traders could run our model on the data from the previous 60 days after the market has closed each day to get an estimate of the price at which the market will close the following day. Our model has a root absolute percentage error of 1.83%, which is pretty good for making predictions on complex topics like the stock market.
This project might end up being a useful tool for traders, allowing them to sell and buy with more insight. And could serve as an excellent illustration of how data and programming knowledge can be applied in the real world data.
![image](https://user-images.githubusercontent.com/88157400/231009918-9864f121-f3ad-40a8-8315-ee7f04bfef33.png)
