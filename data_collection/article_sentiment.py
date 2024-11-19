import yfinance as yf
import pandas_ta as ta
import os
from dotenv import load_dotenv
import datetime
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd
import time

# Load environment variables
load_dotenv()

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
GNEWS_KEY = os.getenv('GNEWS_KEY')
print(GNEWS_KEY + " IS THE KEY")

# Define the ticker symbols for top stocks in different sectors
tickerSymbols = ['TSLA', 'AAPL', 'AMZN']

yesterday = datetime.date.today() - datetime.timedelta(days=1)

# Define a function to get the sentiment of a text
def get_sentiment(text):
    return sid.polarity_scores(text)

# Define a function to get news articles
def get_news(tickerSymbol):
    url = f"https://gnews.io/api/v4/search?q={tickerSymbol}&token={GNEWS_KEY}&max=10&lang=en"
    response = requests.get(url)
    return response.json()

# Loop over the ticker symbols
for tickerSymbol in tickerSymbols:
    articles = get_news(tickerSymbol)
    sentiments = []
    article_data = []
    for article in articles['articles']:
        sentiment = get_sentiment(article['content'])
        sentiments.append(sentiment['compound'])
        article_data.append({
            'title': article['title'],
            'sentiment': sentiment['compound']
        })

    # Save article data to a CSV file
    article_df = pd.DataFrame(article_data)
    article_df.to_csv(f'articles/{tickerSymbol}_articles.csv', index=False)

    # Get data on this ticker
    tickerData = yf.Ticker(tickerSymbol)
    # Get fundamental data
    fundamental_df = tickerData.info
    # Get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2018-1-1', end=yesterday)
    
    actions_df = tickerData.actions
    # Generate technical indicators
    tickerDf.ta.sma(close='Close', length=20, append=True)
    tickerDf.ta.ema(close='Close', length=20, append=True)
    tickerDf.ta.rsi(close='Close', length=14, append=True)
    tickerDf.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
    tickerDf.ta.bbands(close='Close', length=20, append=True)
    tickerDf.ta.vwap(high='High', low='Low', close='Close', volume='Volume', append=True)
    tickerDf.ta.stoch(high='High', low='Low', close='Close', append=True)
    tickerDf.ta.obv(append=True)
    tickerDf.ta.atr(append=True)
    tickerDf.ta.cmf(append=True)

    # Add fundamental data to the DataFrame
    fundamental_data = {key: fundamental_df.get(key, pd.NA) for key in ['marketCap', 'forwardPE', 'dividendYield', 'profitMargins', 'bookValue', 'earningsQuarterlyGrowth', 'debtToEquity', 'revenueQuarterlyGrowth', 'returnOnAssets', 'returnOnEquity']}
    tickerDf = tickerDf.assign(**fundamental_data)
    tickerDf['news_sentiment'] = pd.Series(sentiments).mean()
    # See your data
    print(tickerDf)

    # Save data to a csv file
    tickerDf.to_csv(f'stock_fundamental/{tickerSymbol}_data.csv')
    time.sleep(1)