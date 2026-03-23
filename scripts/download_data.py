# import necessary libraries
import yfinance as yf
import pandas as pd
import time
import os

from massive import RESTClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# load api key from .env
news_api_key = os.getenv("NEWS_API_KEY")
client = RESTClient(news_api_key)

# stocks i want to get data on
tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "V", "JNJ", "AMZN", "WMT", "TSLA"]

# function to download market data
def download_market_data(tickers, period, interval, model):
    # initialise data
    data = []

    # loop through tickers
    for ticker in tickers:
        try:
            # get data according to ticker, period and interval
            df = yf.download(ticker, period=period, interval=interval, progress=False)

            # check for data 
            if len(df) > 0:
                # check for multi index
                if isinstance(df.columns, pd.MultiIndex):
                    # flatten multi index column
                    df.columns = df.columns.get_level_values(0)

                # reset index
                df = df.reset_index()
                # add ticker column
                df['ticker'] = ticker
                # append into data 
                data.append(df)
    
            else:
                # no data available
                print(f"No data")
        except Exception as e:
            # error message
            print(f"Error fetching data for {ticker}: {e}")
    
    if data:
        # concat data array into single dataframe
        combined_df = pd.concat(data, ignore_index=True)
        # check for datasets folder
        os.makedirs("datasets", exist_ok=True)
        # define file path
        csv_filename = os.path.join("datasets", f"{model}_market_data.csv")
        # convert file path into csv and save it
        combined_df.to_csv(csv_filename, index=False)
        # get out of function
        return
    else:
        # no data and return
        print("No data to save.")
        return
    
# download_market_data(tickers, period="60d", interval="15m", model="rf")
# download_market_data(tickers, period="5y", interval="1d", model="lstm")

# function to get stock sentiment
def fetch_stock_news(ticker, days_back, retries=3):
    try:
        # get current date time
        to_date = datetime.now()
        # calculate date of when you want to get the news from
        from_date = to_date - timedelta(days=days_back)

        # intialise news item array
        news_item = []
        # get news item from api
        for n in client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=from_date.strftime('%Y-%m-%d'),
            published_utc_lte=to_date.strftime('%Y-%m-%d'),
            order="desc",
            limit=100,
            sort="published_utc"
        ):
            # append news into news item array
            news_item.append(n)

        # check for news data
        if len(news_item) == 0:
            print(f" {ticker}: No news found")
            # return pd dataframe object if none
            return pd.DataFrame()
        
        # initialise news data array
        news_data = []

        # map sentiment to values
        sentiment_map = {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        }

        # loop through news item
        for article in news_item:
            # initialise sentiment to 0
            sentiment = 0

            # check for insights in article
            if article.insights:
                # loop through insight
                for insight in article.insights:
                    # get sentiment for ticker
                    if insight.ticker == ticker:
                        # get sentiment score using sentiment map
                        sentiment = sentiment_map.get(insight.sentiment, 0)
                        # break out of loop
                        break

            # append object into news data array
            news_data.append({
                "date": pd.to_datetime(article.published_utc).date(),
                "title": article.title,
                "sentiment": sentiment,
                "publisher": article.publisher.name if hasattr(article, 'publisher') else 'Unknown'
            })

        # convert news data into pandas dataframe
        df = pd.DataFrame(news_data)

        # group article sentiment by date
        daily_sentiment = df.groupby('date').agg({
            'sentiment': ['mean', 'std', 'count']
        }).reset_index()

        # define columns
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'news_count']
        # convert to pandas datetime
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

        # return daily sentiment
        return daily_sentiment

    except Exception as e:
        # error for rate limiting
        if "429" in str(e) and retries > 0:
            # increase wait time according to tries
            wait_time = 60 * (4 - retries)
            print(f"Rate limit hit for {ticker}, waiting {wait_time}s before retry {4-retries}/3...")
            # sleep according to wait time
            time.sleep(wait_time)
            # try fetching stock news again
            return fetch_stock_news(ticker, days_back, retries - 1)
        
        # print error
        print(f"Error fetching news for {ticker}: {e}")

# function to download all news
def download_all_news(tickers, days_back):
    # initialise news sentiment object
    news_sentiment = {}
    # loop through tickers
    for i, ticker in enumerate(tickers):
        # fetch sentiment data for specific stock
        sentiment_data = fetch_stock_news(ticker, days_back)
        # check for data
        if sentiment_data is not None and not sentiment_data.empty:
            # save sentiment data accordiing to ticker
            news_sentiment[ticker] = sentiment_data

        # sleep to avoid rate limit
        if i < len(tickers) - 1:
            print(f"Waiting 90 seconds before next ticker...")
            time.sleep(90)

    if news_sentiment:
        # initialise combine data array
        combined_data = []
        # loop through sentiment object
        for ticker, df in news_sentiment.items():
            # copy data frame
            df_copy = df.copy()
            # add ticker
            df_copy['ticker'] = ticker
            # append copy into combined data
            combined_data.append(df_copy)
        
        # concat data in array
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # check for datasets folder
        os.makedirs("datasets", exist_ok=True)
        # define csv file name and path
        csv_filename = os.path.join("datasets", f"{ticker}_news_sentiment.csv")
        # convert into csv and save
        combined_df.to_csv(csv_filename, index=False)
        print(f"Saved news sentiment data to {csv_filename}")

# function to combine all stock news sentiment
def combine_news_sentiment(tickers, input="datasets", output="datasets/news_sentiment.csv"):
    # initialise array
    dfs = []
    successful = []
    failed = []

    # loop through tickers
    for ticker in tickers:
        # get csv of each ticker sentiment
        file_path = os.path.join(input, f"{ticker}_news_sentiment.csv")

        # check if file exists
        if os.path.exists(file_path):
            # read csv into data frame
            df = pd.read_csv(file_path)
            
            # add ticker if not defined
            if 'ticker' not in df.columns:
                df[ticker] = ticker

            # append into dfs and successful array
            dfs.append(df)
            successful.append(ticker)

        else:
            # append into failed array
            failed.append(ticker)

    if dfs:
        # concat data in array
        combined_df = pd.concat(dfs, ignore_index=True)
        # convert to csv and save
        combined_df.to_csv(output, index=False)

    else:
        # message printed if no csv found
        print("No CSV files found")

# download_all_news(tickers, days_back=60)
# combine_news_sentiment(tickers)