# for data wrangling:
import pandas as pd
import numpy as np
import time as tm
import datetime as dt

# for retrieval of market data:
!pip install alpha_vantage
import pandas_datareader.data as web
from alpha_vantage.timeseries import TimeSeries


# for natural language processing:
import nltk
import nltk.data
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import sentiment
from nltk import word_tokenize
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re


# for plotting:
import matplotlib.pyplot as plt


# import pre-cleaned dataset from GitHub:
dataset = pd.read_csv('https://raw.githubusercontent.com/David-Woroniuk/Medium-Articles/master/twitter_data.csv', index_col = 'date', infer_datetime_format= 'date')
dataset.head()


# define the sample period of market data:
today = dt.datetime.now()
delta = dt.timedelta(days = 1)
end_delta = dt.timedelta(days = 9)
end_of_sample = (today - delta)
start_of_sample = (end_of_sample - end_delta)

start_of_sample = start_of_sample.replace(second=0,microsecond=0)
end_of_sample = end_of_sample.replace(second=0,microsecond=0)

# define a Dataframe using the index as our defined sample:
market_data = pd.DataFrame(index=pd.date_range(start=start_of_sample,end=end_of_sample,freq = '1min'))

# call alphavantage's API:
ts = TimeSeries(key='YOUR API KEY HERE', output_format='pandas')
data, meta_data = ts.get_intraday(symbol= 'TSLA', interval='1min', outputsize='full')

# place data into market_data df:
market_data = pd.concat([market_data,data],axis=1)
market_data.index = market_data.index.strftime("%d/%m/%Y %H:%M")
market_data.dropna(axis = 0, how = 'any', inplace = True)
market_data.head()



# now merge twitter data and market data:
market_data = market_data.merge(dataset, left_index=True, right_index=True, how='inner')

# rename the columns for ease:
market_data.rename(columns={"1. open": "open",
                            "2. high": "high",
                            "3. low" : "low",
                            "4. close" : "close",
                            "5. volume" : "volume",
                            "content" : "tweet_content"}, inplace = True)
                            
                            
# remove additional regex characters from twitter data:
for i in range(len(market_data)):
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\n', ' ')
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\r', ' ')
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\t', '')
  market_data.iloc[i, 5] = market_data.iloc[i, 5].replace('\xa0', '')
  
  
# define functions to determine retweets, mentions and hashtags: 
def find_retweeted(tweet):
    '''This function will extract the twitter handles of retweed people'''
    return re.findall('(?<=RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

def find_mentioned(tweet):
    '''This function will extract the twitter handles of people mentioned in the tweet'''
    return re.findall('(?<!RT\s)(@[A-Za-z]+[A-Za-z0-9-_]+)', tweet)  

def find_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet) 


# apply functions to market_data, generate output columns:
market_data['retweeted'] = market_data['tweet_content'].apply(find_retweeted)
market_data['mentioned'] = market_data['tweet_content'].apply(find_mentioned)
market_data['hashtags'] = market_data['tweet_content'].apply(find_hashtags)





# load nltk vader sentiment analysis as analyzer:
analyzer = SentimentIntensityAnalyzer()

# define empty lists:
compound_sentiment = []
vs_pos = []
vs_neu = []
vs_neg = []

# for each row in 'tweet_content', analyze sentiment:
for i in range(0, len(market_data)):
  compound_sentiment.append(analyzer.polarity_scores(market_data['tweet_content'][i])['compound'])
  vs_pos.append(analyzer.polarity_scores(market_data['tweet_content'][i])['pos'])
  vs_neu.append(analyzer.polarity_scores(market_data['tweet_content'][i])['neu'])
  vs_neg.append(analyzer.polarity_scores(market_data['tweet_content'][i])['neg'])


# generate output columns:
market_data['total_sentiment'] = compound_sentiment
market_data['positive'] = vs_pos
market_data['neutral'] = vs_neu
market_data['negative'] = vs_neg



#STRATEGY 1:
# copy market_data to maintain data integrity:
strategy_one = market_data.copy(deep = True)

# devise simple moving average strategy:
strategy_one['21_SMA'] = strategy_one['close'].rolling(window = 21).mean()
strategy_one['50_SMA'] = strategy_one['close'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_one = strategy_one[strategy_one['21_SMA'].notna()]
strategy_one['trigger'] = 0

# define trading triggers:
strategy_one.loc[(strategy_one['21_SMA'] < strategy_one['50_SMA']), 'trigger'] = -1
strategy_one.loc[(strategy_one['21_SMA'] > strategy_one['50_SMA']), 'trigger'] = 1




#STRATEGY 2:
# copy market_data to maintain data integrity:
strategy_two = market_data.copy(deep=True)

# devise a sentiment moving average strategy:
strategy_two['21_SMA_Sentiment'] = strategy_two['total_sentiment'].rolling(window = 21).mean()
strategy_two['50_SMA_Sentiment'] = strategy_two['total_sentiment'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_two = strategy_two[strategy_two['21_SMA_Sentiment'].notna()]
strategy_two['trigger'] = 0

# define trading triggers:
strategy_two.loc[(strategy_two['21_SMA_Sentiment'] < strategy_two['50_SMA_Sentiment']), 'trigger'] = -1
strategy_two.loc[(strategy_two['21_SMA_Sentiment'] > strategy_two['50_SMA_Sentiment']), 'trigger'] = 1



#STRATEGY 3:
# copy market_data to maintain data integrity:
strategy_three = market_data.copy(deep=True)

# devise sentiment variables:
strategy_three['21_SMA_Sentiment'] = strategy_three['total_sentiment'].rolling(window = 21).mean()
strategy_three['50_SMA_Sentiment'] = strategy_three['total_sentiment'].rolling(window = 50).mean()
strategy_three['21_SMA_Positive'] = strategy_three['positive'].rolling(window = 21).mean()
strategy_three['50_SMA_Positive'] = strategy_three['positive'].rolling(window = 50).mean()
strategy_three['21_SMA_Negative'] = strategy_three['negative'].rolling(window = 21).mean()
strategy_three['50_SMA_Negative'] = strategy_three['negative'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_three = strategy_three[strategy_three['21_SMA_Sentiment'].notna()]
strategy_three['trigger'] = 0

# define trading triggers:
strategy_three.loc[(strategy_three['21_SMA_Sentiment'] < strategy_three['50_SMA_Sentiment']) & 
                   (strategy_three['21_SMA_Positive'] < strategy_three['50_SMA_Positive']) &
                   (strategy_three['21_SMA_Negative'] > strategy_three['50_SMA_Negative']), 'trigger'] = -1


strategy_three.loc[(strategy_three['21_SMA_Sentiment'] > strategy_three['50_SMA_Sentiment']) & 
                   (strategy_three['21_SMA_Positive'] > strategy_three['50_SMA_Positive']) &
                   (strategy_three['21_SMA_Negative'] < strategy_three['50_SMA_Negative']), 'trigger'] = 1
                   
                   
STRATEGY 4:
# copy market_data to maintain data integrity:
strategy_four = market_data.copy(deep=True)

# devise sentiment variables:
strategy_four['21_SMA'] = strategy_four['close'].rolling(window = 21).mean()
strategy_four['50_SMA'] = strategy_four['close'].rolling(window = 50).mean()

strategy_four['21_SMA_Sentiment'] = strategy_four['total_sentiment'].rolling(window = 21).mean()
strategy_four['50_SMA_Sentiment'] = strategy_four['total_sentiment'].rolling(window = 50).mean()

strategy_four['21_SMA_Positive'] = strategy_four['positive'].rolling(window = 21).mean()
strategy_four['50_SMA_Positive'] = strategy_four['positive'].rolling(window = 50).mean()

strategy_four['21_SMA_Negative'] = strategy_four['negative'].rolling(window = 21).mean()
strategy_four['50_SMA_Negative'] = strategy_four['negative'].rolling(window = 50).mean()

# remove NA, initialise trigger to 0:
strategy_four = strategy_four[strategy_four['50_SMA'].notna()]
strategy_four['trigger'] = 0

# define trading triggers:
strategy_four.loc[(strategy_four['21_SMA'] < strategy_four['50_SMA']) &
                  (strategy_four['21_SMA_Sentiment'] < strategy_four['50_SMA_Sentiment']) & 
                  (strategy_four['21_SMA_Positive'] < strategy_four['50_SMA_Positive']) &
                  (strategy_four['21_SMA_Negative'] > strategy_four['50_SMA_Negative']), 'trigger'] = -1


strategy_four.loc[(strategy_four['21_SMA'] > strategy_four['50_SMA']) &
                  (strategy_four['21_SMA_Sentiment'] > strategy_four['50_SMA_Sentiment']) & 
                  (strategy_four['21_SMA_Positive'] > strategy_four['50_SMA_Positive']) &
                  (strategy_four['21_SMA_Negative'] < strategy_four['50_SMA_Negative']), 'trigger'] = 1
                  
                  
                  
                  
#BACKTEST: 
def trade(data, price_change, trigger, capital = 10_000, maximum_long = 1, maximum_short = 1):
    """
    price_change = market price change.
    trigger = 1 is a buy order, -1 is sell order.
    capital = initial capital committed to algorithm.
    maximum_long = maximum quantity that can be purchased in one period.
    maximum_short = maximum quantity that can be sold in one period.
    """
    starting_capital = capital
    sell_states = []
    buy_states = []
    inventory = 0

    def buy(i, capital, inventory):
        shares = capital // price_change[i]
        if shares < 1:
            print('{}: total balance {}, not enough capital to buy a unit price {}'.format(data.index[i], capital, price_change[i]))
        else:
            if shares > maximum_long:
                buy_units = maximum_long
            else:
                buy_units = shares
            capital -= buy_units * price_change[i]
            inventory += buy_units
            print('{}: buy {} units at price {}, total balance {}'.format(data.index[i], buy_units, buy_units * price_change[i], capital))
            buy_states.append(0)
        return capital, inventory


    
    for i in range(price_change.shape[0] - int(0.025 * len(price_change))):
        state = trigger[i]
        if state == 1:
            capital, inventory = buy( i, capital, inventory)
            buy_states.append(i)
        elif state == -1:
            if inventory == 0:
                    print('{}: cannot sell anything, inventory 0'.format(data.index[i]))
            else:
                if inventory > maximum_short:
                    sell_units = maximum_short
                else:
                    sell_units = inventory
                inventory -= sell_units
                total_sell = sell_units * price_change[i]
                capital += total_sell
                try:
                    RoC = ((price_change[i] - price_change[buy_states[-1]]) / price_change[buy_states[-1]]) * 100
                except:
                    RoC = 0
                print('{}, sell {} units at price {}, RoC: {}%, total balance: {}'.format(data.index[i], sell_units, total_sell, RoC, capital))
            sell_states.append(i)

    RoC = ((capital - starting_capital) / starting_capital) * 100
    total_gains = capital - starting_capital
    consolidated_position = (capital + inventory * price_change[i])
    print('*'*150)
    print("Consolidated Position:{}, Realised Gains:{}, Realised Return on Capital:{}, Inventory:{}".format(consolidated_position, total_gains, RoC, inventory))
    print('*'*150)


    # Plotting:
    value = data['close']
    fig = plt.figure(figsize = (20,10))
    plt.plot(value, color = 'b', lw=2.)

    # Plot the Entry and Exit Signals generated by the algorithm:
    plt.plot(value, '^', markersize=8, color='g', label = 'Trigger Entry', markevery = buy_states)
    plt.plot(value, 'v', markersize=8, color='r', label = 'Trigger Exit', markevery = sell_states)

    # Chart Title displaying the Absolute Returns, Return on Capital & Benchmark Returns:
    plt.title('Consolidated Position: {}, Realised Absolute Returns: {}, Realised Return on Capital: {}%, Inventory: {}'.format(round(consolidated_position,2), round(total_gains,2), round(RoC,2), inventory))
    plt.legend()
    plt.show()

    return buy_states, sell_states, total_gains, RoC
    
    
#OUTCOME 1:
buy_states, sell_states, total_gains, RoC = trade(strategy_one, strategy_one['close'], strategy_one['trigger'])
    
#OUTCOME 2:
buy_states, sell_states, total_gains, RoC = trade(strategy_two, strategy_two['close'], strategy_two['trigger'])
    
#OUTCOME 3:
buy_states, sell_states, total_gains, RoC = trade(strategy_three, strategy_three['close'], strategy_three['trigger'])
    
#OUTCOME 4:
buy_states, sell_states, total_gains, RoC = trade(strategy_four, strategy_four['close'], strategy_four['trigger'])
    
