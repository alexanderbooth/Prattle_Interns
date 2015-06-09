

# This version tracks the portfolio balance on a day to day basis no matter trading happens or not. 
# strategy is all in and all out
#get_ipython().magic(u'matplotlib inline')
import dateutil.parser
import json
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import pandas.io.data
import requests
import datetime
from datetime import timedelta
import calendar
import math
import pdb
import operator


def parseDate(date):
    try:
        return dateutil.parser.parse(date).date()
    # parse the date to timestamp such that it can be resampled. Datetime objects cannot be resampled.
    # parse the json date to the pandas datetime
    except:
        return None

def getBankData(bank):
    # Config
    root_url = 'https://pa-api-dev.net/api'
    api_key = 'uzuhm52a647vSn5wo3XMI4ZEDJ6ei0'
    headers={'Authorization': 'Bearer {0}'.format(api_key)}
    response = requests.get(url="{0}/documents/bank/{1}".format(root_url, bank),
             headers=headers, verify=False)
    json_data = json.loads(response.content)
    df = pd.DataFrame(json_data)
    
    #only want the date_scored and score column
    df = df.loc[:,['date_scored', 'score']]
    
    df['date_scored'] = df['date_scored'].apply(parseDate)
    
    return df

def getAssetData (bankdf, ticker):

    # getting the asset data on these dates
    startDate = bankdf['date_scored'].min()
    endDate = bankdf['date_scored'].max()
    assetData = pandas.io.data.get_data_yahoo(ticker, startDate, endDate)
    assetData = assetData.loc[:, ['Adj Close']]
    assetData['Date'] = assetData.index
    return assetData

def alignAssetPrice (scoreD, fullPriceD):

    '''
    Function that returns a dataframe that has asset Price, Date and LogReturns only for the days that are bank releases dates.
    Params:

    scoreD: a dictionary of {bank dates:score on that day}
    fullPriceD: a dictionary of all available daily assets prices from a period in {dates: adj close price on that day}
    '''

    # extract the prices on the date that frc has a release
    # if there is frc release on a non-week day(no price data), then put None
    priceNeed = {}
    for r in scoreD:
        if pd.Timestamp(r) in fullPriceD:
            priceNeed[r] = fullPriceD[pd.Timestamp(r)]
        else:
            priceNeed[r] = None

    # put in list of tuples in the form (date, price) and sort them according to the date
    price_tup = [(k,v) for k,v in priceNeed.iteritems()]
    price_tup = sorted(price_tup, key = lambda x:x[0])

    assetData = pd.DataFrame(data = price_tup, columns=['Date', 'Price'])
    assetData.index = assetData['Date']

    # backfill the nan values and getting the change in price for the asset

    #assetData = assetData.fillna(method = 'bfill')

    # calculate the log returns of the asset
    assetData['LogReturns'] = numpy.log(assetData['Price']).diff()
    return assetData

def momentum_mag_trading(date, dictM, Parameters, dframe):
    '''Make trading decsions on a daily basis, want to buy (return 1) if the momentum of the day is higher than some
    cut off value, and sell/short (return -1) if the momentum of the day is lower than some cut-off value. Do nothing
    otherwise.

    input: date --> a single date
           dictM --> a dictionary of date:momentum
           parameters --> the quantile/percentile to use in this iteration
           deframe --> the datafram to get the percentile data from
    '''
    if dictM[date] > dframe['momentum'].quantile(Parameters[0]):
        #print 'Take a long position on {0}'.format(date)
        return 1
    elif dictM[date] < dframe['momentum'].quantile(Parameters[1]):
        #print 'Take a short position on {0}'.format(date)
        return -1
    else:
        #print 'No position on {0}'.format(date)
        return 0


#def price_trading(d, dictP):
    #first turn yesterdays date into datetime
    #d = d.to_datetime()
    #get today's date
    #e = d + timedelta(days = 1)
    #get today's price
    #y = snpData2['Adj Close'][e]
    #if today's price is between a 6% drop in market price and a 15% drop in market price, return True
    #if y <= (dictP[d]-dictP[d]*.06) and (y > (dictP[d]- dictP[d]*.15)):
        #return True
    #if today's price gained or dropped less than 6%, return false
    #elif y > (dictP[d]-dictP[d]*.06):
        #return False
    #if today's price dropped more than 15%, return false (stop limit)
    #elif y <= (dictP[d]- dictP[d]*.15):
        #return False

def backTesting(datels, dictP, dictM, fulldf, Parameters=(0.93, 0.06), startMoney=10000):
    '''backtesting a a set of previous dates

    datels --> a list of CONTINUOUS dates to track portfolio balance
    '''
    portfolio_balance=[]
    shorts=0
    long_count = 0
    Money = startMoney
    stoplong = 0
    stopshort = float("inf")
    for d in datels:
        if d in dictM: #for a fed release day, do trading, track portfolio balance

            # for each day, we have to recalculate the momentum percentile based on new data
            dftodate = fulldf[fulldf['date_scored'] < d.to_datetime().date()]

            if  Money+(long_count-shorts)*dictP[d] > 0 :
                # we only trade when our portfolio value is positive

                    indicator = momentum_mag_trading(d, dictM, Parameters,dftodate)
                    if indicator == 1:
                        Money = Money-shorts*dictP[d]
                        #print 'repurchased {0} shorts at {1}.'.format(shorts, dictP[d])
                        shorts=0
                        stopshort = float("inf") # resetting the short value
                        if Money> dictP[d]:
                            # We buy whatever number of shares our cash allows us to buy
                            long_count= long_count + math.floor(Money/dictP[d])
                            Money= Money - math.floor(Money/dictP[d])*dictP[d]
                        else:
                            pass
                            #print 'No sufficient money on {0}'.format(d)

                    elif indicator == -1:
                        stoplong=0
                        Money = Money + long_count*dictP[d]
                    #print 'Sold {0} stocks at {1}'.format(long_count, dictP[d])
                        long_count = 0
                        if shorts < 10:
                            # We simply arbitrarily limit the number of shorts to 10
                            shorts= shorts + math.floor(Money/dictP[d])
                            Money = Money + math.floor(Money/dictP[d])*dictP[d]
                        #print 'Made a short at {1}. Now have {0} shorts that needs to be paid'.format(shorts, dictP[d])
                        #Money = Money +shorts*dictP[d] previous error
                    else:
                        pass
            else:
                print "We have no money in the bank. Call us Mike Tyson"

            # determine whether stop-loss is needed

        if long_count > 0:
            if dictP[d] > stoplong:
                stoplong = dictP[d]
            if stopLoss(d, dictP, stoplong, 'long'):
                print 'Stop loss with long'
                Money = Money + long_count*dictP[d]
                long_count = 0

        if shorts > 0:
            if dictP[d] < stopshort:
                stopshort = dictP[d]
            if stopLoss(d, dictP, stopshort, 'short'):
                print 'Stop loss with short'
                #pdb.set_trace()
                Money = Money - shorts*dictP[d]
                shorts = 0

            #print 'We have {0} in hand'.format(Money)

        portfolio_balance.append(Money+(long_count-shorts)*dictP[d])

        print 'On {3}, We have {0} dollars, {1} in stock asset, and {2} in shorts that need to be paid'.format(Money,long_count*dictP[d],shorts*dictP[d], d)
    return portfolio_balance

def stopLoss(d, dictP, bench, type = 'long', Parameter =(0.9,1.1)):
    if type == 'long':
        if dictP[d] < Parameter[0]*bench:
            return True
        else:
            return False
    elif type == 'short':
        if dictP[d] > Parameter[1]*bench:
            return True
        else:
            return False
    else:
        return False

def main():

    frcData = getBankData('frc')
    # remove all the null values from the dataframe
    frcData = frcData[~(frcData['score'].isnull() | frcData['date_scored'].isnull())]
    # getting the mean frc score on a date with several releases
    mean_frc = frcData.groupby('date_scored')['score'].mean()
    md = pd.DataFrame(mean_frc)
    # make a copy of the dataframe for further use
    mdupdate = pd.DataFrame(mean_frc)

    # make the date column from the dataframe
    mdupdate['date_scored'] = mdupdate.index
    md['date_scored'] = md.index
    # in this case, we are only doing trading after year 2000, so that we truncate the dataframe, but the copy still has
    # all the dates
    md = md[md['date_scored'] > datetime.date(2000,5,13)]
    # putting the data into dictionary
    score_dict = dict(zip(md.date_scored, md.score))
    snpData = getAssetData(md, '^GSPC')

    # make a copy for further use to create continuous time series from the snpData
    snpData2 = snpData.copy()

    # putting the snp prices to dictionary as well
    fullPriceDict = dict(zip(snpData['Date'], snpData['Adj Close']))

    # modify the orginal dataframe such that the the asset data only contains the price on trading days.
    snpData = alignAssetPrice(score_dict, fullPriceDict)
    # getting the change in FRC score
    frcData = md.copy()
    frcData['ScoreChange'] = frcData['score'].diff()
    frcData = frcData.fillna(0)

    # merge the two DataFrames
    compare = pd.concat([snpData, frcData], axis=1)
    compare.dropna(inplace=True)

    # only look at these 4 columns
    compare = compare.loc[:, ['Price', 'LogReturns', 'score', 'ScoreChange']]
    # getting the Moving average of the FRC score both in the truncated copy and the full copy
    mdupdate['Frc Moving Average'] = pd.rolling_mean(mdupdate['score'],window=10, min_periods=1)
    # calculating the momentum which is just delta Moving average.
    mdupdate['momentum']=mdupdate['Frc Moving Average'].diff()
    compare['Frc Moving Average'] = pd.rolling_mean(compare['score'], window=10, min_periods=1)
    compare['momentum']=compare['Frc Moving Average'].diff()

    compare.columns = ['SNP Price', 'LogReturns', 'FRC Score', 'Score Change', 'Frc Moving Average', 'momentum']
    compare.corr(method='spearman')
    momentum = compare.loc[:, ['momentum']]
    compare['Date'] = compare.index

    # putting all the dates and momentum in a dictionary, taking care of the Timestamp vs datetime juggling.
    #dateMomen = dict(zip(compare.Date, compare.momentum))
    dateMomen = {pd.Timestamp(k):v for k, v in zip(compare.Date, compare.momentum)}
    datePrice = dict(zip(compare.Date, compare['SNP Price'])) # this datePrice only has fed release dates

    snpData2 = snpData2.resample('D', fill_method = 'ffill')
    snpData2['Date'] = snpData2.index
    datePriceC = dict(zip(snpData2.Date, snpData2['Adj Close'])) # this dict includes the prices from a continuous series
    # C for continuous or whatever
    dates = []
    #for d in compare['Date'][:]:
    for d in snpData2['Date'][:]:
        # now iterating through a continuous series of dates
        dates.append(d)

    plt.plot(dates, backTesting(dates, datePriceC, dateMomen, mdupdate))
    plt.show()

if __name__ == '__main__':
    main()