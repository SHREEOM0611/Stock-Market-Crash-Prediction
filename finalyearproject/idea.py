import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import yfinance as yf
import csv
from datetime import datetime
######################
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import rcParams
from datetime import timedelta


def get_data_revised(data_original, crash_thresholds):
    datasets = []
    #     print('df.....')
    #     print(data_original)

    # data_original = pd.read_csv('stocks_list.csv',index_col='Date')

    data_original = pd.read_csv('stock_data.csv', index_col='Date')

    data_original.index = pd.to_datetime(data_original.index, format='%Y-%m-%d')
    list(data_original.columns)
    print(data_original.columns)

    data_norm = data_original['Close'] / data_original['Close'][-1]

    data_ch = data_original['Close'].pct_change()
    window = 10
    data_vol = data_original['Close'].pct_change().rolling(window).std()
    data = pd.concat([data_original['Close'], data_norm, data_ch, data_vol], axis=1).dropna()
    data.columns = ['price', 'norm', 'ch', 'vol']
    datasets.append(data)
    drawdowns = []
    crashes = []

    for df, ct in zip(datasets, [crash_thresholds]):
        pmin_pmax = (df['price'].diff(-1) > 0).astype(int).diff()
        pmax = pmin_pmax[pmin_pmax == 1]
        pmin = pmin_pmax[pmin_pmax == -1]
        # make sure drawdowns start with pmax, end with pmin:
        if pmin.index[0] < pmax.index[0]:
            pmin = pmin.drop(pmin.index[0])
        if pmin.index[-1] < pmax.index[-1]:
            pmax = pmax.drop(pmax.index[-1])
        D = (np.array(df['price'][pmin.index]) - np.array(df['price'][pmax.index])) / np.array(df['price'][pmax.index])
        d = {'Date': pmax.index, 'drawdown': D, 'd_start': pmax.index, 'd_end': pmin.index}
        df_d = pd.DataFrame(d).set_index('Date')
        df_d.index = pd.to_datetime(df_d.index, format='%Y-%m-%d')
        df_d = df_d.reindex(df.index).fillna(0)
        df_d = df_d.sort_values(by='drawdown')
        df_d['rank'] = list(range(1, df_d.shape[0] + 1))
        drawdowns.append(df_d)
        df_d = df_d.sort_values(by='Date')
        df_c = df_d[df_d['drawdown'] < ct]
        df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'rank']
        crashes.append(df_c)
    datasets_revised = []
    for i in range(len(datasets)):
        datasets_revised.append(pd.concat([datasets[i], drawdowns[i]], axis=1))
    return datasets_revised, crashes


#####################################################################################

def get_dfs_xy(dataset_revised, crashes, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)
    dfs_x, dfs_y = [], []
    for df, c in zip(dataset_revised, crashes):
        df['ch'] = df['ch'] / abs(df['ch']).mean()
        df['vol'] = df['vol'] / abs(df['vol']).mean()
        xy = {}
        for date in df.index[252:-126]:  # <--subtract 126 days in the end
            xy[date] = list([df['ch'][(date - timedelta(5)):date].mean()])
            xy[date].append(df['ch'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['ch'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['ch'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['ch'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['ch'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['ch'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['ch'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date].append(df['vol'][(date - timedelta(5)):date].mean())
            xy[date].append(df['vol'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['vol'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['vol'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['vol'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['vol'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['vol'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['vol'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c \
                                        for c in c['crash_st']]) for month in months]
        df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
        df_x = df_xy.iloc[:, :-len(months)]
        df_y = df_xy.iloc[:, -len(months):]
        dfs_x.append(df_x)
        dfs_y.append(df_y)
    return dfs_x, dfs_y


############################################################################################

def get_dfs_xy_predict(dataset_revised, crashes, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)
    dfs_x, dfs_y = [], []

    for df, c in zip(dataset_revised, crashes):

        df['ch'] = df['ch'] / abs(df['ch']).mean()
        df['vol'] = df['vol'] / abs(df['vol']).mean()
        xy = {}
        for date in df.index:  # <--subtract 126 days in the end
            xy[date] = list([df['ch'][(date - timedelta(5)):date].mean()])
            xy[date].append(df['ch'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['ch'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['ch'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['ch'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['ch'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['ch'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['ch'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date].append(df['vol'][(date - timedelta(5)):date].mean())
            xy[date].append(df['vol'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['vol'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['vol'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['vol'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['vol'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['vol'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['vol'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c for c in c['crash_st']]) for
                                   month in months if len(c['crash_st']) > 0]

        df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
        df_x = df_xy.iloc[:, :-len(months)]
        df_y = df_xy.iloc[:, -len(months):]
        dfs_x.append(df_x)
        dfs_y.append(df_y)
    return dfs_x, dfs_y


############################################################################################

def get_train_test(dfs_x, dfs_y, test_data):
    dfs_x_copy = list(dfs_x)
    dfs_y_copy = list(dfs_y)
    np_x_test = None
    np_y_test = None
    if test_data:
        df_x_test = dfs_x_copy.pop(index)
        df_y_test = dfs_y_copy.pop(index)
        np_x_test = np.array(df_x_test)
        np_y_test = np.array(df_y_test)
    np_x_train = np.concatenate(([np.array(x) for x in dfs_x_copy]))
    np_y_train = np.concatenate(([np.array(y) for y in dfs_y_copy]))
    return np_x_train, np_y_train, np_x_test, np_y_test

########################################################################################

# def split_results(self, df_combined, dfs_xy, dataset_names, test_data, y_pred_t_bin, \
#                   y_pred_tr_bin, y_train, y_test):
#     df_combined = [dfc.reindex(dfs.index) for dfc, dfs in zip(df_combined, dfs_xy)]
#     dfs_predict = []
#     n = 0
#     for df, name in zip(df_combined, dataset_names):
#         if name == test_data:
#             df['y'] = y_test
#             df['y_pred'] = y_pred_t_bin
#             dfs_predict.append(df)
#         else:
#             df['y'] = y_train[n:n+df.shape[0]]
#             df['y_pred'] = y_pred_tr_bin[n:n+df.shape[0]]
#             dfs_predict.append(df)
#             n += df.shape[0]
#     return dfs_predict

if __name__ == '__main__':
    months = [1, 3, 6]
    #     print(months)
    model_name = 'Logistic Regression'
    #     print(model_name)
    crash_threshold = -0.0936
    n_lookback = 21
    n_plot = 90
    # put the start date and current date
    START = "2010-01-01"
    TODAY = datetime.now()
    TODAY = TODAY.strftime("%Y-%m-%d")
    print(type(START))
    print(type(TODAY))

    models = ["logreg_model_1months.sav", "logreg_model_3months.sav", "logreg_model_6months.sav"]
    #     df=pd.read_csv('stocks_list.csv','r')

    # title of web_app
    st.title("Stock Market Prediction Application")

    #getting all stocks from stock list csv
    r = open('stock.csv', 'r')
    reader = csv.reader(r)

    available_stock = []

    for row in reader:
        available_stock.append(row)
    # getting all tickers from stocks
    stock_ticker = []
    for item in available_stock:
        stock_ticker.append(item[0])

    # print(stock_ticker)

    # #dataset selection
    selected_stock = st.selectbox("Select Dataset for Prediction",
                                   stock_ticker)  # selectbox will assign a value to that variable


    # #load stock data
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data


    data_load_state = st.text("Load data...")

    data = load_data(selected_stock)
    data_load_state.text("Results are...")

    # #calling raw data
    st.subheader('Stocks Details')
    data = data.sort_values(by='Date', ascending=False)
    st.write(data)
    # data=data.sort_values(by='Date', ascending=True)

    print(type(data))
    data.to_csv('stock_data.csv', index=False)



















    #####################################################3
    # # df = pd.read_csv('GOOG.csv')
    dataset_revised, crashes = get_data_revised(data, crash_threshold)
    #
    print(dataset_revised)


    # st.write("CRASH HISTORY of ")
    # df_crashes = crashes.DataFrame(data, columns=columns)
    # st.write(df_crashes)
    # print('crashes..............')
    # print(crashes)
    # print(type(crashes))


    dfs_x, dfs_y = get_dfs_xy_predict(dataset_revised, crashes, months)
    #     print(dfs_x)

    X, _, _, _ = get_train_test(dfs_x, dfs_y, test_data=None)

    y_pred_weighted_all = []
    #
    # #     ##################################################################################
    #
    # #     ########################---------- analyzing in model ---------- #########################
    for month, model in zip(months, models):
        model = pickle.load(open(model, 'rb'))
        y_pred_bin = model.predict(X).astype(int)
        y_pred_weighted = []
        for i in range(-n_plot, -1):
            y_pred_bin_ = y_pred_bin[:i]
            y_pred_weighted.append(np.dot(np.linspace(0, 1, 21) / \
                                          sum(np.linspace(0, 1, n_lookback)), y_pred_bin_[-n_lookback:]))
        y_pred_weighted.append(np.dot(np.linspace(0, 1, n_lookback) / \
                                      sum(np.linspace(0, 1, n_lookback)), y_pred_bin[-n_lookback:]))
        y_pred_weighted_all.append(y_pred_weighted)
    #
    # #     ##############################################################################################
    #
    print("y_pred_weighted_all[0]--------------------------------------")
    print(y_pred_weighted_all[0])

    print("y_pred_weighted_all[1]--------------------------------------")
    print(y_pred_weighted_all[1])

    print("y_pred_weighted_all[2]--------------------------------------")
    print(y_pred_weighted_all[2])
    #
    # ########################---------- print and plot results ---------- #########################
    df = dataset_revised[0].iloc[-n_plot:, :]
    df['y_pred_weighted_1m'] = y_pred_weighted_all[0]
    df['y_pred_weighted_3m'] = y_pred_weighted_all[1]
    df['y_pred_weighted_6m'] = y_pred_weighted_all[2]
    last_date = str(df.index[-1])[:10]
    #
    # #     last_date="2023-10-30"
    #
    dataset_name = "stock selected"
    #
    print(str(dataset_name) + ' crash prediction ' + str(model_name) + ' model as of ' \
          + str(last_date))
    print('probabilities as weighted average of binary predictions over last ' \
          + str(n_lookback) + str(' days'))
    print('* crash within 6 months: ' + str(np.round(100 \
                                                     * df['y_pred_weighted_6m'][-1], 2)) + '%')
    print('* crash within 3 months: ' + str(np.round(100 \
                                                     * df['y_pred_weighted_3m'][-1], 2)) + '%')
    print('* crash within 1 month:  ' + str(np.round(100 \
                                                     * df['y_pred_weighted_1m'][-1], 2)) + '%')



    st.subheader("Crash Predictor")
    st.write(selected_stock+" Crash prediction Logistic Regression model as of " + TODAY)
    st.write("crash within 1 months:" + str(np.round(100*df['y_pred_weighted_1m'][-1], 2)) + '%')
    st.write("crash within 3 months:" + str(np.round(100*df['y_pred_weighted_3m'][-1], 2)) + '%')
    st.write("crash within 6 months:" + str(np.round(100*df['y_pred_weighted_6m'][-1], 2)) + '%')
    #
    print("---------------df---------------------------")
    print(df)

    plt.style.use('seaborn-darkgrid')
    rcParams['figure.figsize'] = 10, 6
    rcParams.update({'font.size': 12})
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1])
    plt.subplot(gs[0])
    plt.plot(df['price'], color='blue')
    plt.ylabel(str(dataset_name) + ' index')
    plt.title(str(dataset_name) + ' crash prediction ' + str(model_name) + ' ' \
              + str(last_date))
    plt.xticks([])

    plt.subplot(gs[1])
    plt.plot(df['y_pred_weighted_6m'], color='salmon')
    plt.plot(df['y_pred_weighted_3m'], color='red')
    plt.plot(df['y_pred_weighted_1m'], color='brown')
    plt.ylabel('crash probability')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45)
    plt.legend(['crash in 6 months', 'crash in 3 months', 'crash in 1 month'])
    plt.show()























import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import yfinance as yf
import csv
from datetime import datetime
######################
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import rcParams
from datetime import timedelta


def get_data_revised(data_original, crash_thresholds):
    datasets = []
    #     print('df.....')
    #     print(data_original)

    # data_original = pd.read_csv('stocks_list.csv',index_col='Date')

    data_original = pd.read_csv('stock_data.csv', index_col='Date')

    data_original.index = pd.to_datetime(data_original.index, format='%Y-%m-%d')
    list(data_original.columns)
    print(data_original.columns)

    data_norm = data_original['Close'] / data_original['Close'][-1]

    data_ch = data_original['Close'].pct_change()
    window = 10
    data_vol = data_original['Close'].pct_change().rolling(window).std()
    data = pd.concat([data_original['Close'], data_norm, data_ch, data_vol], axis=1).dropna()
    data.columns = ['price', 'norm', 'ch', 'vol']
    datasets.append(data)
    drawdowns = []
    crashes = []

    for df, ct in zip(datasets, [crash_thresholds]):
        pmin_pmax = (df['price'].diff(-1) > 0).astype(int).diff()
        pmax = pmin_pmax[pmin_pmax == 1]
        pmin = pmin_pmax[pmin_pmax == -1]
        # make sure drawdowns start with pmax, end with pmin:
        if pmin.index[0] < pmax.index[0]:
            pmin = pmin.drop(pmin.index[0])
        if pmin.index[-1] < pmax.index[-1]:
            pmax = pmax.drop(pmax.index[-1])
        D = (np.array(df['price'][pmin.index]) - np.array(df['price'][pmax.index])) / np.array(df['price'][pmax.index])
        d = {'Date': pmax.index, 'drawdown': D, 'd_start': pmax.index, 'd_end': pmin.index}
        df_d = pd.DataFrame(d).set_index('Date')
        df_d.index = pd.to_datetime(df_d.index, format='%Y-%m-%d')
        df_d = df_d.reindex(df.index).fillna(0)
        df_d = df_d.sort_values(by='drawdown')
        df_d['rank'] = list(range(1, df_d.shape[0] + 1))
        drawdowns.append(df_d)
        df_d = df_d.sort_values(by='Date')
        df_c = df_d[df_d['drawdown'] < ct]
        df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'rank']
        crashes.append(df_c)
    datasets_revised = []
    for i in range(len(datasets)):
        datasets_revised.append(pd.concat([datasets[i], drawdowns[i]], axis=1))
    return datasets_revised, crashes


#####################################################################################

def get_dfs_xy(dataset_revised, crashes, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)
    dfs_x, dfs_y = [], []
    for df, c in zip(dataset_revised, crashes):
        df['ch'] = df['ch'] / abs(df['ch']).mean()
        df['vol'] = df['vol'] / abs(df['vol']).mean()
        xy = {}
        for date in df.index[252:-126]:  # <--subtract 126 days in the end
            xy[date] = list([df['ch'][(date - timedelta(5)):date].mean()])
            xy[date].append(df['ch'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['ch'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['ch'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['ch'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['ch'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['ch'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['ch'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date].append(df['vol'][(date - timedelta(5)):date].mean())
            xy[date].append(df['vol'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['vol'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['vol'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['vol'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['vol'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['vol'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['vol'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c \
                                        for c in c['crash_st']]) for month in months]
        df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
        df_x = df_xy.iloc[:, :-len(months)]
        df_y = df_xy.iloc[:, -len(months):]
        dfs_x.append(df_x)
        dfs_y.append(df_y)
    return dfs_x, dfs_y


############################################################################################

def get_dfs_xy_predict(dataset_revised, crashes, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)
    dfs_x, dfs_y = [], []

    for df, c in zip(dataset_revised, crashes):

        df['ch'] = df['ch'] / abs(df['ch']).mean()
        df['vol'] = df['vol'] / abs(df['vol']).mean()
        xy = {}
        for date in df.index:  # <--subtract 126 days in the end
            xy[date] = list([df['ch'][(date - timedelta(5)):date].mean()])
            xy[date].append(df['ch'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['ch'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['ch'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['ch'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['ch'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['ch'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['ch'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date].append(df['vol'][(date - timedelta(5)):date].mean())
            xy[date].append(df['vol'][(date - timedelta(10)):(date - timedelta(5))].mean())
            xy[date].append(df['vol'][(date - timedelta(15)):(date - timedelta(10))].mean())
            xy[date].append(df['vol'][(date - timedelta(21)):(date - timedelta(15))].mean())
            xy[date].append(df['vol'][(date - timedelta(42)):(date - timedelta(21))].mean())
            xy[date].append(df['vol'][(date - timedelta(63)):(date - timedelta(42))].mean())
            xy[date].append(df['vol'][(date - timedelta(126)):(date - timedelta(63))].mean())
            xy[date].append(df['vol'][(date - timedelta(252)):(date - timedelta(126))].mean())
            xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c for c in c['crash_st']]) for
                                   month in months if len(c['crash_st']) > 0]

        df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
        df_x = df_xy.iloc[:, :-len(months)]
        df_y = df_xy.iloc[:, -len(months):]
        dfs_x.append(df_x)
        dfs_y.append(df_y)
    return dfs_x, dfs_y


############################################################################################

def get_train_test(dfs_x, dfs_y, test_data):
    dfs_x_copy = list(dfs_x)
    dfs_y_copy = list(dfs_y)
    np_x_test = None
    np_y_test = None
    if test_data:
        df_x_test = dfs_x_copy.pop(index)
        df_y_test = dfs_y_copy.pop(index)
        np_x_test = np.array(df_x_test)
        np_y_test = np.array(df_y_test)
    np_x_train = np.concatenate(([np.array(x) for x in dfs_x_copy]))
    np_y_train = np.concatenate(([np.array(y) for y in dfs_y_copy]))
    return np_x_train, np_y_train, np_x_test, np_y_test

########################################################################################

# def split_results(self, df_combined, dfs_xy, dataset_names, test_data, y_pred_t_bin, \
#                   y_pred_tr_bin, y_train, y_test):
#     df_combined = [dfc.reindex(dfs.index) for dfc, dfs in zip(df_combined, dfs_xy)]
#     dfs_predict = []
#     n = 0
#     for df, name in zip(df_combined, dataset_names):
#         if name == test_data:
#             df['y'] = y_test
#             df['y_pred'] = y_pred_t_bin
#             dfs_predict.append(df)
#         else:
#             df['y'] = y_train[n:n+df.shape[0]]
#             df['y_pred'] = y_pred_tr_bin[n:n+df.shape[0]]
#             dfs_predict.append(df)
#             n += df.shape[0]
#     return dfs_predict

if __name__ == '__main__':
    months = [1, 3, 6]
    #     print(months)
    model_name = 'Logistic Regression'
    #     print(model_name)
    crash_threshold = -0.0936
    n_lookback = 21
    n_plot = 90
    # put the start date and current date
    START = "2010-01-01"
    TODAY = datetime.now()
    TODAY = TODAY.strftime("%Y-%m-%d")
    print(type(START))
    print(type(TODAY))

    models = ["logreg_model_1months.sav", "logreg_model_3months.sav", "logreg_model_6months.sav"]
    #     df=pd.read_csv('stocks_list.csv','r')

    # title of web_app
    st.title("Stock Market Prediction Application")

    #getting all stocks from stock list csv
    r = open('stock.csv', 'r')
    reader = csv.reader(r)

    available_stock = []

    for row in reader:
        available_stock.append(row)
    # getting all tickers from stocks
    stock_ticker = []
    for item in available_stock:
        stock_ticker.append(item[0])

    # print(stock_ticker)

    # #dataset selection
    selected_stock = st.selectbox("Select Dataset for Prediction",
                                   stock_ticker)  # selectbox will assign a value to that variable


    # #load stock data
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data


    data_load_state = st.text("Load data...")

    data = load_data(selected_stock)
    data_load_state.text("Results are...")

    # #calling raw data
    st.subheader('Stocks Details')
    stock_price = data.sort_values(by='Date', ascending=False)
    st.write(stock_price)
    # data=data.sort_values(by='Date', ascending=True)

    print(type(data))
    data.to_csv('stock_data.csv', index=False)



















    #####################################################3
    # # df = pd.read_csv('GOOG.csv')
    dataset_revised, crashes = get_data_revised(data, crash_threshold)
    #
    print(dataset_revised)


    st.write("CRASH HISTORY of ")
    st.write(crashes)
    print('crashes..............')
    print(crashes)

    dfs_x, dfs_y = get_dfs_xy_predict(dataset_revised, crashes, months)
    #     print(dfs_x)

    X, _, _, _ = get_train_test(dfs_x, dfs_y, test_data=None)

    y_pred_weighted_all = []
    #
    # #     ##################################################################################
    #
    # #     ########################---------- analyzing in model ---------- #########################
    for month, model in zip(months, models):
        model = pickle.load(open(model, 'rb'))
        y_pred_bin = model.predict(X).astype(int)
        y_pred_weighted = []
        for i in range(-n_plot, -1):
            y_pred_bin_ = y_pred_bin[:i]
            y_pred_weighted.append(np.dot(np.linspace(0, 1, 21) / \
                                          sum(np.linspace(0, 1, n_lookback)), y_pred_bin_[-n_lookback:]))
        y_pred_weighted.append(np.dot(np.linspace(0, 1, n_lookback) / \
                                      sum(np.linspace(0, 1, n_lookback)), y_pred_bin[-n_lookback:]))
        y_pred_weighted_all.append(y_pred_weighted)
    #
    # #     ##############################################################################################
    #
    print("y_pred_weighted_all[0]--------------------------------------")
    print(y_pred_weighted_all[0])

    print("y_pred_weighted_all[1]--------------------------------------")
    print(y_pred_weighted_all[1])

    print("y_pred_weighted_all[2]--------------------------------------")
    print(y_pred_weighted_all[2])
    #
    # ########################---------- print and plot results ---------- #########################
    df = dataset_revised[0].iloc[-n_plot:, :]
    df['y_pred_weighted_1m'] = y_pred_weighted_all[0]
    df['y_pred_weighted_3m'] = y_pred_weighted_all[1]
    df['y_pred_weighted_6m'] = y_pred_weighted_all[2]
    last_date = str(df.index[-1])[:10]
    #
    # #     last_date="2023-10-30"
    #
    dataset_name = "stock selected"
    #
    print(str(dataset_name) + ' crash prediction ' + str(model_name) + ' model as of ' \
          + str(last_date))
    print('probabilities as weighted average of binary predictions over last ' \
          + str(n_lookback) + str(' days'))
    print('* crash within 6 months: ' + str(np.round(100 \
                                                     * df['y_pred_weighted_6m'][-1], 2)) + '%')
    print('* crash within 3 months: ' + str(np.round(100 \
                                                     * df['y_pred_weighted_3m'][-1], 2)) + '%')
    print('* crash within 1 month:  ' + str(np.round(100 \
                                                     * df['y_pred_weighted_1m'][-1], 2)) + '%')



    st.write("Crash Predictor")
    st.write("stock selected crash prediction Logistic Regression model as of" + TODAY)
    st.write("crash within 1 months:" + str(np.round(100*df['y_pred_weighted_1m'][-1], 2)) + '%')
    st.write("crash within 3 months:" + str(np.round(100*df['y_pred_weighted_3m'][-1], 2)) + '%')
    st.write("crash within 6 months:" + str(np.round(100*df['y_pred_weighted_6m'][-1], 2)) + '%')
    #
    print("---------------df---------------------------")
    print(df)

    plt.style.use('seaborn-darkgrid')
    rcParams['figure.figsize'] = 10, 6
    rcParams.update({'font.size': 12})
    gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1])
    plt.subplot(gs[0])
    plt.plot(df['price'], color='blue')
    plt.ylabel(str(dataset_name) + ' index')
    plt.title(str(dataset_name) + ' crash prediction ' + str(model_name) + ' ' \
              + str(last_date))
    plt.xticks([])

    plt.subplot(gs[1])
    plt.plot(df['y_pred_weighted_6m'], color='salmon')
    plt.plot(df['y_pred_weighted_3m'], color='red')
    plt.plot(df['y_pred_weighted_1m'], color='brown')
    plt.ylabel('crash probability')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45)
    plt.legend(['crash in 6 months', 'crash in 3 months', 'crash in 1 month'])
    plt.show()
