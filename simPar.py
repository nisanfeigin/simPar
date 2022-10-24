from time import sleep
import urllib.request, urllib.parse, urllib.error
import json
import numpy as np
import pandas as pd
from numpy import *
import yahoo_fin
from yahoo_fin.stock_info import *
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import statistics
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets, linear_model
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from multiprocessing import Manager, Process

current_time = datetime.datetime.now()

def lr(predDict, train_features, test_features, train_labels, test_labels, to_pred):
        regr = linear_model.LinearRegression()
        regr.fit(train_features, train_labels)
        predictions = regr.predict(test_features)
        errors = abs(predictions - test_labels)
        lr_MAE = round(np.mean(errors), 3) * 100
        lr_pred = ((regr.predict([to_pred]))[0] - 1) * 100
        predDict[lr_pred] = lr_MAE
def rf(predDict, train_features, test_features, train_labels, test_labels, to_pred):
        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(train_features, train_labels)
        predictions = rf.predict(test_features)
        errors = abs(predictions - test_labels)
        rf_MAE = round(np.mean(errors), 3) * 100
        rf_pred = ((rf.predict([to_pred]))[0] - 1) * 100
        predDict[rf_pred] = rf_MAE
def knn(predDict, train_features, test_features, train_labels, test_labels, to_pred):
        knn_model = KNeighborsRegressor(n_neighbors=3)
        knn_model.fit(train_features, train_labels)
        predictions = knn_model.predict(test_features)
        errors = abs(predictions - test_labels)
        knn_MAE = round(np.mean(errors), 3) * 100
        knn_pred = ((knn_model.predict([to_pred]))[0] - 1) * 100
        predDict[knn_pred] = knn_MAE
def xg(predDict, train_features, test_features, train_labels, test_labels, to_pred):
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        xgb_model.fit(train_features, train_labels)
        predictions = xgb_model.predict(test_features)
        errors = abs(predictions - test_labels)
        xgb_MAE = round(np.mean(errors), 3) * 100
        xgb_pred = ((xgb_model.predict([to_pred]))[0] - 1) * 100
        predDict[xgb_pred] = xgb_MAE

def task(lod, c):
        for i in c:
                print(i)
                if i % int == 0:
                        ld = []
                        ld.append(i)
                        for name in ["msft", "aapl", "adp", "wm", "fico", "unh", "hsy", "mco", "asml", "sci", "TOL",
                                     "TSN", "BLK", "MMC", "DCI", "WSM", "AN", "CSX", "UNP", "ODFL", "RL", "SHOO",
                                     "GWW", "spy"]:
                                features = df_features
                                to_pred = df_features.iloc[i + 260].to_numpy()

                                N = 40
                                n = 892 - i + int

                                features = features.iloc[N:, :]
                                features.drop(features.tail(n).index, inplace=True)
                                labels6 = df_r.iloc[N:, :]
                                labels6 = labels6[str(name) + col]
                                labels6.drop(labels6.tail(n).index, inplace=True)
                                labels6 = np.array(labels6)
                                features = np.array(features)

                                train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                                            labels6,
                                                                                                            test_size=0.2,
                                                                                                            random_state=42)
                                predDict = {}

                                lr(predDict, train_features, test_features, train_labels, test_labels, to_pred)
                                rf(predDict, train_features, test_features, train_labels, test_labels, to_pred)
                                knn(predDict, train_features, test_features, train_labels, test_labels, to_pred)
                                xg(predDict, train_features, test_features, train_labels, test_labels, to_pred)

                                sortPredDict = sorted(predDict.items(), key=lambda x: x[1])
                                avPred = (sortPredDict[0][0] + sortPredDict[1][0]) / 2
                                avEr = (sortPredDict[0][1] + sortPredDict[1][1]) / 2
                                Dict = {}
                                Dict["pred_av"] = avPred
                                Dict["95_AvAbove"] = avPred - avEr * 2
                                Dict["70_AvAbove"] = avPred - avEr
                                Dict["95_AvBelow"] = avPred + avEr * 2
                                Dict["70_AvBelow"] = avPred + avEr
                                Dict["symbol"] = name

                                ld.append(Dict)
                        lod.append(list(ld))
                else:
                        try:
                            ld[0] = i
                            lod.append(ld)
                        except:
                            ld = [i]

df_features = pd.read_excel("features.xlsx")
df_r = pd.read_excel("r.xlsx")

#for int, col in ((1, "ret1w"), (4, "ret1m"), (13, "ret3m"), (26, "ret6m")):
# ld = []
int = 1
col = "ret1w"
rng = [x for x in range(0, 892)]

chunks = np.array_split(rng, 11)
if __name__ == "__main__":
    with Manager() as manager:
        lod = manager.list()  # <-- can be shared between processes.
        #ld = manager.list()  # <-- can be shared between processes.
        processes = []
        for c in chunks:
            p = Process(target=task, args=(lod,c))  # Passing the list
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print(type(lod))
        lod = list(lod)
        print(type(lod))

        df_lod = pd.DataFrame(lod)
        df_lod.round(3)
        df_lod.to_excel("lod" + col[-2:] + ".xlsx")
        print("Time now at greenwich meridian is:", current_time)
        current_time = datetime.datetime.now()
        print("Time now at greenwich meridian is:", current_time)


