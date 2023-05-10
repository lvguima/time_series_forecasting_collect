import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as smapi
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy.dialects.mssql.information_schema import columns


datafile = 'C:\\Users\\lvgui\\Desktop\\coding\\new_data.csv'
sum_rmse = []
sum_mae = []
sum_mape = []
pre = []
for i in range(11):
# Load data
    data = pd.read_csv(datafile, header=None)[i]

    # Split data into training and testing sets
    train_data = data.iloc[:-50]
    test_data = data.iloc[-50:].reset_index(drop=True)

    #find best factor
    model = auto_arima(train_data, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    best_order = model.order
    # Fit ARIMA model
    model = smapi.tsa.arima.ARIMA(train_data, order=best_order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=50)
    predictions = pd.DataFrame(predictions)
    # Plot results
    # plt.plot(train_data)
    # plt.plot(predictions, color='red')
    # plt.show()# There is no problem with this code

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    # 两列数据分别为y_true和y_pred
    y_true = test_data
    y_pred = predictions

    print("RMSE:", rmse(y_true, y_pred))
    print("MAPE:", mape(np.array(y_true), np.array(y_pred)))
    print("MAE:", mae(y_true, y_pred))


    sum_rmse.append(rmse(y_true, y_pred))
    sum_mae.append(mae(y_true, y_pred))
    sum_mape.append(mape(np.array(y_true), np.array(y_pred)))
    pre.append(predictions)
pre_df = pd.concat(pre, axis=1).reset_index(drop =True)
pre_df.to_csv('arimapre.csv',header=None, index=None)
print('final_rmse = {:5.6f}' 'final_mae = {:5.6f}' 'final_mape = {:5.6f}' .format(np.mean(np.array(sum_rmse)), np.mean(np.array(sum_mae)), np.mean(np.array(sum_mape))))




from statsmodels.tsa.arima_model import ARIMA
# 导入ACF和PACF自相关”的库
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from pmdarima.arima import auto_arima
# import pandas as  pd
# datafile = 'C:\\Users\\lvgui\\Desktop\\coding\\new_data.csv'
# # Load data
# ts= pd.read_csv(datafile, header=None)[0]
# import matplotlib.pyplot as plt
#
# # 绘制时间序列
# plt.plot(ts)
#
# # 绘制ACF
# plot_acf(ts)
#
# # 绘制PACF
# plot_pacf(ts)
#
# plt.show()
#
#
# # 寻找最优pdq的值
# model = auto_arima(ts, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
# print(model.order)

