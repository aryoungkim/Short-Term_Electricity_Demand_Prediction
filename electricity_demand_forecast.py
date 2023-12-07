import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tabulate import tabulate
from general2 import (data_preprocess, data_sequencing, confirm_result, data_visual, model_loss_visual, act_pred_visual, time_preprocess)

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, RNN, GRU, LSTM, Dropout, GRUCell
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"]="0"

f_path = "C:/windows/Fonts/malgun.ttf"
fm.FontProperties(fname=f_path).get_name()
plt.rc('font', family='Malgun Gothic')


def LSTM_model(X_train, Y_train, X_test, feature_cnt, periods, epoch, batch):

    ### LSTM architecture
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=256,
                        return_sequences=True,
                        input_shape=(X_train.shape[1], feature_cnt),
                        activation='tanh'))
    lstm_model.add(LSTM(units=128, return_sequences=True, activation='tanh'))
    lstm_model.add(LSTM(units=64, activation='tanh'))
    lstm_model.add(Dense(units=periods))

    # Compiling
    lstm_model.compile(optimizer=Adam(lr=0.01), loss='mse')

    # Training dataset fitting
    hist = lstm_model.fit(X_train, Y_train, epochs=epoch, batch_size=batch, validation_split=0.2, verbose=1)

    # X_test predict
    lstm_prediction = lstm_model.predict(X_test)

    return lstm_model, lstm_prediction, hist

def LSTM_run(feature_sc, X_train, Y_train, X_test, Y_test, ts_test, args, feature_cnt):

    ######### LSTM model ################
    lstm_model, lstm_prediction, hist = LSTM_model(X_train, Y_train, X_test, feature_cnt, args['pred_p'], args['epoch'], args['batch'])

    # create empty table with feature_cnt fields
    lstm_pred = np.zeros(shape=(lstm_prediction.shape[0], feature_cnt + 1))
    y_test = np.zeros(shape=(Y_test.shape[0], feature_cnt + 1))

    lstm_pred[:, -1] = lstm_prediction[:, 0]
    y_test[:, -1] = Y_test[:, 0]

    y_pred = feature_sc.inverse_transform(lstm_pred)[:, -1]
    y_test = feature_sc.inverse_transform(y_test)[:, -1]

    lstm_predict_result = pd.DataFrame()
    lstm_predict_result['date'] = pd.to_datetime(ts_test[-len(y_test):].index.values)
    lstm_predict_result['actual'] = y_test
    lstm_predict_result['predict'] = y_pred
    lstm_predict_result.set_index('date', inplace=True)

    score = confirm_result(y_test, y_pred)

    return lstm_model, lstm_predict_result, hist, score

def GRU_model(X_train, Y_train, X_test, feature_cnt, periods, epoch, batch):
    ### The GRU architecture2
    gru_model = Sequential()
    gru_model.add(GRU(units=256,
                      return_sequences=True,
                      input_shape=(X_train.shape[1], feature_cnt),
                      activation='tanh'))
    gru_model.add(GRU(units=128, return_sequences=True, activation='tanh'))
    gru_model.add(GRU(units=64, activation='tanh'))
    gru_model.add(Dense(units=periods))

    # Compiling
    gru_model.compile(optimizer=Adam(lr=0.01), loss='mse')

    # Training dataset fitting
    hist = gru_model.fit(X_train, Y_train, epochs=epoch, batch_size=batch, validation_split=0.2, verbose=1)

    # X_test predict
    gru_prediction = gru_model.predict(X_test)

    return gru_model, gru_prediction, hist


def GRU_run(feature_sc, X_train, Y_train, X_test, Y_test, ts_test, args, feature_cnt):

    ########### GRU model ################
    gru_model, gru_prediction, hist = GRU_model(X_train, Y_train, X_test, feature_cnt, args['pred_p'], args['epoch'], args['batch'])

    # create empty table with feature_cnt fields
    gru_pred = np.zeros(shape=(gru_prediction.shape[0], feature_cnt + 1))
    y_test = np.zeros(shape=(Y_test.shape[0], feature_cnt + 1))

    gru_pred[:, -1] = gru_prediction[:, 0]
    y_test[:, -1] = Y_test[:, 0]

    y_pred = feature_sc.inverse_transform(gru_pred)[:, -1]
    y_test = feature_sc.inverse_transform(y_test)[:, -1]

    gru_predict_result = pd.DataFrame()
    gru_predict_result['date'] = pd.to_datetime(ts_test[-len(y_test):].index.values)
    gru_predict_result['actual'] = y_test
    gru_predict_result['predict'] = y_pred
    gru_predict_result.set_index('date', inplace=True)

    score = confirm_result(y_test, y_pred)

    return gru_model, gru_predict_result, hist, score


def train_test(fname, args, data_df, summer_predict=False):

    npp_name = fname.split(".")[0]
    initial_time = time.time()
    score_df = pd.DataFrame(columns=['MAE', 'RMSE', 'R2', 'MAPE'])
    seed = 42

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with tf.device("/gpu:0"):
        case_df = data_df.copy()
        allset_plt = data_visual(case_df[args['input_col']])
        allset_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_input_visualize.png"))

        ## Pre-processing
        feature_sc, ts_train, org_test, ts_test = data_preprocess(case_df, args['input_col'], args['target_col'], args['split_ratio'], args['hist_p'], summer_predict)
        allset_scaled_plt = data_visual(pd.concat([ts_train.iloc[:, :-1], ts_test.iloc[:, :-1]], axis=0))
        allset_scaled_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_input_visualize(scaled).png"))

        ## Sequential Dataset
        feature_cnt = ts_train[args['input_col']].shape[1]
        feature_cnt_output = ts_train[args['target_col']].shape[1]
        args['feature_cnt_input'] = feature_cnt
        args['feature_cnt_output'] = feature_cnt_output
        X_train, Y_train, X_test, Y_test = data_sequencing(ts_train.values, ts_test.values, args['hist_p'], args['pred_p'], feature_cnt)

        if args['model'] == 'lstm':
            trained_model, predict_result, hist, score = LSTM_run(feature_sc, X_train, Y_train, X_test, Y_test, ts_test, args, feature_cnt)
        elif args['model'] == 'gru':
            trained_model, predict_result, hist, score = GRU_run(feature_sc, X_train, Y_train, X_test, Y_test, ts_test, args, feature_cnt)
        elif args['model'] == 'seq2seq':
            trained_model, predict_result, hist, score = Seq2Seq_run(feature_sc, X_train, Y_train, X_test, Y_test, ts_test, args, feature_cnt)

        if summer_predict:
            ## Result Visualize
            loss_plt = model_loss_visual(hist, args['epoch'], args['batch'], args['model'])
            loss_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_summer_loss.png"))
            loss_plt.show()

            predict_result.to_csv(os.path.join(args['save_path'], f"{args['model']}_summer_predict_result.csv"), sep=",")
            predict_result_plt = act_pred_visual(f"{args['model'].upper()} prediction result for summertime electricity demand", predict_result, score)
            predict_result_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_summer_predict_result.png"))
            predict_result_plt.show()

            predict_result_expand_plt = act_pred_visual(f"{args['model'].upper()} prediction result for the 1 months of the summertime electricity demand", predict_result.head(528), score)
            predict_result_expand_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_summer_predict_expand_result.png"))
            predict_result_expand_plt.show()

            ## Save Result Score(MAE, RMSE, R2)
            score_df.loc[fname.split('.')[0]] = score
            score_df.to_csv(os.path.join(args['save_path'], f"{args['model']}_summer_score.csv"), sep=",")
        else:
            ## Result Visualize
            loss_plt = model_loss_visual(hist, args['epoch'], args['batch'], args['model'])
            loss_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_loss.png"))
            loss_plt.show()

            predict_result.to_csv(os.path.join(args['save_path'], f"{args['model']}_testset_predict_result.csv"), sep=",")
            predict_result_plt = act_pred_visual(f"{args['model'].upper()} prediction result for testset electricity demand ", predict_result, score)
            predict_result_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_testset_predict_result.png"))
            predict_result_plt.show()

            predict_result_expand_plt = act_pred_visual(f"{args['model'].upper()} prediction result for the 1 months of the testset electricity demand", predict_result.head(528), score)
            predict_result_expand_plt.savefig(os.path.join(args['save_path'], f"{args['model']}_testset_predict_expand_result.png"))
            predict_result_expand_plt.show()

            ## Save Result Score(MAE, RMSE, R2)
            score_df.loc[fname.split('.')[0]] = score
            score_df.to_csv(os.path.join(args['save_path'], f"{args['model']}_testset_score.csv"), sep=",")

        time_elapsed = time.time() - initial_time
        print(f"The whole process runs for {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {((time_elapsed % 3600) % 60) % 60:0f}s")

    return predict_result, score_df



def run(in_col, file_name, model, summer_predict=False):
    # building detection and predict result save
    split_ratio = 0.9
    max_day = 24 * 365 * 3          # 3 years
    hist_p = 9                      # history period
    pred_p = 3                      # future predict period
    epoch = 32
    batch = 64
    #model = 'gru'

    filePath = 'D:/Dataset/itct2023_exp/'
    savePath = 'D:/Dataset/itct2023_exp/exp/'

    file = os.path.join(filePath, file_name)

    try:
        read_df = pd.read_csv(file, header=0, encoding='CP949')
    except FileNotFoundError as e:
        print(str(e))

    save_folder_name = f'{hist_p}history_{pred_p}period_{len(in_col)}f_result256'
    save_path = os.path.join(savePath, save_folder_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    data_df = read_df.copy()
    data_df.set_index("time", inplace=True)
    data_df.index.name = 'time'
    data_df = data_df.query('"2019-01-01 00:00" < time')
    data_df['predict'] = np.nan

    target_col = ['power_demand']

    args = {'split_ratio':split_ratio, 'save_path': save_path, 'hist_p': hist_p, 'pred_p': pred_p, 'model': model,
            'epoch': epoch, 'batch': batch, 'input_col':in_col, 'target_col':target_col}

    future_predict, score_df = train_test(file_name, args, data_df, summer_predict)

    return score_df, save_path


if __name__ == '__main__':
    file_name = 'hour_combined_data.csv'
    input_col = ['power_demand']

    lstm_score_df, _ = run(input_col, file_name, 'lstm', summer_predict=False)
    lstm_summer_score_df, _ = run(input_col, file_name, 'lstm', summer_predict=True)

    gru_score_df, _ = run(input_col, file_name, 'gru', summer_predict=False)
    gru_summer_score_df, _ = run(input_col, file_name, 'gru', summer_predict=True)

    ## Seq2Seq model is private
    seq2seq_score_df, _ = run(input_col, file_name, 'seq2seq', summer_predict=False)
    seq2seq_summer_score_df, save_path = run(input_col, file_name, 'seq2seq', summer_predict=True)

    score_ct = pd.concat([lstm_score_df, gru_score_df, seq2seq_score_df, lstm_summer_score_df, gru_summer_score_df, seq2seq_summer_score_df], axis=0)
    score_ct = score_ct.transpose()
    score_ct.columns = ['lstm_score', 'gru_score', 'seq2seq_score', 'lstm_summer_score', 'gru_summer_score',
                        'seq2seq_summer_score']
    score_ct.to_csv(save_path + f"/score_all.csv", encoding='CP949')
