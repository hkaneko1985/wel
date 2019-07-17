# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""
# wRMSD-based AD considering ensemble learning (WEL)

import numpy as np
import pandas as pd


def calc_y_and_weights(submodels, selected_x_variable_numbers, x_train, y_train, x_test, k_in_knn_in_wrmsd, multiplier_of_inverse_wrmsed):
    
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    estimated_y_test_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのテストデータの y の推定結果を追加
    wrmsd_test = np.zeros([x_test.shape[0], len(submodels)])
    for submodel_number in range(len(submodels)):
        # 説明変数の選択
        selected_autoscaled_x_test = autoscaled_x_test.iloc[:, selected_x_variable_numbers[submodel_number]]

        # テストデータの y の推定
        estimated_y_test = pd.DataFrame(
            submodels[submodel_number].predict(selected_autoscaled_x_test))  # テストデータの y の値を推定し、Pandas の DataFrame 型に変換
        estimated_y_test = estimated_y_test * y_train.std() + y_train.mean()  # スケールをもとに戻します
        estimated_y_test_all = pd.concat([estimated_y_test_all, estimated_y_test], axis=1)
        
        selected_autoscaled_x_train = autoscaled_x_train.iloc[:, selected_x_variable_numbers[submodel_number]]
        # wRMSE の計算
        for test_sample_number in range(x_test.shape[0]):
#            print(submodel_number+1, test_sample_number+1)
            autoscaled_query_sample = selected_autoscaled_x_test.iloc[test_sample_number, :]
            d = abs(selected_autoscaled_x_train - autoscaled_query_sample) / 5
            d[(d > 1)] = 1
            d = d.sum(axis=1)
            w = 1 / (d + 0.5)
            w_y = pd.concat([w, y_train], axis=1)
            w_y.sort_values(0, ascending=False, inplace=True)
            wrmsd = (w_y.iloc[:, 0] ** 2) * ((w_y.iloc[:, 1] - estimated_y_test.iloc[test_sample_number, 0]) ** 2)
            wrmsd = (wrmsd.iloc[:k_in_knn_in_wrmsd].sum() / (w_y.iloc[:k_in_knn_in_wrmsd, 0] ** 2).sum()) ** 0.5
            wrmsd_test[test_sample_number, submodel_number] = wrmsd
    
    # テストデータの推定値の重み付き平均値
    weights = (1 / wrmsd_test) ** multiplier_of_inverse_wrmsed
    sum_of_weights = weights.sum(axis=1)
    weights = weights / np.tile(sum_of_weights.reshape([len(sum_of_weights), 1]), (1, weights.shape[1]))
    return estimated_y_test_all, weights
