# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wel
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression

number_of_training_data = 300  # トレーニングデータの数
number_of_submodels = 100  # サブモデルの数
rate_of_selected_x_variables = 0.5  # 各サブデータセットで選択される説明変数の数の割合。0 より大きく 1 未満
k_in_knn_in_wrmsd = 5
multiplier_of_inverse_wrmsed = 2.5
#candidates_of_multiplier_of_inverse_wrmsed = np.arange(0.5, 10.5, 0.5)

do_autoscaling = True  # True or False
threshold_of_rate_of_same_value = 0.80
fold_number = 5
max_pls_component_number = 30

# load data set
raw_data_with_y = pd.read_csv('descriptors_with_logS.csv', encoding='SHIFT-JIS', index_col=0)
raw_data_with_y = raw_data_with_y.loc[:, raw_data_with_y.mean().index]  # 平均を計算できる変数だけ選択
# raw_data_with_y = raw_data_with_y.loc[raw_data_with_y.mean(axis=1).index,:] #平均を計算できるサンプルだけ選択
raw_data_with_y = raw_data_with_y.replace(np.inf, np.nan).fillna(np.nan)  # infをnanに置き換えておく
raw_data_with_y = raw_data_with_y.dropna(axis=1)  # nanのある変数を削除
# raw_data_with_y = raw_data_with_y.dropna() #nanのあるサンプルを削除
raw_x_train, raw_x_test, y_train, y_test = train_test_split(raw_data_with_y.iloc[:, 1:], raw_data_with_y.iloc[:, 0], test_size=raw_data_with_y.shape[0] - number_of_training_data, random_state=0)

# delete descriptors with high rate of the same values
rate_of_same_value = list()
num = 0
for X_variable_name in raw_x_train.columns:
    num += 1
    print('{0} / {1}'.format(num, raw_x_train.shape[1]))
    same_value_number = raw_x_train[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / raw_x_train.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

"""
# delete descriptors with zero variance
deleting_variable_numbers = np.where( raw_Xtrain.var() == 0 )
"""

if len(deleting_variable_numbers[0]) == 0:
    x_train = raw_x_train.copy()
    x_test = raw_x_test.copy()
else:
    x_train = raw_x_train.drop(raw_x_train.columns[deleting_variable_numbers], axis=1)
    x_test = raw_x_test.drop(raw_x_test.columns[deleting_variable_numbers], axis=1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))

print('# of X-variables: {0}'.format(x_train.shape[1]))

# オートスケーリング
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)

number_of_x_variables = int(np.ceil(x_train.shape[1] * rate_of_selected_x_variables))
print('各サブデータセットの説明変数の数 :', number_of_x_variables)
estimated_y_train_all = pd.DataFrame()  # 空の DataFrame 型を作成し、ここにサブモデルごとのトレーニングデータの y の推定結果を追加
selected_x_variable_numbers = []  # 空の list 型の変数を作成し、ここに各サブデータセットの説明変数の番号を追加
submodels = []  # 空の list 型の変数を作成し、ここに構築済みの各サブモデルを追加
for submodel_number in range(number_of_submodels):
    print(submodel_number + 1, '/', number_of_submodels)  # 進捗状況の表示
    # 説明変数の選択
    # 0 から 1 までの間に一様に分布する乱数を説明変数の数だけ生成して、その乱数値が小さい順に説明変数を選択
    random_x_variables = np.random.rand(x_train.shape[1])
    selected_x_variable_numbers_tmp = random_x_variables.argsort()[:number_of_x_variables]
    selected_autoscaled_x_train = autoscaled_x_train.iloc[:, selected_x_variable_numbers_tmp]
    selected_x_variable_numbers.append(selected_x_variable_numbers_tmp)
    
    pls_components = np.arange(1, min(np.linalg.matrix_rank(selected_autoscaled_x_train) + 1, max_pls_component_number + 1), 1)
    r2all = list()
    r2cvall = list()
    for pls_component in pls_components:
        pls_model_in_cv = PLSRegression(n_components=pls_component)
        pls_model_in_cv.fit(selected_autoscaled_x_train, autoscaled_y_train)
        calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(selected_autoscaled_x_train))
        estimated_y_in_cv = np.ndarray.flatten(
            model_selection.cross_val_predict(pls_model_in_cv, selected_autoscaled_x_train, autoscaled_y_train, cv=fold_number))
        if do_autoscaling:
            calculated_y_in_cv = calculated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
        r2all.append(float(1 - sum((y_train - calculated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
        r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
    optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
    optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
    submodel = PLSRegression(n_components=optimal_pls_component_number)
    submodel.fit(selected_autoscaled_x_train, autoscaled_y_train)  # モデルの構築
    submodels.append(submodel)

# サブデータセットの説明変数の種類やサブモデルを保存。同じ名前のファイルがあるときは上書きされるため注意
pd.to_pickle(selected_x_variable_numbers, 'selected_x_variable_numbers.bin')
pd.to_pickle(submodels, 'submodels.bin')

# サブデータセットの説明変数の種類やサブモデルを読み込み
# 今回は、保存した後にすぐ読み込んでいるため、あまり意味はありませんが、サブデータセットの説明変数の種類やサブモデルを
# 保存しておくことで、後で新しいサンプルを予測したいときにモデル構築の過程を省略できます
selected_x_variable_numbers = pd.read_pickle('selected_x_variable_numbers.bin')
submodels = pd.read_pickle('submodels.bin')

# テストデータの y の推定, 重みの計算
estimated_y_test_all, weights = wel.calc_y_and_weights(submodels, selected_x_variable_numbers, x_train, y_train, x_test, k_in_knn_in_wrmsd, multiplier_of_inverse_wrmsed)

# テストデータの推定値の重み付き平均値
estimated_y_test = (estimated_y_test_all.values * weights).sum(axis=1)
estimated_y_test = pd.DataFrame(estimated_y_test, index=x_test.index, columns=['estimated_y'])

plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
predicted_ytest = np.ndarray.flatten(np.array(estimated_y_test))
if raw_x_test.shape[0]:
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(y_test, predicted_ytest)
    y_max = np.max(np.array([np.array(y_test), predicted_ytest]))
    y_min = np.min(np.array([np.array(y_test), predicted_ytest]))
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.show()
    # r2p, RMSEp, MAEp
    print('r2p: {0}'.format(float(1 - sum((y_test - predicted_ytest) ** 2) / sum((y_test - y_test.mean()) ** 2))))
    print('RMSEp: {0}'.format(float((sum((y_test - predicted_ytest) ** 2) / len(y_test)) ** 0.5)))
    print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_ytest)) / len(y_test))))
 

