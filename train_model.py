import os
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from loss_function import bagging_loss_function, mlp_loss_function, svm_loss_function
from result_pred import accPrdRec, mhCart, mtnl

X_reSubMain = pd.read_csv('data/submain/data.csv')
X_subMain = X_reSubMain.drop(['label'], axis=1)
y_subMain = X_reSubMain['label']
X_subMain = np.array(X_subMain)

sc = StandardScaler()
X_combined_subMain = sc.fit_transform(X_subMain)


def fi_Para_svm(X_train, y_train):

    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1, 'scale', 'auto'],
                  'kernel': ['rbf', 'poly', 'linear']}

    grid_search = GridSearchCV(
        SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print("Tham số tốt nhất: ", grid_search.best_params_)
    print("Điểm số tốt nhất: ", grid_search.best_score_)

    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        if 0.8 < mean_score < 0.99:
            print("Điểm:", mean_score, "Tham số:", params)
            svm_model = SVC(
                C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
            print(svm_model)


def fi_Para_cart(X_train, y_train):

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(
    ), param_grid=param_grid, cv=5, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    print("Tham số tốt nhất: ", grid_search.best_params_)
    print("Điểm số tốt nhất: ", grid_search.best_score_)

    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        if 0.8 < mean_score < 0.99:
            print(f"Điểm: {mean_score:.4f}, Tham số: {params}")


def fi_Para_mlp(X_train, y_train):

    param_grid = {
        'hidden_layer_sizes': [(2048, 1024), (4096, 2048), (4096, 2048, 1024), (2048, 1024, 512)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000],
        'tol': [0.000000001, 0.0000001]
    }

    mlp = MLPClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    print("Tham số tốt nhất ", grid_search.best_params_)
    print("Điểm tốt nhất: ", grid_search.best_score_)

    results_df = pd.DataFrame(grid_search.cv_results_)
    filtered_results = results_df[(results_df['mean_test_score'] > 0.8) & (
        results_df['mean_test_score'] < 0.99)]

    print("Kết quả với độ chính xác từ 0.8 đén 0.99:")
    print(filtered_results[['params', 'mean_test_score', 'std_test_score']])


def submain_data_svm():

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined_subMain, y_subMain, test_size=0.2, random_state=42)

    # fi_Para_svm(X_train, y_train)

    svm_model = SVC(C=10, gamma='scale', kernel='poly')

    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    accPrdRec(y_test, y_pred)

    mtnl(y_test, y_pred)

    svm_loss_function(X_test, y_test, svm_model)

    filename = 'data_train/svm/model_svm.sav'
    pickle.dump(svm_model, open(filename, 'wb'))

    filename_scaler = 'data_train/svm/scaler_svm.sav'
    pickle.dump(sc, open(filename_scaler, 'wb'))

    print("Thành công!!")


# submain_data_svm()


def submain_data_cart():

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined_subMain, y_subMain, test_size=0.2, random_state=42)

    cart_model = DecisionTreeClassifier(
        criterion='gini', max_depth=25, min_samples_leaf=1, min_samples_split=5)

    # fi_Para_cart(X_train, y_train)

    cart_model.fit(X_train, y_train)

    y_pred = cart_model.predict(X_test)

    accPrdRec(y_test, y_pred)

    mtnl(y_test, y_pred)

    mhCart(cart_model)

    filename = 'data_train/cart/model_cart.sav'
    pickle.dump(cart_model, open(filename, 'wb'))

    filename_scaler = 'data_train/cart/scaler_cart.sav'
    pickle.dump(sc, open(filename_scaler, 'wb'))

    print("Thành công!!")


# submain_data_cart()


def submain_data_mlp():

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined_subMain, y_subMain, test_size=0.2, random_state=42)

    # fi_Para_mlp(X_train, y_train)

    mlp_model = MLPClassifier(hidden_layer_sizes=(4096,), activation='tanh', solver='sgd',
                              alpha=0.0001, learning_rate='adaptive', tol=0.000001, max_iter=1000)

    mlp_model.fit(X_train, y_train)

    y_pred = mlp_model.predict(X_test)

    accPrdRec(y_test, y_pred)

    mtnl(y_test, y_pred)

    mlp_loss_function(X_test, y_test, mlp_model)

    filename = 'data_train/mlp/model_mlp.sav'
    pickle.dump(mlp_model, open(filename, 'wb'))

    filename_scaler = 'data_train/mlp/scaler_mlp.sav'
    pickle.dump(sc, open(filename_scaler, 'wb'))

    print("Thành công!!")


# submain_data_mlp()


def submain_data_bag():

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined_subMain, y_subMain, test_size=0.2, random_state=42)

    bagging_svm = SVC(C=10, gamma='scale', kernel='poly')
    bagging_cart = DecisionTreeClassifier()
    bagging_mlp = MLPClassifier(hidden_layer_sizes=(4096,), activation='tanh', solver='sgd',
                                alpha=0.0001, learning_rate='adaptive', tol=0.000001, max_iter=1000)

    bagging_svm_miauli = BaggingClassifier(
        estimator=bagging_svm, n_estimators=10, random_state=42)
    bagging_cart_gihbn = BaggingClassifier(
        estimator=bagging_cart, n_estimators=10, random_state=42)
    bagging_mlp_bhtre = BaggingClassifier(
        estimator=bagging_mlp, n_estimators=10, random_state=42)

    bagging_svm_miauli.fit(X_train, y_train)
    bagging_cart_gihbn .fit(X_train, y_train)
    bagging_mlp_bhtre.fit(X_train, y_train)

    y_pred_bagging_svm = bagging_svm_miauli.predict(X_test)
    y_pred_bagging_cart = bagging_cart_gihbn.predict(X_test)
    y_pred_bagging_mlp = bagging_mlp_bhtre.predict(X_test)

    y_pred_combined = (y_pred_bagging_svm +
                       y_pred_bagging_cart + y_pred_bagging_mlp) / 3

    y_pred_combined_rounded = np.round(y_pred_combined).astype(int)

    accPrdRec(y_test, y_pred_combined_rounded)

    mtnl(y_test, y_pred_combined_rounded)

    bagging_loss_function(
        X_test, y_test, {'svm': bagging_svm_miauli, 'cart': bagging_cart_gihbn})

    # filename_svm = 'data_train/bag/bagging_svm_model.sav'
    # pickle.dump(bagging_svm_miauli, open(filename_svm, 'wb'))

    # filename_cart = 'data_train/bag/bagging_cart_model.sav'
    # pickle.dump(bagging_cart_gihbn, open(filename_cart, 'wb'))

    # # filename_mlp = 'data_train/bag/bagging_mlp_model.sav'
    # # pickle.dump(bagging_mlp_bhtre, open(filename_mlp, 'wb'))

    # filename_cart = 'data_train/bag/scaler_model.sav'
    # pickle.dump(bagging_cart_gihbn, open(filename_cart, 'wb'))

    print("Lưu thành công!!")


# submain_data_bag()
