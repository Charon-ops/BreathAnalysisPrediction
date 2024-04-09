import os
import csv

import joblib
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

from sklearn import metrics


class PredictionModel:
    def __init__(self):
        self.data_copy = None

        self.data = {
            "features": [],
            "train": np.zeros((1, 1)),
            "eval": np.zeros((1, 1))
        }
        self.model = None
        self.result = {
            'accuracy': {
                'train': [],
                'eval': []
            },
            'auc': {
                'train': [],
                'eval': []
            },
            'f1score': {
                'train': [],
                'eval': []
            },
            'confusion matrix': {
                'train': [],
                'trainAll': [],
                'eval': []
            },
            'roc curve': {
                'trainFpr': [],
                'trainTpr': [],
                'evalFpr': [],
                'evalTpr': []
            }
        }

    def loadData(self, path_train, path_eval, features, status):
        if not os.path.exists(path_train):
            print('文件路径不存在')
        self.data["features"] = features
        # 读入训练集、验证集数据
        _data_train = pd.read_excel(path_train)
        _data_eval = pd.read_excel(path_eval)
        # 构建训练集、验证集
        self.data["train"] = np.concatenate([np.expand_dims(_data_train[feature], axis=1) for feature in features],
                                            axis=1)
        self.data["train"] = np.concatenate([self.data["train"], np.expand_dims(_data_train[status], axis=1)], axis=1)
        self.data["eval"] = np.concatenate([np.expand_dims(_data_eval[feature], axis=1) for feature in features],
                                           axis=1)
        self.data["eval"] = np.concatenate([self.data["eval"], np.expand_dims(_data_eval[status], axis=1)], axis=1)
        return

    def plotDistribution(self, column_num=3):
        _row_num = math.ceil(len(self.data["features"]) / column_num)
        for idx in range(len(self.data["features"])):
            _y = np.squeeze(np.take(self.data["train"], [idx], axis=1), axis=1)
            plt.subplot2grid((_row_num, column_num), (int(idx / column_num), int(idx % column_num)))
            plt.hist(_y, np.arange(min(_y), 1.1 * max(_y), (max(_y) - min(_y)) / 10), rwidth=0.8)
            plt.xlabel(self.data["features"][idx], labelpad=0)
            plt.ylabel('Probability')
        plt.suptitle('data distribution')
        plt.show()
        return

    def dataProcessing(self, manners):
        for feature in manners:
            idx = self.data["features"].index(feature)
            for manner in manners[feature]:
                if manner == 'log10':
                    self.data["train"][:, idx] = np.log10(self.data["train"][:, idx] + 1)
                    self.data["eval"][:, idx] = np.log10(self.data["eval"][:, idx] + 1)
                elif type(manner) == int:
                    self.data["train"][:, idx] = self.data["train"][:, idx] * pow(10, manner)
                    self.data["eval"][:, idx] = self.data["eval"][:, idx] * pow(10, manner)
                elif manner == 'Standardization':
                    _std = np.std(self.data["train"][:, idx])
                    _mean = np.mean(self.data["train"][:, idx])
                    self.data["train"][:, idx] = (self.data["train"][:, idx] - _mean) / _std
                    self.data["eval"][:, idx] = (self.data["eval"][:, idx] - _mean) / _std
                elif manner == 'Normalization':
                    _min = np.min(self.data["train"][:, idx])
                    _max = np.max(self.data["eval"][:, idx])
                    self.data["train"][:, idx] = (self.data["train"][:, idx] - _min) / (_max - _min)
                    self.data["eval"][:, idx] = (self.data["eval"][:, idx] - _min) / (_max - _min)
        return

    def dataAugment(self):
        _X, _y = np.delete(self.data["train"], -1, axis=1), np.squeeze(self.data["train"][:, -1])
        _imb = RandomOverSampler(random_state=42)
        _X, _y = _imb.fit_resample(_X, _y)
        self.data["train"] = np.concatenate((_X, np.expand_dims(_y, axis=1)), axis=1)
        return

    def makeModel(self, model='xgboost', pkl=None, save_path=None, mod='train'):
        if mod == 'train':
            if model == 'xgboost':
                self.xgboost(save_path=save_path)
            elif model == 'svm':
                self.svm(save_path=save_path)
            elif model == 'lr':
                self.lr(save_path=save_path)
            return
        if mod == 'val':
            self.model_val(model, 100, pkl=pkl)

        print('-----------------------------------------')
        print('|        train confusion matrix         |')
        print('|---------------------------------------|')
        print('|  TP:{:>6d} |   FP:{:>6d}  |  {:>6d}   |'.format(self.result['confusion matrix']['trainAll'][3],
                                                                 self.result['confusion matrix']['trainAll'][1],
                                                                 self.result['confusion matrix']['trainAll'][3]
                                                                 + self.result['confusion matrix']['trainAll'][1]))
        print('|---------------------------------------|')
        print('|  FN:{:>6d} |   TN:{:>6d}  |  {:>6d}   |'.format(self.result['confusion matrix']['trainAll'][2],
                                                                 self.result['confusion matrix']['trainAll'][0],
                                                                 self.result['confusion matrix']['trainAll'][2]
                                                                 + self.result['confusion matrix']['trainAll'][0]))
        print('|---------------------------------------|')
        print('|   {:>6d}   |    {:>6d}    |  {:>6d}   |'.format(self.result['confusion matrix']['trainAll'][3]
                                                                 + self.result['confusion matrix']['trainAll'][2],
                                                                 self.result['confusion matrix']['trainAll'][1]
                                                                 + self.result['confusion matrix']['trainAll'][0],
                                                                 sum(self.result['confusion matrix']['trainAll'])))
        print('-----------------------------------------')

        print('-----------------------------------------')
        print('|       eval confusion matrix           |')
        print('|---------------------------------------|')
        print('|  TP:{:>6d} |   FP:{:>6d}  |  {:>6d}   |'.format(self.result['confusion matrix']['evalAll'][3],
                                                                 self.result['confusion matrix']['evalAll'][1],
                                                                 self.result['confusion matrix']['evalAll'][3]
                                                                 + self.result['confusion matrix']['evalAll'][1]))
        print('|---------------------------------------|')
        print('|  FN:{:>6d} |   TN:{:>6d}  |  {:>6d}   |'.format(self.result['confusion matrix']['evalAll'][2],
                                                                 self.result['confusion matrix']['evalAll'][0],
                                                                 self.result['confusion matrix']['evalAll'][2]
                                                                 + self.result['confusion matrix']['evalAll'][0]))
        print('|---------------------------------------|')
        print('|   {:>6d}   |    {:>6d}    |  {:>6d}   |'.format(self.result['confusion matrix']['evalAll'][3]
                                                                 + self.result['confusion matrix']['evalAll'][2],
                                                                 self.result['confusion matrix']['evalAll'][1]
                                                                 + self.result['confusion matrix']['evalAll'][0],
                                                                 sum(self.result['confusion matrix']['evalAll'])))
        print('-----------------------------------------')

        _train_maxIndex = self.result['accuracy']['train'].index(sorted(self.result['accuracy']['train'])[4])
        _train_sensitivity = []
        _train_specificity = []
        _train_precision = []
        _eval_maxIndex = self.result['accuracy']['eval'].index(sorted(self.result['accuracy']['eval'])[4])
        _eval_sensitivity = []
        _eval_specificity = []
        _eval_precision = []
        # tn, fp, fn, tp
        for matrix in self.result['confusion matrix']['train']:
            _train_sensitivity.append(matrix[3] / (matrix[3] + matrix[2]))
            _train_specificity.append(matrix[0] / (matrix[0] + matrix[1]))
            _train_precision.append(matrix[3] / (matrix[3] + matrix[1]))
        for matrix in self.result['confusion matrix']['eval']:
            _eval_sensitivity.append(matrix[3] / (matrix[3] + matrix[2]))
            _eval_specificity.append(matrix[0] / (matrix[0] + matrix[1]))
            _eval_precision.append(matrix[3] / (matrix[3] + matrix[1]))
        print('-------------------------------------------------------------------------')
        print('|           |             train           |       validation            |')
        print('|-----------------------------------------------------------------------|')
        print('|    auc    | {:>8.3f}({:>.3f}-{:<.3f}) | {:>8.3f}({:>.3f}-{:<.3f}) |'
              .format(np.mean(self.result['auc']['train']),
                      np.percentile(self.result['auc']['train'], [5, 95])[0],
                      np.percentile(self.result['auc']['train'], [5, 95])[1],
                      np.mean(self.result['auc']['eval']),
                      np.percentile(self.result['auc']['eval'], [5, 95])[0],
                      np.percentile(self.result['auc']['eval'], [5, 95])[1]))
        print('|-----------------------------------------------------------------------|')
        print('| accuracy  | {:>8.3f}({:>.3f}-{:<.3f}) | {:>8.3f}({:>.3f}-{:<.3f}) |'
              .format(np.mean(self.result['accuracy']['train']),
                      np.percentile(self.result['accuracy']['train'], [5, 95])[0],
                      np.percentile(self.result['accuracy']['train'], [5, 95])[1],
                      np.mean(self.result['accuracy']['eval']),
                      np.percentile(self.result['accuracy']['eval'], [5, 95])[0],
                      np.percentile(self.result['accuracy']['eval'], [5, 95])[1]))
        print('|-----------------------------------------------------------------------|')
        print('|sensitivity| {:>8.3f}({:>.3f}-{:<.3f}) | {:>8.3f}({:>.3f}-{:<.3f}) |'
              .format(np.mean(_train_sensitivity),
                      np.percentile(_train_sensitivity, [5, 95])[0],
                      np.percentile(_train_sensitivity, [5, 95])[1],
                      np.mean(_eval_sensitivity),
                      np.percentile(_eval_sensitivity, [5, 95])[0],
                      np.percentile(_eval_sensitivity, [5, 95])[1]))
        print('|-----------------------------------------------------------------------|')
        print('|specificity| {:>8.3f}({:>.3f}-{:<.3f}) | {:>8.3f}({:>.3f}-{:<.3f}) |'
              .format(np.mean(_train_specificity),
                      np.percentile(_train_specificity, [5, 95])[0],
                      np.percentile(_train_specificity, [5, 95])[1],
                      np.mean(_eval_specificity),
                      np.percentile(_eval_specificity, [5, 95])[0],
                      np.percentile(_eval_specificity, [5, 95])[1]))
        print('|-----------------------------------------------------------------------|')
        print('| precision | {:>8.3f}({:>.3f}-{:<.3f}) | {:>8.3f}({:>.3f}-{:<.3f}) |'
              .format(np.mean(_train_precision),
                      np.percentile(_train_precision, [5, 95])[0],
                      np.percentile(_train_precision, [5, 95])[1],
                      np.mean(_eval_precision),
                      np.percentile(_eval_precision, [5, 95])[0],
                      np.percentile(_eval_precision, [5, 95])[1]))
        print('|-----------------------------------------------------------------------|')
        print('|  f1score  | {:>8.3f}({:>.3f}-{:<.3f}) | {:>8.3f}({:>.3f}-{:<.3f}) |'
              .format(np.mean(self.result['f1score']['train']),
                      np.percentile(self.result['f1score']['train'], [5, 95])[0],
                      np.percentile(self.result['f1score']['train'], [5, 95])[1],
                      np.mean(self.result['f1score']['eval']),
                      np.percentile(self.result['f1score']['eval'], [5, 95])[0],
                      np.percentile(self.result['f1score']['eval'], [5, 95])[1]))
        print('-------------------------------------------------------------------------')

        _reference_x = np.linspace(0, 1)
        _reference_y = np.linspace(0, 1)
        plt.rc('font', family='Times New Roman')
        plt.subplot(121)
        plt.plot(self.result['roc curve']['trainFpr'],
                 self.result['roc curve']['trainTpr'],
                 'r-')
        # plt.plot(_reference_x, _reference_y, 'b', linewidth=0.25)
        plt.ylabel('Sensitivity'), plt.xlabel('1-Specificity'), plt.title('Train ROC Curve')
        plt.subplot(122)
        plt.plot(self.result['roc curve']['evalFpr'],
                 self.result['roc curve']['evalTpr'], 'r-')
        # plt.plot(_reference_x, _reference_y, 'b', linewidth=0.25)
        plt.ylabel('Sensitivity'), plt.xlabel('1-Specificity'), plt.title('Eval ROC Curve')
        plt.show()
        return

    def xgboost(self, threshold=0.5, save_path=None):
        _accuracy_max = 0

        for i in range(10):

            _train_index = [idx for idx in range(len(self.data["train"][:, 0])) if idx % 10 != i]
            _train_var = np.take(np.delete(self.data["train"], -1, axis=1), _train_index, axis=0)
            _train_res = np.take(self.data["train"][:, -1], _train_index, axis=0)

            _val_index = [idx for idx in range(len(self.data["train"][:, 0])) if idx % 10 == i]
            _val_var = np.take(np.delete(self.data["train"], -1, axis=1), _val_index, axis=0)
            _val_res = np.take(self.data["train"][:, -1], _val_index, axis=0)
            _DTrain_var = xgb.DMatrix(_train_var, label=_train_res)
            _DVal_var = xgb.DMatrix(_val_var)
            _params = {
                'booster': 'gbtree',
                'objective': 'binary:logistic',
                'eta': 0.15,
                'learning_rate': 1,
                'eval_metric': 'auc',
                'max_depth': 10,
                'lambda': 5,
                'subsample': 0.75,
                'min_child_weight': 2,
                'gamma': 0.15,
            }
            _model = xgb.train(_params, _DTrain_var, num_boost_round=50)

            _val_probability = np.array(_model.predict(_DVal_var))
            _val_prediction = [1 if i > threshold else 0 for i in _val_probability]

            _accuracy = metrics.accuracy_score(_val_res, _val_prediction)
            if _accuracy > _accuracy_max:
                self.model = _model

        if save_path is not None:
            self.model.save_model(save_path)

    def svm(self, threshold=0.5, save_path=None):
        _accuracy_max = 0

        for i in range(1):
            print('starting....')
            _train_index = [idx for idx in range(len(self.data["train"][:, 0])) if idx % 10 != i]
            _train_var = np.take(np.delete(self.data["train"], -1, axis=1), _train_index, axis=0)
            _train_res = np.take(self.data["train"][:, -1], _train_index, axis=0)

            _val_index = [idx for idx in range(len(self.data["train"][:, 0])) if idx % 10 == i]
            _val_var = np.take(np.delete(self.data["train"], -1, axis=1), _val_index, axis=0)
            _val_res = np.take(self.data["train"][:, -1], _val_index, axis=0)

            _eval_var = np.delete(self.data["eval"], -1, axis=1)
            _eval_res = self.data["eval"][:, -1]

            _params = [{
                'kernel': ['linear'],
                'probability': [True]
            }]
            _model = GridSearchCV(SVC(), _params)
            print('fitting...')
            _model.fit(_train_var, _train_res)
            print('fitting finish...')

            _val_probability = np.array(_model.predict_proba(_val_var)[:, 1])
            _val_prediction = [1 if i > threshold else 0 for i in _val_probability]

            _accuracy = metrics.accuracy_score(_val_res, _val_prediction)
            if _accuracy > _accuracy_max:
                self.model = _model

        if save_path is not None:
            joblib.dump(self.model.best_estimator_, save_path)
        return

    def lr(self, threshold=0.5, save_path=None):
        _accuracy_max = 0

        for i in range(10):
            _train_index = [idx for idx in range(len(self.data["train"][:, 0])) if idx % 10 != i]
            _train_var = np.take(np.delete(self.data["train"], -1, axis=1), _train_index, axis=0)
            _train_res = np.take(self.data["train"][:, -1], _train_index, axis=0)

            _val_index = [idx for idx in range(len(self.data["train"][:, 0])) if idx % 10 == i]
            _val_var = np.take(np.delete(self.data["train"], -1, axis=1), _val_index, axis=0)
            _val_res = np.take(self.data["train"][:, -1], _val_index, axis=0)

            _eval_var = np.delete(self.data["eval"], -1, axis=1)
            _eval_res = self.data["eval"][:, -1]

            _params = [{
                'penalty': ['l2'],
                'max_iter': [10000]
            }]
            _model = GridSearchCV(LogisticRegression(), _params)
            _model.fit(_train_var, _train_res)

            _val_probability = np.array(_model.predict_proba(_val_var)[:, 1])
            _val_prediction = [1 if i > threshold else 0 for i in _val_probability]

            _accuracy = metrics.accuracy_score(_val_res, _val_prediction)
            if _accuracy > _accuracy_max:
                self.model = _model

        if save_path is not None:
            joblib.dump(self.model.best_estimator_, save_path)
        return

    def subgroupsAnalysis(self, feature, choose_features, pkl, model='xgboost', partition=None):
        _feature_idx = self.data["features"].index(feature)
        _choose_idx = [self.data["features"].index(_feature) for _feature in choose_features]
        _choose_idx.append(-1)
        if partition is None:
            _chooseList_train = [_item for _item in range(len(self.data["train"][:, _feature_idx]))
                                 if self.data["train"][_item, :][_feature_idx] == 0]
            _classification0_train = np.take(self.data["train"], _chooseList_train, axis=0)
            _classification0_train = np.take(_classification0_train, _choose_idx, axis=1)

            _chooseList_eval = [_item for _item in range(len(self.data["eval"][:, _feature_idx]))
                                if self.data["eval"][_item, :][_feature_idx] == 0]
            _classification0_eval = np.take(self.data["eval"], _chooseList_eval, axis=0)
            _classification0_eval = np.take(_classification0_eval, _choose_idx, axis=1)

            _chooseList_train = [_item for _item in range(len(self.data["train"][:, _feature_idx]))
                                 if self.data["train"][_item, :][_feature_idx] == 1]
            _classification1_train = np.take(self.data["train"], _chooseList_train, axis=0)
            _classification1_train = np.take(_classification1_train, _choose_idx, axis=1)

            _chooseList_eval = [_item for _item in range(len(self.data["eval"][:, _feature_idx]))
                                if self.data["eval"][_item, :][_feature_idx] == 1]
            _classification1_eval = np.take(self.data["eval"], _chooseList_eval, axis=0)
            _classification1_eval = np.take(_classification1_eval, _choose_idx, axis=1)
        else:
            _chooseList_train = [_item for _item in range(len(self.data["train"][:, _feature_idx]))
                                 if self.data["train"][_item, :][_feature_idx] <= partition]
            _classification0_train = np.take(self.data["train"], _chooseList_train, axis=0)
            _classification0_train = np.take(_classification0_train, _choose_idx, axis=1)

            _chooseList_eval = [_item for _item in range(len(self.data["eval"][:, _feature_idx]))
                                if self.data["eval"][_item, :][_feature_idx] <= partition]
            _classification0_eval = np.take(self.data["eval"], _chooseList_eval, axis=0)
            _classification0_eval = np.take(_classification0_eval, _choose_idx, axis=1)

            _chooseList_train = [_item for _item in range(len(self.data["train"][:, _feature_idx]))
                                 if self.data["train"][_item, :][_feature_idx] > partition]
            _classification1_train = np.take(self.data["train"], _chooseList_train, axis=0)
            _classification1_train = np.take(_classification1_train, _choose_idx, axis=1)

            _chooseList_eval = [_item for _item in range(len(self.data["eval"][:, _feature_idx]))
                                if self.data["eval"][_item, :][_feature_idx] > partition]
            _classification1_eval = np.take(self.data["eval"], _chooseList_eval, axis=0)
            _classification1_eval = np.take(_classification1_eval, _choose_idx, axis=1)

        print('classification 0 or less partition')
        self.data["train"] = _classification0_train
        self.data["eval"] = _classification0_eval
        self.makeModel(model, pkl=pkl, mod='val')
        print('classification 1 or above partition')
        self.data["train"] = _classification1_train
        self.data["eval"] = _classification1_eval
        self.makeModel(model, pkl=pkl, mod='val')

        self.data["train"] = np.concatenate((_classification0_train, _classification1_train))
        self.data["eval"] = np.concatenate((_classification0_eval, _classification1_eval))

        return

    def model_val(self, model, n_bootstraps, pkl=None, threshold=0.5):

        if pkl is None and self.model is None:
            print("You must input model file")

        if pkl is not None:
            if model == 'xgboost':
                self.model = xgb.Booster()
                self.model.load_model(pkl)
            else:
                self.model = joblib.load(pkl)

        _train_var = np.delete(self.data["train"], -1, axis=1)
        _train_res = self.data["train"][:, -1]
        _eval_var = np.delete(self.data["eval"], -1, axis=1)
        _eval_res = self.data["eval"][:, -1]
        if model == 'xgboost':
            _train_var = xgb.DMatrix(_train_var)
            _eval_var = xgb.DMatrix(_eval_var)

        if model == 'xgboost':
            _train_proba = np.array(self.model.predict(_train_var))
            _train_pre = np.array([1 if i >= threshold else 0 for i in _train_proba])
            _eval_proba = np.array(self.model.predict(_eval_var))
            _eval_pre = np.array([1 if i >= threshold else 0 for i in _eval_proba])
        else:
            _train_proba = np.array(self.model.predict_proba(_train_var)[:, -1])
            _train_pre = np.array([1 if i >= threshold else 0 for i in _train_proba])
            _eval_proba = np.array(self.model.predict_proba(_eval_var)[:, -1])
            _eval_pre = np.array([1 if i >= threshold else 0 for i in _eval_proba])

        _rng = np.random.RandomState(42)
        for i in range(n_bootstraps):
            _indices_train = _rng.randint(0, len(_train_pre), len(_train_pre))
            if len(np.unique(_train_res[_indices_train])) < 2:
                continue
            _train_tn, _train_fp, _train_fn, _train_tp = metrics.confusion_matrix(_train_res[_indices_train], _train_pre[_indices_train]).ravel()

            self.result["auc"]["train"].append(metrics.roc_auc_score(_train_res[_indices_train], _train_proba[_indices_train]))
            self.result["accuracy"]["train"].append(metrics.accuracy_score(_train_res[_indices_train], _train_pre[_indices_train]))
            self.result["f1score"]["train"].append(metrics.f1_score(_train_res[_indices_train], _train_pre[_indices_train]))
            self.result["confusion matrix"]['train'].append([_train_tn, _train_fp, _train_fn, _train_tp])

            _indices_eval = _rng.randint(0, len(_eval_pre), len(_eval_pre))
            if len(np.unique(_eval_res[_indices_eval])) < 2:
                continue
            _eval_tn, _eval_fp, _eval_fn, _eval_tp = metrics.confusion_matrix(_eval_res[_indices_eval], _eval_pre[_indices_eval]).ravel()
            self.result["auc"]["eval"].append(metrics.roc_auc_score(_eval_res[_indices_eval], _eval_proba[_indices_eval]))
            self.result["accuracy"]["eval"].append(metrics.accuracy_score(_eval_res[_indices_eval], _eval_pre[_indices_eval]))
            self.result["f1score"]["eval"].append(metrics.f1_score(_eval_res[_indices_eval], _eval_pre[_indices_eval]))
            self.result["confusion matrix"]['eval'].append([_eval_tn, _eval_fp, _eval_fn, _eval_tp])

        _train_tn, _train_fp, _train_fn, _train_tp = metrics.confusion_matrix(_train_res, _train_pre).ravel()
        _eval_tn, _eval_fp, _eval_fn, _eval_tp = metrics.confusion_matrix(_eval_res, _eval_pre).ravel()
        _train_fpr, _train_tpr, _ = metrics.roc_curve(_train_res, _train_proba)
        _eval_fpr, _eval_tpr, _ = metrics.roc_curve(_eval_res, _eval_proba)
        self.result["confusion matrix"]['trainAll'] = [_train_tn, _train_fp, _train_fn, _train_tp]
        self.result["confusion matrix"]['evalAll'] = [_eval_tn, _eval_fp, _eval_fn, _eval_tp]
        self.result["roc curve"]["trainTpr"] = _train_tpr
        self.result["roc curve"]["trainFpr"] = _train_fpr
        self.result["roc curve"]["evalTpr"] = _eval_tpr
        self.result["roc curve"]["evalFpr"] = _eval_fpr

    def featureImportance(self, features):

        data_var = np.delete(self.data["train"], -1, axis=1)
        data_res = self.data["train"][:, -1]

        feature_names = features
        lr = LogisticRegression()
        lr.fit(data_var, data_res)

        result = permutation_importance(
            lr, data_var, data_res, n_repeats=10, random_state=42, n_jobs=2
        )

        lr_importance = pd.Series(result.importances_mean, index=feature_names)

        fig, ax = plt.subplots()
        lr_importance.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        plt.show()
