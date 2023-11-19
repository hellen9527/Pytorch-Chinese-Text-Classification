# -*- coding:utf-8 -*-
# File       : text_classify.py
# Time       : 02/11/2023 下午 10:55
# Author     ：rain
# Description：传统机器学习方法: 决策树、随机森林、XGBOOST、SVM
#             无法batch学习, 只能最后打印训练结果
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from processors.text_classify import cls_processors
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from tools.finetuning_argparse import get_argparse
from tqdm import tqdm
import numpy as np
import os
import copy
import joblib
import xgboost as xgb


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    return joblib.load(filename)


def train(args, X_train, X_dev, X_test, train_labels, dev_labels, test_labels):
    if args.model_type == "dt":
        # 决策树
        clf = DecisionTreeClassifier()
    elif args.model_type == "rf":
        # 随机森林
        clf = RandomForestClassifier(n_jobs=args.n_jobs)
    elif args.model_type == "svm":
        # SVM
        clf = LinearSVC()
    elif args.model_type == "xgb":
        # XGBoost
        clf = xgb.XGBClassifier()
    elif args.model_type == "lr":
        clf = LogisticRegression()
    # 训练模型
    clf.fit(X_train, train_labels)
    if hasattr(clf, "predict_proba"):
        y_train_pred_proba = clf.predict_proba(X_train)
        y_dev_pred_proba = clf.predict_proba(X_dev)
        y_test_pred_proba = clf.predict_proba(X_test)
    else:
        train_prob_pos = clf.decision_function(X_train)
        y_train_pred_proba = (train_prob_pos - train_prob_pos.min()) / (train_prob_pos.max() - train_prob_pos.min())
        dev_prob_pos = clf.decision_function(X_dev)
        y_dev_pred_proba = (dev_prob_pos - dev_prob_pos.min()) / (dev_prob_pos.max() - dev_prob_pos.min())
        test_prob_pos = clf.decision_function(X_test)
        y_test_pred_proba = (test_prob_pos - test_prob_pos.min()) / (test_prob_pos.max() - test_prob_pos.min())
    train_loss = log_loss(train_labels, y_train_pred_proba)
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(train_labels, y_train_pred)
    dev_loss = log_loss(dev_labels, y_dev_pred_proba)
    y_dev_pred = clf.predict(X_dev)
    dev_acc = accuracy_score(dev_labels, y_dev_pred)
    print(f"train loss: {train_loss}, train acc: {train_acc}, dev loss: {dev_loss}, dev acc: {dev_acc}")
    # 测试集打印分类报告与混淆矩阵
    y_pred = clf.predict(X_test)
    test_loss = log_loss(test_labels, y_test_pred_proba)
    test_acc = accuracy_score(test_labels, y_pred)
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    print(classification_report(test_labels, y_pred))
    print(confusion_matrix(test_labels, y_pred))
    output_dir = f"./{args.output_dir}/{args.model_type}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_model(clf, f"{output_dir}/{args.model_type}.model")


def test(args, X_test, test_labels):
    model_file = f"./{args.output_dir}/{args.model_type}/{args.model_type}.model"
    if not os.path.exists(model_file):
        print("error, model file not exists...")
        return
    clf = load_model(model_file)
    if hasattr(clf, "predict_proba"):
        y_test_pred_proba = clf.predict_proba(X_test)
    else:
        test_prob_pos = clf.decision_function(X_test)
        y_test_pred_proba = (test_prob_pos - test_prob_pos.min()) / (test_prob_pos.max() - test_prob_pos.min())
    # 测试集打印分类报告与混淆矩阵
    y_pred = clf.predict(X_test)
    test_loss = log_loss(test_labels, y_test_pred_proba)
    test_acc = accuracy_score(test_labels, y_pred)
    print(f"test loss: {test_loss}, test acc: {test_acc}")
    print(classification_report(test_labels, y_pred))
    print(confusion_matrix(test_labels, y_pred))


def main():
    args = get_argparse().parse_args()  # 训练输入参数处理, 需要新增/修改参数可以进入get_argparse配置

    processor = cls_processors[args.task_name](args.data_dir, data_format=args.data_format)

    label_list, label2id, id2label = processor.get_labels()

    train_examples = processor.get_train_examples(args.data_dir)
    dev_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    train_corpus = [example.text_a for example in train_examples]
    train_labels = np.array([label2id[example.label] for example in train_examples])

    dev_corpus = [example.text_a for example in dev_examples]
    dev_labels = np.array([label2id[example.label] for example in dev_examples])

    test_corpus = [example.text_a for example in test_examples]
    test_labels = np.array([label2id[example.label] for example in test_examples])
    print("load datasets is finished...")

    vectorizer = TfidfVectorizer()  # 实例一个模型
    X_train = vectorizer.fit_transform(train_corpus).toarray()  # 把语料库传进去
    X_dev = vectorizer.transform(dev_corpus).toarray()
    X_test = vectorizer.transform(test_corpus).toarray()
    print("transform feature is finished...")

    if args.do_train:
        train(args, X_train, X_dev, X_test, train_labels, dev_labels, test_labels)
    elif args.do_test:
        test(args, X_test, test_labels)


if __name__ == "__main__":
    main()
