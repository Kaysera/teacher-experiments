from teacher.datasets import load_beer, load_compas, load_heloc, load_pima, load_breast
from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points, dataset_membership
from teacher.tree import FDT
from teacher.explanation import mr_factual, m_factual, c_factual, f_counterfactual, i_counterfactual
from teacher.explanation._factual import _robust_threshold, _fired_rules, _get_class_fired_rules
from teacher.fuzzy import FuzzyContinuousSet
import sys
from functools import reduce
import os

import argparse
import random
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

DATASETS = {
    'beer':  load_beer,
    'breast':  load_breast,
    'compas': load_compas,
    'heloc': load_heloc,
    'pima': load_pima
}


def ld_instance(ic_zip, le):
    ninst = []
    for col, val in ic_zip:
        if col in le:
            nval = le[col].inverse_transform([val])[0]
        else:
            nval = val
        ninst.append((col, nval))
    return ninst

def _get_fuzzy_element(fuzzy_X, idx):
    element = {}
    for feat in fuzzy_X:
        element[feat] = {}
        for fuzzy_set in fuzzy_X[feat]:
            try:
                element[feat][str(fuzzy_set)] = pd.to_numeric(fuzzy_X[feat][fuzzy_set][idx])
            except ValueError:
                element[feat][str(fuzzy_set)] = fuzzy_X[feat][fuzzy_set][idx]

    return element

def prepare_dataset(ds):
    try:
        dataset = DATASETS[ds]()
    except KeyError:
        print('Dataset not found')
        sys.exit()

    df = dataset['df']
    class_name = dataset['class_name']
    X = df.drop(class_name, axis=1)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    df_categorical_columns = dataset['discrete']
    class_name = dataset['class_name']
    df_categorical_columns.remove(class_name)
    df_numerical_columns = dataset['continuous']

    X_num = X[dataset['continuous']]
    num_cols = X_num.columns
    fuzzy_points = get_fuzzy_points('entropy', num_cols, X_num, y)

    discrete_fuzzy_values = {col: df[col].unique() for col in df_categorical_columns}
    fuzzy_variables_order = {col: i for i, col in enumerate(X.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)

    df_train_membership = dataset_membership(X_train, fuzzy_variables)
    df_test_membership = dataset_membership(X_test, fuzzy_variables)

    fuzzy_element_idx = 73
    fuzzy_element = _get_fuzzy_element(df_test_membership, fuzzy_element_idx)
    all_classes = dataset['possible_outcomes']


    return [df_train_membership, df_test_membership, X_train, y_train,
            X_test, y_test, fuzzy_element, fuzzy_element_idx, all_classes, df_numerical_columns, fuzzy_variables]


def cf_stats(cf, inst, target, fuzzy_vars, fdt):
    for i, (r, d) in enumerate(cf):
        n_instance, n_changes = apply_rule(r, inst, fuzzy_vars)
        n_target = fdt.predict(n_instance)
        if n_target != target: 
            break
    # Number of changes, rules tested, distance
    return n_changes, i + 1, d


def apply_rule(rule, instance, fuzzy_vars):
    changes = {a[0]: a[1] for a in rule.antecedent}
    n_inst = instance.copy()
    n_changes = 0
    for i, fv in enumerate(fuzzy_vars):
        fs_idx = np.argmax([a[0] for a in fv.membership(instance[:,i]).values()])
        if fv.name in changes and changes[fv.name] != fv.fuzzy_sets[fs_idx].name:
            n_changes += 1
            if isinstance(fv.fuzzy_sets[fs_idx], FuzzyContinuousSet):
                n_inst[:,i] = float(changes[fv.name])
            else:
                n_inst[:,i] = changes[fv.name]

    return n_inst, n_changes


def run_experiment(db):
    print(f'Preparing dataset {db}')
    [df_train_membership, df_test_membership, X_train, y_train,
                X_test, y_test, fuzzy_element, fuzzy_element_idx, all_classes, df_numerical_columns, fuzzy_variables] = prepare_dataset(db)

    print(f'Training FDT')
    new_fdt = FDT(fuzzy_variables, min_num_examples=1)
    new_fdt.fit(X_train, y_train)

    rules = new_fdt.to_rule_based_system()

    m_changes_avg = m_cf_rules_avg = m_cf_dist_avg = 0
    mr_changes_avg = mr_cf_rules_avg = mr_cf_dist_avg = 0
    c_changes_avg = c_cf_rules_avg = c_cf_dist_avg = 0
    cavg_changes_avg = cavg_cf_rules_avg = cavg_cf_dist_avg = 0
    crth_changes_avg = crth_cf_rules_avg = crth_cf_dist_avg = 0
    i_changes_avg = i_cf_rules_avg = i_cf_dist_avg = 0
    num_cf_avg = 0
    counter = 0

    for fuzzy_idx in range(len(X_test)):
        print(f'Processing instance {fuzzy_idx}/{len(X_test)}')
        instance = X_test.iloc[fuzzy_idx].to_numpy().reshape(1, -1)
        new_fdt_predict = new_fdt.predict(instance)

        fuzzy_element = _get_fuzzy_element(df_test_membership, fuzzy_idx)
        m_fact = m_factual(fuzzy_element, rules, new_fdt_predict)
        mr_fact = mr_factual(fuzzy_element, rules, new_fdt_predict)
        c_fact = c_factual(fuzzy_element, rules, new_fdt_predict, lam=0.1)
        fired_rules = _fired_rules(fuzzy_element, rules)
        class_fired_rules = _get_class_fired_rules(fired_rules, new_fdt_predict)
        class_fired_rules.sort(key=lambda rule: rule.matching(fuzzy_element) * rule.weight, reverse=True)
        beta_1 = reduce(lambda x, y: x + (y.matching(fuzzy_element) * y.weight), class_fired_rules, 0) / 2
        c_avg_fact = c_factual(fuzzy_element, rules, new_fdt_predict, lam=0.1, beta=beta_1)
        
        r_th = _robust_threshold(fuzzy_element, rules, new_fdt_predict)
        c_rth_fact = c_factual(fuzzy_element, rules, new_fdt_predict, lam=0.1, beta=r_th)

        m_cf = f_counterfactual(m_fact, fuzzy_element, rules, new_fdt_predict, df_numerical_columns, tau=0.5)
        m_changes, m_rules, m_dist = cf_stats(m_cf, instance, new_fdt_predict, fuzzy_variables, new_fdt)
        m_changes_avg += m_changes
        m_cf_dist_avg += m_dist
        m_cf_rules_avg += m_rules

        mr_cf = f_counterfactual(mr_fact, fuzzy_element, rules, new_fdt_predict, df_numerical_columns, tau=0.5)
        mr_changes, mr_rules, mr_dist = cf_stats(mr_cf, instance, new_fdt_predict, fuzzy_variables, new_fdt)
        mr_changes_avg += mr_changes
        mr_cf_dist_avg += mr_dist
        mr_cf_rules_avg += mr_rules

        c_cf = f_counterfactual(c_fact, fuzzy_element, rules, new_fdt_predict, df_numerical_columns, tau=0.5)
        c_changes, c_rules, c_dist = cf_stats(c_cf, instance, new_fdt_predict, fuzzy_variables, new_fdt)
        c_changes_avg += c_changes
        c_cf_dist_avg += c_dist
        c_cf_rules_avg += c_rules

        cavg_cf = f_counterfactual(c_avg_fact, fuzzy_element, rules, new_fdt_predict, df_numerical_columns, tau=0.5)
        cavg_changes, cavg_rules, cavg_dist = cf_stats(cavg_cf, instance, new_fdt_predict, fuzzy_variables, new_fdt)
        cavg_changes_avg += cavg_changes
        cavg_cf_dist_avg += cavg_dist
        cavg_cf_rules_avg += cavg_rules

        crth_cf = f_counterfactual(c_rth_fact, fuzzy_element, rules, new_fdt_predict, df_numerical_columns, tau=0.5)
        crth_changes, crth_rules, crth_dist = cf_stats(crth_cf, instance, new_fdt_predict, fuzzy_variables, new_fdt)
        crth_changes_avg += crth_changes
        crth_cf_dist_avg += crth_dist
        crth_cf_rules_avg += crth_rules


        i_cf = i_counterfactual(fuzzy_element, rules, new_fdt_predict, df_numerical_columns)
        i_changes, i_rules, i_dist = cf_stats(i_cf, instance, new_fdt_predict, fuzzy_variables, new_fdt)
        i_changes_avg += i_changes
        i_cf_dist_avg += i_dist
        i_cf_rules_avg += i_rules

        num_cf_avg += len(i_cf) # ALL NUM CF ARE THE SAME
        counter += 1

    filename = f"./cf_study/{db}.csv"

    if not os.path.isdir("./cf_study/"):
        os.mkdir('./cf_study')

    with open(filename, 'w+') as file:
        file.write(f'cf_name,n_changes,n_rules\n')


    with open(filename, 'a+') as file:
        file.write(f'i_cf, {i_changes_avg/counter}, {i_cf_rules_avg/counter}\n')
        file.write(f'm_cf, {m_changes_avg/counter}, {m_cf_rules_avg/counter}\n')
        file.write(f'c_cf, {c_changes_avg/counter}, {c_cf_rules_avg/counter}\n')
        file.write(f'c_beta1_cf, {cavg_changes_avg/counter}, {cavg_cf_rules_avg/counter}\n')
        file.write(f'c_beta2_cf, {crth_changes_avg/counter}, {crth_cf_rules_avg/counter}\n')
        file.write(f'mr_cf, {mr_changes_avg/counter}, {mr_cf_rules_avg/counter}\n')
        file.write('--------------------')
        file.write(f'numcf, {num_cf_avg/counter}\n')
    
if __name__ == "__main__":
    results = []
    seed = 256

    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('db')
    parser.add_argument('-q', '--quick', action='store_true')
    args = parser.parse_args()
    run_experiment(args.db)

