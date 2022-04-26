from teacher.datasets import load_compas, load_heloc, load_beer, load_breast, load_pima
from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points, dataset_membership
from teacher.explanation import m_factual, c_factual, mr_factual
from teacher.tree import FDT
from teacher.explanation._factual import _robust_threshold, _fired_rules, _get_class_fired_rules
from sklearn.model_selection import train_test_split
from functools import reduce
import argparse
import sys
import os

import random
import numpy as np

DATASETS = {
    'beer':  load_beer,
    'breast':  load_breast,
    'compas': load_compas,
    'heloc': load_heloc,
    'pima': load_pima
}


def _get_fuzzy_element(fuzzy_X, idx):
    element = {}
    for feat in fuzzy_X:
        element[feat] = {}
        for fuzzy_set in fuzzy_X[feat]:
            element[feat][fuzzy_set] = fuzzy_X[feat][fuzzy_set][idx]

    return element


def check_robustness(instance, target, rule_list, factual):
    r_th = _robust_threshold(instance, rule_list, target)
    fact_AD = 0
    for rule in factual:
        fact_AD += rule.matching(instance) * rule.weight
    
    # RETURNS 0 IF IT IS ROBUST, 1 OTHERWISE
    if fact_AD >= r_th:
        return 0
    else:
        return 1


def factual_study(fdt, X_test, df_test_membership, filename):

    rules_tested = len(X_test)
    rules = fdt.to_rule_based_system()
    m_fact_length = 0
    m_q_multiple = 0
    m_not_robust = 0

    mr_fact_length = 0
    mr_q_multiple = 0
    mr_not_robust = 0

    c_fact_length = 0
    c_q_multiple = 0
    c_not_robust = 0

    c_avg_fact_length = 0
    c_avg_q_multiple = 0
    c_avg_not_robust = 0

    c_rth_fact_length = 0
    c_rth_q_multiple = 0
    c_rth_not_robust = 0


    for fuzzy_element_idx in range(rules_tested):
        print(f'Testing instance {fuzzy_element_idx+1}/{rules_tested}')
        instance = X_test.iloc[fuzzy_element_idx].to_numpy().reshape(1, -1)
        fuzzy_element = _get_fuzzy_element(df_test_membership, fuzzy_element_idx)
        target = fdt.predict(instance)
        m_fact = m_factual(fuzzy_element, rules, target)
        mr_fact = mr_factual(fuzzy_element, rules, target)
        c_fact = c_factual(fuzzy_element, rules, target, lam=0.1)
        fired_rules = _fired_rules(fuzzy_element, rules)
        class_fired_rules = _get_class_fired_rules(fired_rules, target)
        class_fired_rules.sort(key=lambda rule: rule.matching(fuzzy_element) * rule.weight, reverse=True)
        beta_1 = reduce(lambda x, y: x + (y.matching(fuzzy_element) * y.weight), class_fired_rules, 0) / 2
        c_avg_fact = c_factual(fuzzy_element, rules, target, lam=0.1, beta=beta_1)
        
        beta_2 = _robust_threshold(fuzzy_element, rules, target)
        c_rth_fact = c_factual(fuzzy_element, rules, target, lam=0.1, beta=beta_2)


        m_fact_length += len(m_fact)
        m_not_robust += check_robustness(fuzzy_element, target, rules, m_fact)
        if len(m_fact) > 1:
            m_q_multiple += 1

        mr_fact_length += len(mr_fact)
        mr_not_robust += check_robustness(fuzzy_element, target, rules, mr_fact)
        if len(mr_fact) > 1:
            mr_q_multiple += 1

        c_fact_length += len(c_fact)
        c_not_robust += check_robustness(fuzzy_element, target, rules, c_fact)
        if len(c_fact) > 1:
            c_q_multiple += 1

        c_avg_fact_length += len(c_avg_fact)
        c_avg_not_robust += check_robustness(fuzzy_element, target, rules, c_avg_fact)
        if len(c_avg_fact) > 1:
            c_avg_q_multiple += 1

        c_rth_fact_length += len(c_rth_fact)
        c_rth_not_robust += check_robustness(fuzzy_element, target, rules, c_rth_fact)
        if len(c_rth_fact) > 1:
            c_rth_q_multiple += 1

    m_fact_length /= rules_tested
    m_q_multiple /= rules_tested
    m_not_robust /= rules_tested

    mr_fact_length /= rules_tested
    mr_q_multiple /= rules_tested
    mr_not_robust /= rules_tested

    c_fact_length /= rules_tested
    c_q_multiple /= rules_tested
    c_not_robust /= rules_tested

    c_avg_fact_length /= rules_tested
    c_avg_q_multiple /= rules_tested
    c_avg_not_robust /= rules_tested

    c_rth_fact_length /= rules_tested
    c_rth_q_multiple /= rules_tested
    c_rth_not_robust /= rules_tested

    with open(filename, 'a+') as file:
        file.write(f'm_fact;{m_fact_length};{m_q_multiple};{m_not_robust}\n')
        file.write(f'c_fact;{c_fact_length};{c_q_multiple};{c_not_robust}\n')
        file.write(f'c_beta_1;{c_avg_fact_length};{c_avg_q_multiple};{c_avg_not_robust}\n')
        file.write(f'c_beta_2;{c_rth_fact_length};{c_rth_q_multiple};{c_rth_not_robust}\n')
        file.write(f'mr_fact;{mr_fact_length};{mr_q_multiple};{mr_not_robust}\n')
        

def explain_experiments(ds, seed, quick=False):
    print(f'Preparing dataset {ds}')
    try:
        dataset = DATASETS[ds]()
    except KeyError:
        print('Dataset not found')
        sys.exit()

    df = dataset['df']
    class_name = dataset['class_name']

    # SAMPLING TO MAKE THINGS QUICK
    if quick:
        df = df.sample(n=100, random_state=seed)


    X = df.drop(class_name, axis=1)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    df_categorical_columns = dataset['discrete']
    class_name = dataset['class_name']
    df_categorical_columns.remove(class_name)

    X_num = X[dataset['continuous']]
    num_cols = X_num.columns
    fuzzy_points = get_fuzzy_points('entropy', num_cols, X_num, y)

    discrete_fuzzy_values = {col: df[col].unique() for col in df_categorical_columns}
    fuzzy_variables_order = {col: i for i, col in enumerate(X.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)

    df_test_membership = dataset_membership(X_test, fuzzy_variables)

    print(f'Training FDT')
    fdt = FDT(fuzzy_variables)
    fdt.fit(X_train, y_train)


    filename = f"./factual_study/{dataset['name']}.csv"

    if not os.path.isdir("./factual_study/"):
        os.mkdir('./factual_study')

    with open(filename, 'w+') as file:
        file.write(f'fact_name;length;q_multiple;nr_fact\n')
    
    factual_study(fdt, X_test, df_test_membership, filename)

if __name__ == "__main__":
    results = []
    seed = 42

    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('db')
    parser.add_argument('-q', '--quick', action='store_true')
    args = parser.parse_args()

    explain_experiments(args.db, seed, args.quick)
