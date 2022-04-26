from teacher.datasets import load_compas, load_heloc, load_beer, load_breast, load_pima
from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points, dataset_membership
from teacher.explanation import c_factual
from teacher.tree import FDT
from teacher.explanation._factual import _robust_threshold
from sklearn.model_selection import train_test_split
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


def lambda_beta_study(fdt, X_test, df_test_membership, fact_params, filename):

    rules_tested = len(X_test)
    rules = fdt.to_rule_based_system()
    for i, kwargs in enumerate(fact_params):
        print(f'Testing configuration {i+1}/{len(fact_params)} for params {kwargs}')
        avg_fact_length = 0
        avg_q_multiple = 0
        avg_not_robust = 0

        for fuzzy_element_idx in range(rules_tested):
            instance = X_test.iloc[fuzzy_element_idx].to_numpy().reshape(1, -1)
            fuzzy_element = _get_fuzzy_element(df_test_membership, fuzzy_element_idx)
            target = fdt.predict(instance)
            fact = c_factual(fuzzy_element, rules, target, **kwargs)
            avg_fact_length += len(fact)
            avg_not_robust += check_robustness(fuzzy_element, target, rules, fact)
            if len(fact) > 1:
                avg_q_multiple += 1

        avg_fact_length /= rules_tested
        avg_q_multiple /= rules_tested
        avg_not_robust /= rules_tested

        # print(f'{kwargs=}, {avg_fact_length=}, {avg_q_multiple=}, {avg_not_robust=}')

        with open(filename, 'a+') as file:
            file.write(f'{kwargs};{avg_fact_length};{avg_q_multiple};{avg_not_robust}\n')
        

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
    df_numerical_columns = dataset['continuous']

    X_num = X[dataset['continuous']]
    num_cols = X_num.columns
    fuzzy_points = get_fuzzy_points('entropy', num_cols, X_num, y)

    discrete_fuzzy_values = {col: df[col].unique() for col in df_categorical_columns}
    fuzzy_variables_order = {col: i for i, col in enumerate(X.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)

    df_train_membership = dataset_membership(X_train, fuzzy_variables)
    df_test_membership = dataset_membership(X_test, fuzzy_variables)

    all_classes = dataset['possible_outcomes']

    print(f'Training FDT')
    fdt = FDT(fuzzy_variables)
    fdt.fit(X_train, y_train)


    # experiments = [(c_factual, (round(x*0.01,2),)) for x in range(0,100, 5)]
    # experiments += [(round(x*0.01,2),round(y*0.01,2)) for x in range(0,100, 5) for y in range(0,100, 5)]

    experiments = [{"lam":round(x*0.01,2)} for x in range(0,100,5)]
    experiments += [{"lam": round(x*0.01,2), "beta": round(y*0.01,2)} for x in range(0,100, 5) for y in range(0,100, 5)]


    filename = f"./lambda_beta_study/{dataset['name']}.csv"

    if not os.path.isdir("./lambda_beta_study/"):
        os.mkdir('./lambda_beta_study')

    with open(filename, 'w+') as file:
        file.write(f'kwargs;length;q_multiple;nr_fact\n')
    
    lambda_beta_study(fdt, X_test, df_test_membership, experiments, filename)

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
