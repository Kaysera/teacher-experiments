# Experiments for the article titled: Factual and counterfactual explanations in fuzzy classification trees
-----------------------
## Installation

-----------------------
## Datasets
The datasets used in the experiment are the following ones:

| DATASETS | PARAMETER |
|----------:|----------:|
| Beer      | ``beer``      |
| Breast    | ``breast``    |
| Compas    | ``compas``    |
| Heloc     | ``heloc``     |
| Pima      | ``pima``      |

-----------------------
## Execution

### Illustrative example

For the illustrative example of Section VII refer to the Jupyter Notebook `illustrative_example.ipynb`

### Experiment 1: Lambda and Beta Study

For the experiment carried out in Section VIII.D.1 and Section VIII.D.2, the experiments are run by the file `lambda_beta_study.py` as follows:
```Bash
$ python lambda_beta_study.py db -q?
```
Where `db` can take any of the values of the datasets previously explained and `-q` is an optional parameter to make a small sample of the database for quick results. The flag `-q` has not been used to generate the tables of the article and it is included for debugging purposes only.

This program will generate a file `lambda_beta_study/db.csv` that has the structure:

|                    kwargs |                length |              q_multiple |                       nr_fact |
|--------------------------:|----------------------:|------------------------:|------------------------------:|
| lambda and beta arguments | length of the factual | more than a single rule | non robust factual percentage |

In order to generate the plots for Table II and Fig. 2, refer to the Jupyter notebook `generate_multiplots.ipynb`.

### Experiment 2: Factual Study
For the experiment carried out in Section VIII.D.3, the experiments are run by the file `factual_study.py` as follows:
```Bash
$ python factual_study.py db -q?
```
Where `db` can take any of the values of the datasets previously explained and `-q` is an optional parameter to make a small sample of the database for quick results. The flag `-q` has not been used to generate the tables of the article and it is included for debugging purposes only.

This program will generate a file `factual_study/db.csv` that has the structure:
|                    fact_name |                length |              q_multiple |                       nr_fact |
|--------------------------:|----------------------:|------------------------:|------------------------------:|
| Name of the factual method | length of the factual | more than a single rule | non robust factual percentage |

In this case, each `db.csv` represents a different column of Tables III and IV, where if `nr_fact > 0` the column is in italics.
### Experiment 3: Counterfactual Study

For the experiment carried out in Section VIII.E, the experiments are run by the file `cf_study.py` as follows:
```Bash
$ python cf_study.py db -q?
```
Where `db` can take any of the values of the datasets previously explained and `-q` is an optional parameter to make a small sample of the database for quick results. The flag `-q` has not been used to generate the tables of the article and it is included for debugging purposes only.

This program will generate a file `cf_study/db.csv` that has the structure:
|                    cf_name |                n_changes |              n_rules |
|--------------------------:|----------------------:|------------------------:|
| Name of the factual method | Number of changes made to the instance | Number of rules tested by the algorithm |

With a last row that represents the number of counterfactuals.

In this case, each `db.csv` represents a different column of Table V, where the last row of the file represents the `NumCF`

### Experiment 4: Study against existing proposals

TBD