import numpy as np
import pandas as pd
from sklearn import datasets

# needed to unpickle!
# from p4 import *
from p4 import DistBasedRepo, P4Regressor
# from p4.p4 import LassoTracer, CompressedSaverHelper, get_time_str, DistBasedRepo
from sklearn.model_selection import train_test_split


# from code.p4.src.p4.p4 import P4Regressor


def get_training_data():
    sys_name = "LLVM"
    attribute = "Performance"
    sys_dir = "/application/Distance-Based_Data/SupplementaryWebsite"
    cfg_sys = DistBasedRepo(sys_dir, sys_name, attribute=attribute)
    configs = pd.DataFrame(list(cfg_sys.all_configs.keys()))
    config_attrs = pd.DataFrame(list(cfg_sys.all_configs.values()))

    # if available, we could use available option names. skipping for this example.
    pos_map = {ft: idx for idx, ft in enumerate(list(configs.columns))}

    df_configs = pd.concat([configs, config_attrs], axis=1)
    all_xs = np.array(df_configs.iloc[:, :-1])
    all_ys = list(df_configs.iloc[:, -1])
    return all_xs, all_ys


def main():
    # all_xs, all_ys = get_training_data()
    all_xs, all_ys = datasets.load_diabetes(return_X_y=True)
    feature_names = datasets.load_diabetes().feature_names
    n_train = 0.15  # 5%
    train_x, eval_test_x, train_y, eval_test_y = train_test_split(all_xs, all_ys, train_size=n_train)
    print("Absolute training set size of", len(train_y))

    reg = P4Regressor("/tmp/p4/results-debugging")
    print("Start fitting")
    # fit without feature names first
    reg.fit(train_x, train_y)
    # fit with feature names
    reg.fit(train_x, train_y, feature_names=feature_names)
    n_raw_samples = 1000
    y_samples = reg.predict(eval_test_x, n_raw_samples)
    print(reg.score(all_xs, all_ys))
    print(reg.coef_)
    reg.coef_samples_
    reg.coef_ci(0.5)
    y_scalar = reg.predict(eval_test_x)
    y_cis = reg.predict(eval_test_x, n_raw_samples, ci=0.95)  # 95-% confidence intervals


if __name__ == "__main__":
    main()
