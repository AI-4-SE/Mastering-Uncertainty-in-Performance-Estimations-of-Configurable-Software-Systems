# import networkx as nx
import argparse
import copy
import datetime
import itertools
import time
from math import sqrt
import os
import platform
import math
from string import ascii_lowercase

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma
import pymc3 as pm
import seaborn as sns
from pymc3 import stats as stats
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoLars, LassoLarsCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import cauchy, norm

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.metrics.regression import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from theano import shared
from activesampler import systems
from activesampler.multiproxy import SaverHelper
from activesampler.multiproxy import get_time_str
from pymc3.variational.callbacks import CheckParametersConvergence
import theano.tensor as T
import bz2
import pickle
import sys
import gc

T.config.compute_test_value = 'ignore'


class CompressedSaverHelper(SaverHelper):
    def store_pickle(self, obj, f_name, folder='.'):
        f_name_clean = self.get_clean_file_name(f_name)
        current_folder = self.safe_folder_join(self.get_cwd(), folder)
        print("storing compressed pickle")
        file_pickle = os.path.join(current_folder, 'results-{}.pc'.format(f_name_clean))
        with bz2.open(file_pickle, 'wb') as f:
            pickle.dump(obj, f)
        print("storing complete.")
        abs_path = os.path.abspath(file_pickle)
        return abs_path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    method_list = ("mcmc", "advi", "lasso")
    parser = argparse.ArgumentParser()

    parser.add_argument("--sys-dir",
                        help="specifies the parent folder for all config sys",
                        type=str, required=False, default="/application/Distance-Based_Data/SupplementaryWebsite/",
                        )
    parser.add_argument("--sys-name",
                        help="name of the conf sys",
                        type=str
                        )
    parser.add_argument("--attribute",
                        help="specifies the attribute name within the system config xml",
                        type=str, default=None,
                        )
    parser.add_argument("--method",
                        help="which method to use for prob prog",
                        type=str, choices=method_list, default=method_list[0],
                        )
    parser.add_argument("--train-size",
                        help="size of train set either relative 0-1 or absolute when >1",
                        type=int,
                        )
    parser.add_argument("--mcmc-tune",
                        help="tune steps if mcmc is used as method",
                        type=int, default=1000,
                        )
    parser.add_argument("--mcmc-cores",
                        help="number of cores to use if mcmc is used as method",
                        type=int, default=3,
                        )
    parser.add_argument("--mcmc-samples",
                        help="mcmc sampling steps if mcmc is used as method",
                        type=int, default=3000,
                        )
    parser.add_argument("--p-mass",
                        help="probability mass that needs to be either positive or negative to be considered for higher interactions",
                        type=float, default=0.8,
                        )
    parser.add_argument("--advi-its",
                        help="number of iterations if advi is used as method",
                        type=int, default=100000,
                        )
    parser.add_argument("--advi-trace-draws",
                        help="number of draws from fitted model to form a trace if advi is used as method",
                        type=int, default=5000,
                        )
    parser.add_argument("--folder",
                        help="parent folder for results", required=False,
                        type=str, default="/results/last-inference/",
                        )
    parser.add_argument("--active-sampler",
                        help="git dir for activesampler",
                        type=str, default=None,
                        )
    parser.add_argument("--t",
                        help="used pre-difined sampling set for t-wise samples",
                        type=int, default=None
                        )
    parser.add_argument("--inters-between-influential-and-all-fts",
                        help="generates interation pairs not only for all influential features, but also ",
                        type=int, default=None
                        )
    parser.add_argument("--rnd",
                        help="random seed",
                        type=int,
                        )
    parser.add_argument("--no-plots",
                        help="prevents storage of plots",
                        action='store_true'
                        )
    parser.add_argument("--relative-error",
                        help="models relative aleatoric errors",
                        type=str2bool, default=False, const=True, nargs='?',
                        )
    parser.add_argument("--absolute-error",
                        help="models absolute aleatoric errors",
                        type=str2bool, default=True, const=True, nargs='?',
                        )

    ###############################

    args = parser.parse_args()

    attribute = args.attribute
    sys_name = args.sys_name
    sys_dir = args.sys_dir

    method = args.method
    train_size = args.train_size
    t_wise = args.t
    assert train_size or t_wise and not (train_size and t_wise), "chose either train_size or t"
    rnd_seed = args.rnd
    mcmc_tune = args.mcmc_tune
    mcmc_cores = args.mcmc_cores
    mcmc_samples = args.mcmc_samples
    p_mass = args.p_mass
    no_plots = args.no_plots
    advi_its = args.advi_its
    relative_error = args.relative_error
    absolute_error = args.absolute_error
    advi_trace_draws = args.advi_trace_draws
    folder = args.folder
    saver = CompressedSaverHelper(folder, dpi=150, fig_pre='')
    time_str = get_time_str()
    session_dir_name = "-".join([time_str, sys_name, attribute])
    saver.set_session_dir(session_dir_name)

    meta_dict = vars(args)
    meta_dict["system"] = platform.uname()
    saver.store_dict(meta_dict, "args")
    np.random.seed(rnd_seed)

    cfg_sys = DistBasedRepo(sys_dir, sys_name, attribute=attribute)

    pos_map = dict(cfg_sys.position_map)
    ft_names_ordered = np.array(list(pos_map.keys()))[np.argsort(list(pos_map.values()))]
    configs = pd.DataFrame(list(cfg_sys.all_configs.keys()), columns=ft_names_ordered)
    config_attrs = pd.DataFrame(list(cfg_sys.all_configs.values()), columns=['<y>'])
    pos_map = {ft: idx for idx, ft in enumerate(list(configs.columns))}
    df_configs = pd.concat([configs, config_attrs], axis=1)
    all_xs = np.array(df_configs.iloc[:, :-1])
    all_ys = list(df_configs.iloc[:, -1])

    if t_wise:
        train_x, train_y, eval_x, eval_y = cfg_sys.get_train_eval_split(t_wise)
        train_x_df = pd.DataFrame(train_x)
        eval_x_df = pd.DataFrame(eval_x)
    else:
        n_train = float(train_size) if train_size < 1 else int(train_size)
        train_x, eval_test_x, train_y, eval_test_y = train_test_split(all_xs, all_ys, train_size=n_train)
        eval_x, test_x, eval_y, test_y = train_test_split(eval_test_x, eval_test_y, train_size=0.7)

    feature_names = list(pos_map)
    ', '.join(feature_names)
    print_baseline_perf(train_x, train_y, eval_x, eval_y)

    start_experiment_time = time.time()
    if method == "lasso":
        internal_method = "mcmc"
        tracer = LassoTracer(saver, conf_sys=cfg_sys, inters_only_between_influentials=True, no_plots=no_plots,
                             relative_obs_noise=relative_error, absolute_obs_noise=absolute_error, t_wise=t_wise)
        tracer.fit(train_x, train_y, feature_names=feature_names, pos_map=pos_map, attribute=attribute,
                   method=internal_method,
                   mcmc_tune=mcmc_tune, mcmc_cores=mcmc_cores, tree_depth=200, mcmc_samples=mcmc_samples,
                   p_mass=p_mass,
                   advi_its=advi_its, advi_trace_draws=advi_trace_draws)
    else:
        tracer = Tracer(saver, conf_sys=cfg_sys, inters_only_between_influentials=True, no_plots=no_plots,
                        relative_obs_noise=relative_error, absolute_obs_noise=absolute_error, t_wise=t_wise)
        tracer.fit(train_x, train_y, feature_names=feature_names, pos_map=pos_map, attribute=attribute,
                   method=method,
                   mcmc_tune=mcmc_tune, mcmc_cores=mcmc_cores, tree_depth=200, mcmc_samples=mcmc_samples, p_mass=p_mass,
                   advi_its=advi_its, advi_trace_draws=advi_trace_draws)
    end_time = time.time()
    diff = end_time - start_experiment_time
    print(f'Finished within {int(diff // 60)}m{int(diff % 60)}s')

    sys.stdout.flush()
    print("Storing Tracer Object")
    del tracer.X
    del tracer.y

    # this stores the trained model
    saver.store_pickle(tracer, "tracer")


def set_zero_to_min(arr):
    np_arr = np.array(arr)
    mask = np_arr != 0.0
    non_zero_min = min(np_arr[mask])
    r = []
    for value in arr:
        if value == 0.0:
            r.append(non_zero_min)
        else:
            r.append(value)
    return r


def update_fig_ax_titles(fig, ft_names_and_root):
    for ax, new_title in zip(fig.axes, ft_names_and_root):
        ax.set_title(new_title)


class Tracer(BaseEstimator, RegressorMixin):
    def __init__(self, saver, conf_sys=None, x_eval=None, y_eval=None, inters_only_between_influentials=True,
                 no_plots=False, snapshot_err_scores=False, prior_broaden_factor=1, absolute_obs_noise=True,
                 relative_obs_noise=False, t_wise=None):

        self.t_wise = t_wise
        self.prior_spectrum_cost = None
        self.prior_broaden_factor = prior_broaden_factor
        self.final_var_names = None
        self.absolute_obs_noise = absolute_obs_noise
        self.relative_obs_noise = relative_obs_noise
        self.saver = saver
        self.prediction_sample_size = 1000
        self.no_plots = no_plots
        self.X = None
        self.y = None
        self.train_data = None
        self.feature_names = None
        self.pos_map = None
        self.best_front = None
        self.top_candidate_lr = None
        self.top_candidate = None
        self.final_model = None
        self.final_trace = None
        self.MAP = None
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.x_shared = None
        self.y_shared = None
        self.styles = ('-', '--', '-.', ':')
        self.history = {}
        self.models = {}
        self.fitting_times = {}
        self.total_experiment_time = None
        self.weighted_errs_per_sample = None
        self.weighted_rel_errs_per_sample = None
        self.snapshot_err_scores = snapshot_err_scores
        self.inters_only_between_influentials = inters_only_between_influentials

        print("Using absolute measurement error: {} | relative measurement error: {}".format(absolute_obs_noise,
                                                                                             relative_obs_noise))

    def set_eval_values(self):
        self.x_shared.set_value(self.x_eval)
        self.y_shared.set_value(self.y_eval)

    def fit(self, X, y, feature_names, pos_map, attribute='unknown-attrib', method="advi", mcmc_tune=2000,
            mcmc_cores=5, tree_depth=200, mcmc_samples=15000, p_mass=0.8, advi_its=100000, advi_trace_draws=5000):
        fit_start = time.time()
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.pos_map = pos_map
        noise_str = 'noise'

        self.x_shared = shared(np.array(self.X))
        self.y_shared = shared(np.array(self.y))
        lin_reg_features = []
        root_beta = 10 ** 1
        coef_sd = None
        noise_sd = None

        seed_lin = np.random.randint(0, 10 ** 5)
        seed_inter = np.random.randint(0, 10 ** 5)
        seed_final = np.random.randint(0, 10 ** 5)

        stage_name_lin = "it-1"
        lin_start = time.time()
        linear_model, lin_reg_features, lin_trace = self.get_and_fit_model(advi_its, advi_trace_draws, feature_names,
                                                                           [], mcmc_cores, mcmc_samples,
                                                                           mcmc_tune, method, noise_str, pos_map,
                                                                           root_beta, seed_lin, tree_depth, coef_sd,
                                                                           noise_sd)
        lin_end = time.time()
        self.fitting_times[stage_name_lin] = lin_end - lin_start

        print_flush("Getting influential terms from model")
        first_stage_influential_ft, first_stage_influential_inters = self.get_ft_and_inters_from_rvs(lin_trace,
                                                                                                     noise_str, p_mass,
                                                                                                     lin_reg_features)
        print_flush("Starting snapshot")
        lin_snapshot = self.construct_snapshot(first_stage_influential_ft, first_stage_influential_inters,
                                               linear_model, lin_reg_features, lin_trace, stage_name_lin)
        self.history[stage_name_lin] = lin_snapshot
        model_trace_dict = get_model_trace_dict(lin_trace, linear_model)
        self.models[stage_name_lin] = model_trace_dict
        print_flush("Finished snapshot")
        print("Influential features: ", first_stage_influential_ft)
        sys.stdout.flush()
        only_feature_wise_data = len(self.X) == len(self.X[0])
        first_stage_new_inters = self.generate_valid_combinations(first_stage_influential_ft, lin_reg_features)
        no_inters = len(first_stage_new_inters) == 0

        finish_early = False
        if only_feature_wise_data:
            print(
                "Returning early because there is only featurewise training data, so interactions cannot be inferred.")
            print("Influential features were:", first_stage_influential_ft)
            finish_early = True
        if no_inters:
            print("Returning early because there are no interactions between " +
                  "influential features that on and off at least once.")
            print("Influential features were:", first_stage_influential_ft)
            finish_early = True

        t1_given = np.array(self.X).shape[1] + len(first_stage_new_inters) >= np.array(self.X).shape[0]
        if t1_given:
            print("Returning early because interactions may be too many " +
                  "to learn with given t=1-wise training set.")
            print("Influential features were:", first_stage_influential_ft)
            print(np.array(self.X).shape[1], "feaures,", len(first_stage_new_inters), "possible interactions,",
                  np.array(self.X).shape[0], "samples")
            finish_early = True

        if finish_early:
            self.final_trace = lin_trace
            self.final_model = linear_model
            print()
            sys.stdout.flush()
            fit_end = time.time()
            self.total_experiment_time = fit_end - fit_start
            return

        stage_name_inter = "it-2"
        print_flush("Starting it-2")
        inter_start = time.time()
        inter_model, inter_reg_features, inter_trace = self.get_and_fit_model(advi_its, advi_trace_draws, feature_names,
                                                                              first_stage_new_inters,
                                                                              mcmc_cores,
                                                                              mcmc_samples, mcmc_tune, method,
                                                                              noise_str, pos_map, root_beta, seed_inter,
                                                                              tree_depth, coef_sd, noise_sd)
        inter_end = time.time()
        self.fitting_times[stage_name_inter] = inter_end - inter_start
        stage_2_influential_fts, stage_2_influential_inters = self.get_ft_and_inters_from_rvs(inter_trace, noise_str,
                                                                                              p_mass,
                                                                                              inter_reg_features)
        print("Starting snapshot")
        snapshot = self.construct_snapshot(stage_2_influential_fts, stage_2_influential_inters, inter_model,
                                           inter_reg_features, inter_trace,
                                           stage_name_inter)
        self.history[stage_name_inter] = snapshot
        model_trace_dict = get_model_trace_dict(inter_trace, inter_model)
        self.models["it-2"] = model_trace_dict
        print("Finished snapshot")
        print("Final features: ", stage_2_influential_fts)
        print("Final interactions: ", stage_2_influential_inters)

        stage_name_final = "it-final"
        final_start = time.time()
        final_model, final_reg_features, final_trace = self.get_and_fit_model(advi_its, advi_trace_draws, feature_names,
                                                                              stage_2_influential_inters, mcmc_cores,
                                                                              mcmc_samples,
                                                                              mcmc_tune, method, noise_str, pos_map,
                                                                              root_beta, seed_final, tree_depth,
                                                                              coef_sd, noise_sd)
        final_end = time.time()
        self.fitting_times[stage_name_final] = final_end - final_start
        print("Starting snapshot")
        snapshot = self.construct_snapshot(stage_2_influential_fts, stage_2_influential_inters, final_model,
                                           final_reg_features, final_trace,
                                           stage_name=stage_name_final)
        self.history[stage_name_final] = snapshot
        model_trace_dict = get_model_trace_dict(final_trace, final_model)
        self.models["it-final"] = model_trace_dict
        print_flush("Finished snapshot")
        self.final_model = final_model
        self.final_trace = final_trace
        fit_end = time.time()
        self.total_experiment_time = fit_end - fit_start

    def generate_valid_combinations(self, first_stage_influential_ft, all_ft):
        print_flush("Generating Interaction Terms")
        if self.inters_only_between_influentials:
            all_inter_pairs = list(itertools.combinations(first_stage_influential_ft, 2))
        else:
            all_inter_pairs = list(itertools.product(first_stage_influential_ft, all_ft))
        valid_pairs = []
        print("Computing x values for", len(all_inter_pairs), "interactions")
        sys.stdout.flush()
        for a, b in all_inter_pairs:
            idx_a = self.pos_map[a]
            idx_b = self.pos_map[b]
            x_np = self.x_shared.get_value()
            vals_a_np = np.array(list(x_np[:, idx_a]))
            vals_b_np = np.array(list(x_np[:, idx_b]))
            is_non_constant = self.not_constant_term_cheap(vals_a_np, vals_b_np, x_np)
            if is_non_constant:
                valid_pairs.append((a, b))
        print_flush("Checked all interactions for constance ones.")
        return valid_pairs

    def not_constant_term(self, vals_a_np, vals_b_np):
        vals_prod = np.multiply(vals_a_np, vals_b_np)
        is_non_constant = len(np.unique(vals_prod)) > 1
        return is_non_constant

    def not_constant_term_cheap(self, vals_a_np, vals_b_np, train_set, slice_size=500):
        n_samples = len(vals_a_np)
        is_non_constant = False
        duplicates_any_features = True
        for slice_start in range(0, n_samples, slice_size):
            slice_end = min(slice_start + slice_size, n_samples)
            small_a = vals_a_np[slice_start:slice_end]
            small_b = vals_b_np[slice_start:slice_end]
            small_train_set = train_set[slice_start:slice_end]
            vals_prod = np.multiply(small_a, small_b)

            if duplicates_any_features:
                duplicates_any_features = np.any(
                    [np.all(vals_prod == small_train_set[:, i]) for i in range(small_train_set.shape[1])])

            if not is_non_constant:
                is_non_constant = len(np.unique(vals_prod)) > 1
            if is_non_constant and not duplicates_any_features:
                return True
        return False

    def vector_inner_prod_slow(self, a, b, slice_size=1000):
        length = len(a)
        products = []
        for slice_start in range(0, length, slice_size):
            slice_end = min(slice_start + slice_size, length)
            small_a = a[slice_start:slice_end]
            small_b = b[slice_start:slice_end]
            vals_prod = np.prod([small_a, small_b], axis=0)
            products.append(vals_prod)
        joint_prod = np.concatenate(products)
        return joint_prod

    def get_ft_and_inters_from_rvs(self, inter_trace, noise_str, p_mass, ft_inter_names):
        significant_inter_ft = self.get_significant_fts(inter_trace, noise_str, p_mass, ft_inter_names)
        ft_or_inter = [a.replace("influence_", "").split("&") for a in significant_inter_ft if "root" not in a]
        final_ft = [ft[0] for ft in ft_or_inter if len(ft) == 1]
        final_inter = [ft for ft in ft_or_inter if len(ft) > 1]
        return final_ft, final_inter

    def construct_snapshot(self, influential_ft_list, influential_inter_list, model, reg_fts, trace, stage_name):
        reg_dict_final, err_dict = self.get_reg_dict(reg_fts)
        err_dict = {lr: remove_raw_field(errs) for lr, errs in err_dict.items()}
        self.trace_plots(trace, pre=stage_name, model_dict=reg_dict_final, ft_names=reg_fts)
        if self.snapshot_err_scores:
            print("Snapshotting Error Scores")
            y_pred = self.predict(self.X, model=model, trace=trace)
            inter_errors = get_err_dict_from_predictions(y_pred, self.X, self.y)

            inter_errors = remove_raw_field(inter_errors)
        else:
            print("Skipping Error Scores")
            inter_errors = {}
        trace_errs = {}
        snapshot = get_snapshot_dict(reg_dict_final, err_dict, inter_errors, reg_fts, model,
                                     influential_ft_list + influential_inter_list, trace, trace_errs)
        return snapshot

    def get_and_fit_model(self, advi_its, advi_trace_draws, feature_names, inter_pairs, mcmc_cores, mcmc_samples,
                          mcmc_tune, method, noise_str, pos_map, root_sd, rnd_seed, tree_depth, coef_sd=10,
                          noise_sd=10):
        rv_names = list(feature_names)
        train_data = self.x_shared.eval()
        observed_y = self.y_shared
        print_flush("Generating interactions.")
        start = time.time()

        inter_strs = []

        if inter_pairs is not None and len(inter_pairs) > 0:
            for i, (a, b) in enumerate(inter_pairs):
                idx_a = self.pos_map[a]
                idx_b = self.pos_map[b]
                x_np = self.x_shared.get_value()
                vals_a_np = x_np[:, idx_a]
                vals_b_np = x_np[:, idx_b]
                is_non_constant = self.not_constant_term_cheap(vals_a_np, vals_b_np, x_np)
                inter_combi_str = "{}&{}".format(a, b)
                print("Interaction", inter_combi_str, "is constant?", (not is_non_constant))
                sys.stdout.flush()

                np.alltrue(np.multiply(vals_a_np, vals_b_np) == vals_a_np)
                if is_non_constant:
                    print_flush("Computing Sliced Product")
                    vals_prod = self.vector_inner_prod_slow(vals_a_np, vals_b_np)
                    vals_prod_2d = vals_prod.reshape((-1, 1))
                    print_flush("Concatenating new Feature to Train Set")
                    train_data = T.concatenate([train_data, vals_prod_2d], axis=1)
                    print_flush("Finished Concat")
                    if i % 100 == 0:
                        train_shape = train_data.shape.eval()
                        print("New size:", train_shape, "Total entries:",
                              np.prod(train_shape), "Bytes:", train_data.eval().nbytes)
                        sys.stdout.flush()
                    rv_names.append(inter_combi_str)
                    inter_str = "influence_{}".format(inter_combi_str)
                    inter_strs.append(inter_str)
                    gc.collect()
                    train_data = train_data.eval()

        end = time.time()
        print()
        print("Finished generating interactions in {0:9.1} minutes.".format((end - start) / 60))
        print()
        sys.stdout.flush()

        pm_model, pred, err_mean, err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample = self.get_pm_model(
            rv_names, train_data)
        new_trace = self.fit_pm_model(advi_its, advi_trace_draws, mcmc_cores, mcmc_samples, mcmc_tune, method, err_mean,
                                      err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample, pm_model, pred,
                                      rnd_seed)
        return pm_model, rv_names, new_trace

    def get_priors_from_lin_reg(self, rv_names, sd_scale=None):
        if sd_scale is None:
            sd_scale = self.prior_broaden_factor
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum(rv_names)
        if not self.no_plots:
            self.save_spectrum_fig(reg_dict_final, err_dict, rv_names)

        all_raw_errs = [errs['raw'] for errs in list(err_dict.values())]
        all_abs_errs = np.array([abs(err['y_pred'] - err['y_true']) for err in all_raw_errs])

        noise_sd_over_all_regs = sd_scale * 2 * float(all_abs_errs.mean())
        reg_list = list(reg_dict_final.values())
        alphas = []
        betas = []
        for coef_id, _ in enumerate(reg_list[0].coef_):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list])
            alpha, beta = norm.fit(coef_candidates)
            alpha = max(coef_candidates)
            alphas.append(alpha)
            betas.append(beta)
        prior_coef_means = np.array(alphas)
        prior_coef_stdvs = np.array(betas)
        prior_root_mean, prior_root_std = norm.fit(np.array([reg.intercept_ for reg in reg_list]))

        return noise_sd_over_all_regs, prior_coef_means, prior_coef_stdvs, prior_root_mean, prior_root_std

    def get_prior_weighted_normal(self, rv_names, gamma=1, stddev_multiplier=1):
        if stddev_multiplier is None:
            stddev_multiplier = self.prior_broaden_factor
        print("Getting priors from lin regs.")
        reg_dict_final, err_dict = self.get_regression_spectrum(rv_names)
        if not self.no_plots:
            self.save_spectrum_fig(reg_dict_final, err_dict, rv_names)
        all_raw_errs = [errs['raw'] for errs in list(err_dict.values())]
        all_abs_errs = np.array([abs(err['y_pred'] - err['y_true']) for err in all_raw_errs])
        mean_abs_errs = all_abs_errs.mean(axis=1)
        all_rel_errs = np.array([abs((err['y_pred'] - err['y_true']) / err['y_true']) for err in all_raw_errs])
        mean_rel_errs = all_rel_errs.mean(axis=1)
        reg_list = list(reg_dict_final.values())

        means_weighted = []
        stds_weighted = []
        weights = 1 - MinMaxScaler().fit_transform(np.atleast_2d(mean_abs_errs).T).ravel()
        err_mean, err_std = weighted_avg_and_std(mean_abs_errs, weights, gamma=gamma)
        noise_sd_over_all_regs = err_mean + 3 * err_std
        root_candidates = np.array([reg.intercept_ for reg in reg_list])
        root_mean, root_std = weighted_avg_and_std(root_candidates, weights, gamma=gamma)
        for coef_id, coef in enumerate(rv_names):
            coef_candidates = np.array([reg.coef_[coef_id] for reg in reg_list])
            mean_weighted, std_weighted = weighted_avg_and_std(coef_candidates, weights, gamma=gamma)
            means_weighted.append(mean_weighted)
            stds_weighted.append(stddev_multiplier * std_weighted)

        weighted_errs_per_sample = np.average(all_abs_errs, axis=0, weights=mean_abs_errs)
        weighted_rel_errs_per_sample = np.average(all_rel_errs, axis=0, weights=mean_rel_errs)
        return np.array(means_weighted), np.array(stds_weighted), root_mean, root_std, \
               err_mean, err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample

    def get_priors_from_train_set(self, rv_names, sd_scale, n_influentials=10):
        mean_perf = np.mean(self.y)
        expected_ft_mean = mean_perf / n_influentials
        expected_std = expected_ft_mean / n_influentials * sd_scale
        mean_priors = np.array([expected_ft_mean] * len(rv_names))
        std_priors = np.array([expected_std] * len(rv_names))
        noise_sd = expected_ft_mean * sd_scale

        root_mean_prior = mean_perf / 2
        root_std_prior = mean_perf / 2 * sd_scale

        return noise_sd, mean_priors, std_priors, root_mean_prior, root_std_prior

    def get_uninformed_priors_from_train_set(self, rv_names):
        mean_perf = np.mean(self.y)
        expected_ft_mean = 0
        expected_std = mean_perf * 10
        mean_priors = np.array([expected_ft_mean] * len(rv_names))
        std_priors = np.array([expected_std] * len(rv_names))
        noise_sd = expected_std

        root_mean_prior = expected_ft_mean
        root_std_prior = expected_std

        return noise_sd, mean_priors, std_priors, root_mean_prior, root_std_prior

    def get_reg_dict(self, lin_reg_features):
        ridge, err_lr = self.fit_and_eval_lin_reg(lin_reg_features, RidgeCV(cv=3))
        lasso, err_lasso = self.fit_and_eval_lin_reg(lin_reg_features, reg_proto=LassoCV(cv=3))
        net, err_net = self.fit_and_eval_lin_reg(lin_reg_features, reg_proto=ElasticNetCV(cv=3))
        reg_dict = {"ridge": ridge, "lasso": lasso, "elastic net": net}
        err_dict = {"ridge": err_lr, "lasso": err_lasso, "elastic net": err_net}
        return reg_dict, err_dict

    def get_regression_spectrum(self, lin_reg_features, n_steps=50, cv=3, n_jobs=-1):
        start = time.time()
        regs = []
        step_list = np.linspace(0, 1, n_steps)
        for l1_ratio in step_list:
            if 0 < l1_ratio < 1:
                reg_prototype = ElasticNetCV(l1_ratio=l1_ratio, cv=cv, n_jobs=n_jobs)
                reg, err = self.fit_and_eval_lin_reg(lin_reg_features, reg_proto=reg_prototype, verbose=False)
                regs.append((reg, err))
        ridge = RidgeCV(cv=cv)
        lasso = LassoCV(cv=cv, n_jobs=n_jobs)
        for reg in [ridge, lasso]:
            fitted_reg, err = self.fit_and_eval_lin_reg(lin_reg_features, reg_proto=reg, verbose=False)
            regs.append((fitted_reg, err))

        reg_dict = {l1_ratio: tup[0] for tup, l1_ratio in zip(regs, step_list)}
        err_dict = {l1_ratio: tup[1] for tup, l1_ratio in zip(regs, step_list)}

        end = time.time()
        cost = end - start
        self.prior_spectrum_cost = cost
        print("Prior Spectrum Computation took", cost)

        return reg_dict, err_dict

    def get_significant_fts(self, lin_trace, noise_str, p_mass, ft_inter_names):
        significant_ft = []
        for ft in lin_trace.varnames:
            if "_log__" not in ft and "relative_error" not in ft and noise_str not in ft \
                    and "root" not in ft and "active_" not in ft and "_scale" not in ft:
                if lin_trace[ft].shape[1] == 1:
                    mass_less_zero = (lin_trace[ft] > 0).mean()
                    if mass_less_zero > p_mass or mass_less_zero < (1 - p_mass):
                        ft_name = ft.replace("influence_", "")
                        ft_name = ft_name.replace("active_", "")
                        significant_ft.append(ft_name)
                else:
                    # single variabel with shape for all ft_inter_names
                    masses_less_zero = (lin_trace[ft] > 0).mean(axis=0)
                    relevant_mask = np.any([masses_less_zero > p_mass, masses_less_zero < (1 - p_mass)], axis=0)
                    rel_idx = np.nonzero(relevant_mask)
                    significant_ft = np.array(ft_inter_names)[rel_idx]
        return significant_ft

    def predict(self, x, model=None, trace=None):
        prediction_samples = self.predict_raw(x, model=model, trace=trace, )
        predictions = np.median(prediction_samples, axis=0)
        return predictions

    def predict_raw(self, x, model=None, trace=None, sample_size=None):
        if not model:
            model = self.final_model
            trace = self.final_trace
        sample_size = self.prediction_sample_size if sample_size is None else sample_size
        # self.x_shared.set_value(np.array(x))
        transformed_x = self.get_pm_train_data(np.array(x))[1]
        self.train_data.set_value(transformed_x)

        with model:
            ppc = pm.sample_posterior_predictive(trace, samples=sample_size)
        prediction_samples = list(ppc.values())[0]
        return prediction_samples

    def score_uncert(self, x_eval, y_eval, model=None, trace=None):
        y_trace = self.predict_raw(x_eval, model, trace)
        r_dict = {"y_true": y_eval, "trace_pred": y_trace}
        for conf_prob in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
            in_range_ratio = self.calc_confidence_err(conf_prob, y_eval, y_trace)
            print("correct ratio within {}% certainty band: {}%".format(int(conf_prob * 100), in_range_ratio * 100))
            r_dict[conf_prob] = in_range_ratio
        return r_dict

    @staticmethod
    def calc_confidence_err(conf_prob, y_eval, y_trace):
        y_conf = stats.hpd(y_trace, credible_interval=conf_prob)

        pred_in_conf_rande_arr = [y_low < true_y < y_up for true_y, (y_low, y_up) in zip(y_eval, y_conf)]
        in_range_ratio = np.array(pred_in_conf_rande_arr).mean()
        return in_range_ratio

    @staticmethod
    def calc_confidence_closest_mape(conf_prob, y_eval, y_trace):
        y_conf = stats.hpd(y_trace, credible_interval=conf_prob)
        closest_mape = []
        for true_y, (y_low, y_up) in zip(y_eval, y_conf):
            if y_low <= true_y <= y_up:
                mape = 0
            elif y_low > true_y:
                mape = (y_low - true_y) / true_y
            else:
                mape = (true_y - y_up) / true_y
            mape *= 100  # make percent
            closest_mape.append(mape)
        np_mapes = np.array(closest_mape)
        closest_mape = float(np.array(np_mapes).mean())
        closest_mape_if_outside = float(np_mapes[np.nonzero(np_mapes)].mean())
        return closest_mape, closest_mape_if_outside

    def trace_plots(self, trace, pre='', model_dict=None, store_trace=False, ft_names=None):
        if not self.no_plots:
            n_bins = 100
            print("Starting plotting")
            time_template = "%Y-%m-%d_%H%M%S-{}".format(pre)
            file = datetime.datetime.now().strftime(time_template)
            ft_names_and_root = ["root"] + ft_names
            if model_dict is not None:
                print("Storing lr comp plot.")
                reference_val_dict = self.map_lr_coeffs_to_trace_vars(ft_names_and_root, model_dict)
                pm.plot_posterior(trace, point_estimate='mode', ref_val=None, varnames=["root", "influence"],
                                  bins=n_bins)
                fig = plt.gcf()
                update_fig_ax_titles(fig, ft_names_and_root)
                self.add_ref_vals_to_fig(fig, reference_val_dict)
                self.saver.store_figure('lr-comp-{}'.format(file))

            if store_trace:
                print("Storing trace plot.")
                az.plot_trace(trace, max_plots=20)
                fig = plt.gcf()
                update_fig_ax_titles(fig, ft_names_and_root)
                plt.tight_layout()
                self.saver.store_figure('trace-{}'.format(pre))

            print("Storing posterior plot with ref=0.")
            pm.plot_posterior(trace, point_estimate='mode', ref_val=0, bins=n_bins)
            fig = plt.gcf()
            update_fig_ax_titles(fig, ft_names_and_root)
            self.saver.store_figure('posterior-{}'.format(file))

            print("Storing energy plot.")
            pm.energyplot(trace)
            self.saver.store_figure('energy-{}'.format(file))
            print("Finished plotting")

    def map_lr_coeffs_to_trace_vars(self, ft_names, lr, label=None):
        if type(lr) is dict:
            return_dict = {}
            for l, model in lr.items():
                new_coeffs = self.map_lr_coeffs_to_trace_vars(ft_names, model, l)
                return_dict = {**return_dict, **new_coeffs}
        else:
            reference_vals = []
            coeffs = lr.coef_
            root = lr.intercept_
            coef_idx = 0
            filtered_var_names = []
            for v in ft_names:
                if v == "root":
                    coef = root
                    filtered_var_names.append(v)
                else:
                    coef = coeffs[coef_idx]
                    coef_idx += 1
                    filtered_var_names.append(v)

                if coef is not None:
                    reference_vals.append(coef)
            return_dict = {label: reference_vals}
        return return_dict

    def predict_2(self, X):
        X_ = self.transform_data_to_candidate_features(self.top_candidate, X)
        pred = self.top_candidate_lr.predict(X_)
        return pred

    def transform_data_to_candidate_features(self, candidate, train_x):
        mapped_features = []
        for term in candidate:
            idx = [self.pos_map[ft] for ft in term]
            selected_cols = np.array(train_x)[:, idx]
            if len(idx) > 1:
                mapped_feature = np.product(selected_cols, axis=1).ravel()
            else:
                mapped_feature = selected_cols.ravel()
            mapped_features.append(list(mapped_feature))
        reshaped_mapped_x = np.atleast_2d(mapped_features).T
        return reshaped_mapped_x

    def fit_and_eval_lin_reg(self, lin_reg_features, reg_proto=None, verbose=True):
        if not reg_proto:
            reg_proto = Ridge()
        inters = [get_feature_names_from_rv_id(ft_inter_Str) for ft_inter_Str in lin_reg_features]
        x_mapped = self.transform_data_to_candidate_features(inters, self.X)
        lr = copy.deepcopy(reg_proto)
        lr.fit(x_mapped, self.y)
        if verbose:
            print_scores("analogue LR", lr, 'train set', x_mapped, self.y)
        errs = get_err_dict(lr, x_mapped, self.y)
        return lr, errs

    def fit_and_eval_non_param_reg(self, reg_proto, id_str):
        model = copy.deepcopy(reg_proto)
        model.fit(self.X, self.y)
        print_scores(id_str, model, 'train set', self.X, self.y)
        errs = get_err_dict(model, self.X, self.y)
        return model, errs

    def add_ref_vals_to_fig(self, fig, reference_vals: dict):
        axs = fig.axes
        colors = sns.color_palette("colorblind", len(reference_vals))
        num_vars = len(list(reference_vals.values())[0])

        for n, (label, reference) in enumerate(reference_vals.items()):
            style = self.get_line_style(n, colors=colors)
            for i, ax in zip(range(num_vars), axs):
                ref_val = reference[i]
                if ref_val is not None:
                    ax.axvline(x=ref_val, **style, label=label)
        axs[1].legend()

    def get_line_style(self, i, colors):
        color_id = i % len(colors)
        color = colors[color_id]
        dash_style_idx = i % len(self.styles)
        dash_style = self.styles[dash_style_idx]
        style = {
            "linestyle": dash_style,
            "c": color,
            "linewidth": 2}
        return style

    def explain(self):
        return self.history

    def get_pm_model(self, rv_names, train_data):
        prior_coef_means, prior_coef_stdvs, prior_root_mean, \
        prior_root_std, err_mean, err_std, self.weighted_errs_per_sample, self.weighted_rel_errs_per_sample = self.get_prior_weighted_normal(
            rv_names, gamma=3)
        n_features = len(rv_names)
        self.train_data = shared(train_data)
        print_flush("Creating model with {} feature-wise and pair-wise RVs: {}".format(n_features, rv_names))
        start = time.time()
        with pm.Model() as pm_model:
            root = pm.Normal('root', mu=prior_root_mean, sigma=prior_root_std)
            pred = root
            rv_s = pm.Normal("influence",
                             mu=prior_coef_means,
                             sigma=prior_coef_stdvs,
                             shape=n_features)
            ft_prod = T.dot(self.train_data, rv_s)
            pred = pred + ft_prod
        end = time.time()
        print_flush("RVs created in {0:9.1} minutes.".format((end - start) / 60))
        return pm_model, pred, err_mean, err_std, self.weighted_errs_per_sample, self.weighted_rel_errs_per_sample

    def fit_pm_model(self, advi_its, advi_trace_draws, mcmc_cores, mcmc_samples, mcmc_tune, method, err_mean, err_std,
                     weighted_errs_per_sample, weighted_rel_errs_per_sample,
                     pm_model,
                     pred, rnd_seed):
        start = time.time()
        with pm_model:
            processsed_pred = pred
            if self.relative_obs_noise:
                gamma_alpha, gamma_loc, gamma_scale = gamma.fit(weighted_rel_errs_per_sample, floc=0)
                pymc3_gamma_alpha = gamma_alpha
                pymc3_gamma_beta = 1 / gamma_scale
                relative_noise = pm.Gamma("relative_error", alpha=pymc3_gamma_alpha, beta=pymc3_gamma_beta)
                noise_calculated = relative_noise * pm.math.abs_(processsed_pred)
                if not self.absolute_obs_noise:
                    processsed_pred = pm.Normal('y_observed', mu=processsed_pred, sd=noise_calculated,
                                                observed=self.y)

            elif self.absolute_obs_noise:
                abs_mean = np.mean(weighted_errs_per_sample)
                abs_std = np.std(weighted_errs_per_sample)
                processsed_pred = pm.Normal('y_observed', mu=processsed_pred, sd=abs_mean + abs_std,
                                            observed=self.y)

            storing_start = time.time()
            print("Storing Prior Model pickle")
            self.saver.store_pickle(pm_model, "prior-model")
            print("Successfully stored Prior Model pickle")
            storing_cost = time.time() - storing_start

            if method == "advi":
                print_flush("Start ADVI.")
                approx = pm.fit(method='asvgd', n=advi_its, callbacks=[CheckParametersConvergence(diff='relative')],
                                total_grad_norm_constraint=1000.)
                print_flush("Start sampling.")
                new_trace = approx.sample(draws=advi_trace_draws)
            elif method == "mcmc":
                print_flush("Start sampling.")
                new_trace = pm.sample(mcmc_samples, init='advi+adapt_diag', tune=mcmc_tune, cores=mcmc_cores,
                                      chains=mcmc_cores, random_seed=rnd_seed)
        end = time.time()
        print_flush("Done Fitting. Calulating time.")
        total_cost = (end - start) - storing_cost
        print_flush("Fitted model in {0:9.1} minutes.".format((total_cost) / 60))
        return new_trace

    def get_pm_train_data(self, x=None):
        if x is None:
            x = self.x_shared
        vars_and_biases = self.final_var_names
        rv_names = []
        inter_strs = []
        columns = []
        for var, _ in vars_and_biases.items():
            if len(var) == 1:
                var_name = var[0]
                idx_ft = self.pos_map[var_name]
                vals_ft_np = x[:, idx_ft]
                columns.append(vals_ft_np)
                rv_names.append(var_name)
                inter_str = "influence_{}".format(var_name)
                inter_strs.append(inter_str)
            else:
                a, b = var
                idx_a = self.pos_map[a]
                idx_b = self.pos_map[b]
                inter_combi_str = "{}&{}".format(a, b)
                vals_a_np = x[:, idx_a]
                vals_b_np = x[:, idx_b]
                vals_prod = self.vector_inner_prod_slow(vals_a_np, vals_b_np)

                columns.append(vals_prod)
                rv_names.append(inter_combi_str)
                inter_str = "influence_{}".format(inter_combi_str)
                inter_strs.append(inter_str)
        train_data = np.concatenate([c.reshape((-1, 1)) for c in columns], axis=1)
        return rv_names, train_data

    def save_spectrum_fig(self, reg_dict_final, err_dict, rv_names):
        iteration_id = iteration_id = hash(tuple(rv_names))
        print("saving spectrum")
        err_tuples = [(i, errs['r2'], errs['mape'], errs['rmse']) for i, errs in enumerate(list(err_dict.values()))]
        df_errors = pd.DataFrame(err_tuples, columns=["i", 'r2', 'mape', 'rmse'])
        df_melted_errs = pd.melt(df_errors, id_vars=['i'], value_vars=['r2', 'mape', 'rmse'],
                                 var_name='error type', value_name='model error')
        g = sns.FacetGrid(df_melted_errs, col="error type", sharex=False, sharey=False)
        g = g.map(plt.hist, "model error")
        spectrum_id = "spect-{}".format(iteration_id)
        err_id = "{}-errs".format(spectrum_id)
        self.saver.store_figure(err_id)
        coef_list = [(i, reg.intercept_, *list(reg.coef_)) for i, reg in enumerate(reg_dict_final.values())]
        coef_cols = ['root', *rv_names]
        df_coefs = pd.DataFrame(coef_list, columns=['i', *coef_cols])
        df_melted_coefs = pd.melt(df_coefs, id_vars=['i'], value_vars=coef_cols,
                                  var_name='feature', value_name='value')
        g = sns.FacetGrid(df_melted_coefs, col="feature", sharex=False, sharey=False, col_wrap=4)

        g = g.map(sns.distplot, "value")
        coef_id = "{}-coefs".format(spectrum_id)
        self.saver.store_figure(coef_id)
        pass


class LassoTracer(Tracer):
    def __init__(self, *args, feature_names=None, pos_map=None, **kwargs):
        super().__init__(*args, **kwargs)
        if feature_names:
            self.feature_names = feature_names
        if pos_map:
            self.pos_map = pos_map

    def fit(self, X, y, feature_names=None, pos_map=None, attribute='unknown-attrib', method="advi", mcmc_tune=2000,
            mcmc_cores=5, tree_depth=200, mcmc_samples=15000, p_mass=0.8, advi_its=100000, advi_trace_draws=5000,
            model_interactions=True):
        fit_start = time.time()
        self.X = X
        self.y = y
        if feature_names:
            self.feature_names = feature_names
        if pos_map:
            self.pos_map = pos_map
        else:
            self.pos_map = {opt: idx for idx, opt in enumerate(feature_names)}

        if model_interactions:
            if self.t_wise:
                self.interactions_possible = self.t_wise > 1
                print("Interactions possible because t =", self.t_wise)
            else:
                self.interactions_possible = len(self.X[0]) < len(self.X)
            noise_str = 'noise'
        else:
            self.interactions_possible = False

        self.x_shared = np.array(self.X)

        seed_lin = np.random.randint(0, 10 ** 5)

        start_ft_selection = time.time()
        print("Starting feature and itneraction selection.")
        self.final_var_names, lars_pipe, pruned_x = self.get_influentials_from_lasso()
        self.cost_ft_selection = time.time() - start_ft_selection
        print("Feature selection with lasso took {}s".format(self.cost_ft_selection))
        stage_name_lasso = "it-lasso"
        print_flush("Starting {}".format(stage_name_lasso))
        lasso_start = time.time()
        lasso_model, lasso_reg_features, lasso_trace = \
            self.get_and_fit_model_biased(advi_its, advi_trace_draws, mcmc_cores, mcmc_samples,
                                          mcmc_tune, method, seed_lin)

        lasso_end = time.time()
        self.fitting_times[stage_name_lasso] = lasso_end - lasso_start
        stage_2_influential_fts, stage_2_influential_inters = self.get_ft_and_inters_from_rvs(lasso_trace, noise_str,
                                                                                              p_mass,
                                                                                              lasso_reg_features)
        print("Starting snapshot")
        snapshot = self.construct_snapshot(stage_2_influential_fts, stage_2_influential_inters, lasso_model,
                                           lasso_reg_features, lasso_trace,
                                           stage_name=stage_name_lasso)
        self.history[stage_name_lasso] = snapshot
        model_trace_dict = get_model_trace_dict(lasso_trace, lasso_model)
        self.models["it-final"] = model_trace_dict
        print_flush("Finished snapshot")
        self.final_model = lasso_model
        self.final_trace = lasso_trace
        print("linked trace to Tracer object")
        fit_end = time.time()
        self.total_experiment_time = fit_end - fit_start

    def predict_raw_keep_trace_samples(self, x, model=None, trace=None, n_post_samples=None):
        if not model:
            model = self.final_model
            trace = self.final_trace
        # self.x_shared.set_value(np.array(x))
        transformed_x = self.get_pm_train_data(np.array(x))[1]
        self.train_data.set_value(transformed_x)
        self.x_shared = transformed_x
        with model:
            # pm.set_data({'x_shared': x})
            # ppc = pm.sample_posterior_predictive(trace )
            if n_post_samples:
                ppc = pm.sample_posterior_predictive(trace, samples=n_post_samples)
            else:
                ppc = pm.sample_posterior_predictive(trace)
        # ppc = pm.sample_posterior_predictive(trace[1000:], model=self.final_model, samples=2000)
        prediction_samples = list(ppc.values())[0]
        return prediction_samples

    def get_influentials_from_lasso(self, degree=2):
        train_x_2d = np.atleast_2d(self.X)
        train_y = self.y
        lars = LassoCV(cv=3, positive=False, )  # .fit(train_x_2d, train_y)
        if self.interactions_possible:
            poly_mapping = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
            lars_pipe = make_pipeline(poly_mapping, lars)
        else:
            lars_pipe = lars
        lars_pipe.fit(train_x_2d, train_y)

        if self.interactions_possible:
            transformed_x = poly_mapping.transform(train_x_2d)
            if self.feature_names is not None:
                ft_inters = poly_mapping.get_feature_names(input_features=self.feature_names)
            else:
                ft_inters = poly_mapping.get_feature_names()
        else:
            transformed_x = train_x_2d
            ft_inters = self.feature_names
        coefs = lars.coef_

        ft_inters_and_influences = {}
        inf_idx = []
        for i, (c, ft_inter) in enumerate(zip(coefs, ft_inters)):
            if c != 0.0:
                inf_idx.append(i)
                ft_inters_and_influences[tuple(ft_inter.split())] = c
        pruned_x = transformed_x[:, inf_idx]

        ft_inters_and_influences = {tuple(ft_inter.split()): c for c, ft_inter in zip(coefs, ft_inters) if c != 0.0}
        return ft_inters_and_influences, lars_pipe, pruned_x

    def get_and_fit_model_biased(self, advi_its, advi_trace_draws, mcmc_cores, mcmc_samples, mcmc_tune,
                                 method, rnd_seed):
        print_flush("Generating interactions.")
        start = time.time()
        rv_names, train_data = self.get_pm_train_data()
        end = time.time()
        print()
        print("Finished generating interactions in {0:9.1} minutes.".format((end - start) / 60))
        print()
        sys.stdout.flush()

        pm_model, pred, err_mean, err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample = self.get_pm_model(
            rv_names, train_data)
        new_trace = self.fit_pm_model(advi_its, advi_trace_draws, mcmc_cores, mcmc_samples, mcmc_tune, method, err_mean,
                                      err_std, weighted_errs_per_sample, weighted_rel_errs_per_sample,
                                      pm_model, pred, rnd_seed)
        return pm_model, rv_names, new_trace


def get_feature_names_from_rv_id(ft_inter):
    new_ft_inter = ft_inter.replace("_log__", '')
    new_ft_inter = new_ft_inter.replace("active_", '')
    new_ft_inter = new_ft_inter.replace("_scale", '')
    new_ft_inter = new_ft_inter.replace("influence_", '')
    result = new_ft_inter.split("&")
    return result


def print_baseline_perf(train_x, train_y, eval_x, eval_y):
    train_x_2d = np.atleast_2d(train_x)
    rf = RandomForestRegressor(n_estimators=10, n_jobs=1).fit(train_x_2d, train_y)
    lr = LinearRegression(n_jobs=1).fit(train_x_2d, train_y)
    ridge = Ridge(alpha=1.0, fit_intercept=True).fit(train_x_2d, train_y)
    lars = LassoLars(alpha=.1, positive=False, fit_path=True).fit(train_x_2d, train_y)

    for name, reg in [('lr', lr), ('rf', rf), ('ridge', ridge), ('lars', lars), ]:
        print('\t__ {} __'.format((name)))
        print_scores(name, reg, 'train', train_x, train_y)
        print_scores(name, reg, 'eval', eval_x, eval_y)
        print('')

    print('lr intercept: {}'.format(lr.intercept_))
    print('lr coefs: {}'.format(lr.coef_))
    print('ridge intercept: {}'.format(ridge.intercept_))
    print('ridge coefs: {}'.format(ridge.coef_))


def get_random_seed():
    return np.random.randint(np.iinfo(np.uint32).max)


def print_scores(model_name, reg, sample_set_id, xs, ys, print_raw=False):
    errors = get_err_dict(reg, xs, ys)
    for score_id, score in errors.items():
        if not print_raw and "raw" in score_id:
            continue
        print('{} {} set {} score: {}'.format(model_name, sample_set_id, score_id, score))
    print()


def get_err_dict(reg, xs, ys):
    y_pred = reg.predict(xs)
    errors = get_err_dict_from_predictions(y_pred, xs, ys)
    return errors


def get_err_dict_from_predictions(y_pred, xs, ys):
    mape = score_mape(None, xs, ys, y_pred)
    rmse = score_rmse(None, xs, ys, y_pred)
    r2 = r2_score(ys, y_pred)
    errors = {"r2": r2, "mape": mape, "rmse": rmse,
              "raw": {"x": xs, "y_pred": y_pred, "y_true": ys}}
    return errors


def score_rmse(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    rms = sqrt(mean_squared_error(y_true, y_predicted))
    return rms


def score_mape(reg, xs, y_true, y_predicted=None):
    if y_predicted is None:
        y_predicted = reg.predict(np.atleast_2d(xs))
    mape = np.mean(np.abs((y_true - y_predicted) / y_true)) * 100
    return mape


def get_snapshot_dict(lr_reg_dict, err_dict, errors, lin_reg_features, linear_model, significant_ft, trace, trace_errs):
    snapshot = {
        "used-features": lin_reg_features,
        "significant-ft": significant_ft,
        "prob-err": errors,
        "prob-model": linear_model,
        "lr-err": err_dict,
        "lr-models": lr_reg_dict,
        "trace": trace,
        "pred-trace-errs": trace_errs,
    }
    return snapshot


class DistBasedRepo(systems.ConfigSysProxy):
    PARENT_BASE_FOLDER = "SupplementaryWebsite"
    PERF_PRED_BASE_FOLDER = "PerformancePredictions"
    SUMMARY_BASE_FOLDER = "Summary"
    MEASUREMENTS_BASE_FOLDER = "MeasuredPerformanceValues"
    T_FILE_NAMES = {
        1: "twise_t1.txt",
        2: "twise_t2.txt",
        3: "twise_t3.txt",
    }
    SPLC_COLUMNS = ["Round", "Model", "LearningError", "LearningErrorRel", "ValidationError", "ValidationErrorRel",
                    "ElapsedSeconds", "ModelComplexity", "BestCandidate", "BestCandidateSize", "BestCandidateScore",
                    "TestError"]

    def __init__(self, root, sys_name, attribute=None, val_set_size=0, val_set_rnd_seed=None):
        self.root = self.get_common_root(root)
        self.sys_name = sys_name

        self.summary_folder = self.get_summary_folder()
        self.measurements_folder = self.get_measurements_folder()

        super().__init__(self.measurements_folder, attribute)
        self.sample_sets = self.parse_sample_sets()

    def get_train_eval_split(self, t):
        x_train = list(self.sample_sets[t].keys())
        y_train = list(self.sample_sets[t].values())
        x_eval = list(self.all_configs.keys())
        y_eval = list(self.all_configs.values())
        return x_train, y_train, x_eval, y_eval

    def get_summary_folder(self):
        f_sum = os.path.join(self.root, DistBasedRepo.PERF_PRED_BASE_FOLDER, DistBasedRepo.SUMMARY_BASE_FOLDER,
                             self.sys_name)
        return f_sum

    def get_measurements_folder(self):
        f_meas = os.path.join(self.root, DistBasedRepo.MEASUREMENTS_BASE_FOLDER, self.sys_name)
        return f_meas

    @staticmethod
    def get_common_root(root):
        new_root = str(root)
        if DistBasedRepo.PARENT_BASE_FOLDER in list(os.listdir(root)):
            new_root = os.path.join(new_root, DistBasedRepo.PARENT_BASE_FOLDER)
        return new_root

    def parse_sample_sets(self):
        splc_lines = {}
        for filename in os.listdir(self.summary_folder):
            for t, t_name in DistBasedRepo.T_FILE_NAMES.items():
                if t_name.lower() in filename.lower():
                    abs_path = os.path.join(self.summary_folder, filename)
                    with open(abs_path) as f:
                        content = f.readlines()
                        splc_lines[t] = content
        splc_conf_sets = {}
        for t, lines in splc_lines.items():
            confs = [self.parse_line(line) for line in lines]
            splc_conf_sets[t] = {conf: y for conf, y in confs}
        return splc_conf_sets

    def parse_logs_for_coeffs(self):
        log_lines = {}
        for filename in os.listdir(self.summary_folder):
            for t, t_name in DistBasedRepo.T_FILE_NAMES.items():
                if t_name.lower() in filename.lower():
                    abs_path = os.path.join(self.summary_folder, filename)
                    with open(abs_path) as f:
                        content = f.readlines()
                        log_lines[t] = content[-1]
        splc_coeff_sets = {}
        for t, lines in log_lines.items():
            final_coeff_line = self.find_final_coeff_line(lines)
            coeff_map = self.parse_coeffs(final_coeff_line)
            splc_coeff_sets[t] = coeff_map
        return splc_coeff_sets

    def parse_line(self, line):
        conf_decoded = line.split(' ')[1][1:-1]
        feature_set = [f for f in conf_decoded.split('%;%') if len(f) > 0 and f != "root"]
        conf = tuple(self.get_conf_for_feature_set(feature_set))
        y = self.eval(conf)
        return conf, y

    def find_final_coeff_line(self, lines):
        rev_lines = lines[::-1]
        final_line = None
        for i, line in enumerate(rev_lines):
            if "Analyze finished" in line:
                final_line = rev_lines[i + 1]
                break
        return final_line

    def parse_coeffs(self, final_coeff_line):
        df = pd.DataFrame(final_coeff_line, columns=DistBasedRepo.SPLC_COLUMNS)
        model_str = df["Model"]
        terms = model_str.split(" + ")
        coeff_map = {}
        for term in terms:
            term_comps = term.split(" * ")
            coeff, ft = term_comps[0], tuple(term_comps[1:])
            coeff_map[ft] = coeff_map
        return coeff_map


def remove_raw_field(inter_errors):
    inter_errors = {key: val for key, val in inter_errors.items() if key != "raw"}
    return inter_errors


def get_model_trace_dict(lin_trace, linear_model):
    model_trace_dict = {"model": linear_model, "trace": lin_trace}
    return model_trace_dict


def print_flush(print_text):
    print(print_text)
    sys.stdout.flush()


def weighted_avg_and_std(values, weights, gamma=1):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    if gamma != 1:
        weights = np.power(weights, gamma)
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    if variance <= 0:
        sqr_var = 0.0
    else:
        sqr_var = math.sqrt(variance)
    return average, sqr_var


if __name__ == "__main__":
    main()


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)


class P4Regressor(BaseEstimator, RegressorMixin):
    def __init__(self, out_dir: str):
        """
        Constructor for a P4Regressor.

        Parameters
        ----------
        out_dir :  path specifying where to store trained P4 model
        """
        saver = CompressedSaverHelper(out_dir, dpi=150, fig_pre='')
        time_str = get_time_str()
        session_dir_name = "-".join([time_str])
        saver.set_session_dir(session_dir_name)
        self.lasso_tracer = LassoTracer(saver, inters_only_between_influentials=True, no_plots=True,
                                        relative_obs_noise=True, absolute_obs_noise=False)
        self.coef_ = None
        self.coef_samples_ = None
        self.pos_map = None

    def fit(self, X, y, feature_names: list = None, pos_map: dict = None, mcmc_cores: int = 3, mcmc_samples: int = 1000,
            mcmc_tune: int = 500,
            advi_its: int = 1000, method="mcmc", model_interactions=True):
        """

        Parameters
        ----------

        X : training data
        y : training labels
        feature_names : list of names for the feaures, which are represented as columns in X - give either feature_names or pos_map
        pos_map : map with indexes as keys and feature names as values, e.g. {0:"optA", 1:"optB"}. Will be generated is used internally to label generated interactions - give either feature_names or pos_map
        mcmc_cores : Number of traces to compute in parallel during MCMC
        mcmc_samples : Number of effective samples to return for each random varible, i.e., for each feature influence
        mcmc_tune : Number of tuning samples that are discarded before conducting mcmc_samples samples
        advi_its : Number of variational inference steps that approximate MCMC before MCMC starts
        method : "mcmc" if combining variational inference with MCMC (recommended)
        model_interactions : Learns pairwise interactions between influential options, and only option-wise influences
        """
        if pos_map:
            feature_names = list(pos_map)
        elif feature_names:
            pos_map = {opt: idx for idx, opt in enumerate(feature_names)}
        else:
            pos_map = {opt: idx for idx, opt in zip(range(X.shape[1]), iter_all_strings())}
            feature_names = list(pos_map)

        self.pos_map = pos_map
        self.lasso_tracer.fit(X, y, pos_map=pos_map, feature_names=feature_names, mcmc_cores=mcmc_cores,
                              mcmc_samples=mcmc_samples, mcmc_tune=mcmc_tune,
                              advi_its=advi_its, method=method, model_interactions=model_interactions)
        self.update_coefs()

    def update_coefs(self):
        """
        Uses the current inferred trace to compute self.coef_ and self.coef_samples_
        """
        tracer = self.lasso_tracer
        trace = tracer.final_trace
        root_samples = trace["root"]
        influence_samples = trace["influence"]
        influence_dict = {varname: influence_samples[:, i] for i, varname in
                          enumerate(self.lasso_tracer.final_var_names)}
        relative_error_samples = trace["relative_error"]
        self.coef_samples_ = {
            "root": root_samples,
            "influences": influence_dict,
            "relative_error": relative_error_samples
        }
        root_mode = float(np.mean(az.hpd(root_samples, credible_interval=0.01)))
        influence_modes = list(np.mean(az.hpd(influence_samples, credible_interval=0.01), axis=1))
        influence_modes_dict = {varname: mode for mode, varname in
                                zip(influence_modes, self.lasso_tracer.final_var_names)}
        rel_error_mode = float(np.mean(az.hpd(relative_error_samples, credible_interval=0.01)))
        self.coef_ = {
            "root": root_mode,
            "influences": influence_modes_dict,
            "relative_error": rel_error_mode
        }

    def coef_ci(self, ci: float):
        """
        Returns confidence intervals with custom confidence for p
        Parameters
        ----------
        ci : specifies confidence of the confidence interval to compute from sampled influence values

        Returns dictionary containing a confidence interval for each influence
        -------

        """
        coef_cis = {}
        assert_ci(ci)
        for key, val in self.coef_samples_.items():
            if key == "influences":
                inf_dict = {}
                for feature_name, feature_samples in val.items():
                    inf_dict[feature_name] = az.hpd(feature_samples, credible_interval=ci)
                coef_cis[key] = inf_dict
            else:
                coef_cis[key] = az.hpd(val, credible_interval=ci)
        return coef_cis

    def predict(self, X, n_samples: int = None, ci: float = None):
        """
        Performs a prediction conforming to the sklearn interface.

        Parameters
        ----------
        X : Array-like data
        n_samples : number of posterior predictive samples to return for each prediction
        ci : value between 0 and 1 representing the desired confidence of returned confidence intervals. E.g., ci= 0.8 will generate 80%-confidence intervals

        Returns
        -------
         - a scalar if only x is specified
         - a set of posterior predictive samples of size n_samples if is given and n_samples > 0
         - a set of pairs, representing lower and upper bounds of confidence intervals for each prediction if ci is given

        """
        tracer = self.lasso_tracer
        if not n_samples:
            n_samples = 500
            y_samples = tracer.predict_raw_keep_trace_samples(X, n_post_samples=n_samples)
            y_pred = np.mean(az.hpd(y_samples, credible_interval=0.01), axis=1)
        else:
            y_samples = tracer.predict_raw_keep_trace_samples(X, n_post_samples=n_samples)
            if ci:
                assert_ci(ci)
                y_pred = az.hpd(y_samples, credible_interval=ci)
            else:
                y_pred = y_samples
        return y_pred


def assert_ci(ci):
    assert 0 < ci < 1, "Confidence should be given 0 < ci < 1"
