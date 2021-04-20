import os
import argparse
import pickle
import bz2
import arviz as az
import pprint

# needed to unpickle!
from p4 import *


def get_result_files(parent):
    contents = []
    for run in os.listdir(parent):
        abs_job_path = os.path.join(parent, run)
        if os.path.isdir(abs_job_path):
            result_candidates = list(os.listdir(abs_job_path))
            trace_results_file_name = "results-tracer.pc"
            if trace_results_file_name in result_candidates and "results-args.p" in result_candidates:
                pickle_args_path = os.path.join(abs_job_path, "results-args.p")
                tracer_path = os.path.join(abs_job_path, trace_results_file_name)
                args = pickle.load(open(pickle_args_path, "rb"))
                if "no_plots" in args:
                    del args["no_plots"]
                    print(tracer_path)
                try:
                    tracer = pickle.load(bz2.open(tracer_path))
                    hist = tracer.history
                    contents.append((args, hist, tracer))
                except Exception as e:
                    print("Error unpickling", tracer_path, "for run", run, "-", e)
    return contents


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sys-dir",
                        help="specifies the parent folder for all config sys",
                        type=str, required=False, default="/application/Distance-Based_Data/SupplementaryWebsite/",
                        )
    parser.add_argument("--results-dir",
                        help="specifies the parent folder for all run results",
                        type=str, required=False, default="/results/",
                        )
    parser.add_argument("--conf-steps",
                        help="how many steps to check for confidence calibration",
                        type=int, default=20, )
    parser.add_argument("--predictive-samples",
                        help="how many samples to draw from the predictive distribution",
                        type=int, default=4000, )
    args = parser.parse_args()

    sys_dir = args.sys_dir
    results_dir = args.results_dir
    conf_steps = args.conf_steps
    n_post_samples = args.predictive_samples
    inference_data = os.path.join(results_dir, "last-inference")
    output = os.path.join(results_dir, "last-evaluation")
    contents = get_result_files(inference_data)

    repos = {}
    idx_keys = None
    obs = []
    for args, hist, tracer in contents:
        t = int(args["t"])
        sys_name = args["sys_name"]
        run_id = get_run_id_from_path(args["folder"])
        attribute = args["attribute"]
        if sys_name not in repos:
            repo = DistBasedRepo(sys_dir, sys_name, attribute)
            repos[sys_name] = repo
        repo = repos[sys_name]
        x_eval = list(repo.all_configs.keys())
        y_eval = list(repo.all_configs.values())
        err_dict = {}
        for it in hist:
            idx = tuple(args.values())
            model = hist[it]["prob-model"]
            trace = hist[it]["trace"]
            fitting_time = tracer.fitting_times[it]
            weighted_errs_per_sample = list(tracer.weighted_errs_per_sample)
            weighted_rel_errs_per_sample = list(tracer.weighted_rel_errs_per_sample)
            err_dict[it] = {"weighted_errs_per_sample": weighted_errs_per_sample,
                            "weighted_rel_errs_per_sample": weighted_rel_errs_per_sample}
            fitting_time = tracer.fitting_times[it]
            ft_selection_seconds = tracer.prior_spectrum_cost

            t_start = time.time()

            alt_preds = tracer.predict_raw(x_eval)

            with model:
                ppc = pm.sample_posterior_predictive(trace, samples=n_post_samples, )
            t_end = time.time()
            pred_time = t_end - t_start

            y_samples = ppc["y_observed"]
            # y_pred_median = np.median(y_samples, axis=0)
            y_pred = np.mean(az.hpd(y_samples, credible_interval=0.01), axis=1)
            correct_in_dict = {}
            conf_mape_dict = {}
            conf_mape_without_zeros = {}
            intervals = np.linspace(0, 1, conf_steps + 1)[1:-1]
            print("computing confidence accuracies")
            for perc in intervals:
                correct_in_dict[perc] = float(Tracer.calc_confidence_err(perc, y_eval, y_samples))
                conf_mape_dict[perc], conf_mape_without_zeros[perc] = Tracer.calc_confidence_closest_mape(perc, y_eval,
                                                                                                          y_samples)
                print("Finished", perc)

            # correct_in_095 = float(Tracer.calc_confidence_err(0.95, y_eval, y_samples))
            assert len(y_eval) == len(y_pred)
            eval_mape = score_mape(None, None, y_eval, y_pred)

            results = {"args":args, "it":it, "eval_mape":eval_mape, "fitting_time":fitting_time,
                       "ft_selection_seconds":ft_selection_seconds, "n_post_samples":n_post_samples,
                       "pred_time":pred_time, "corect_in_conf":correct_in_dict,
                       "mape_to_nearest_conf_bound":conf_mape_dict}


            obs.append(results)
        print("storing pickle")
        if not os.path.exists(output):
            os.mkdir(output)
        out_path = os.path.join(output, run_id + ".p")
        with open(out_path, 'wb') as f:
            pickle.dump(obs, f)
        obs_str = pprint.PrettyPrinter().pformat(obs)
        str_out_path = os.path.join(output, run_id + ".txt")
        with open(str_out_path, 'w') as f:
            f.write(obs_str)
        out_path_err_dist = os.path.join(output, "errs-" + run_id + ".p")
        with open(out_path_err_dist, 'wb') as f:
            pickle.dump(err_dict, f)

    print("Index:", idx_keys)
    print("DONE")


def get_run_id_from_content(c):
    path = c[0]["folder"]
    return get_run_id_from_path(path)


def get_run_id_from_path(path):
    return os.path.basename(os.path.dirname(path))


if __name__ == "__main__":
    main()
