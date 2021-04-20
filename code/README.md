# Running P4
## Set-up with Docker
We recommend running P4 using the provided [Dockerfile](Dockerfile).
Create the docker container by changing to the `code` directory and running
```
docker build -t "p4:p4-dev" .
```

This will automatically install the correct dependencies including the correct version of PyMC3.
Running the Dockerfile may take around 10 minutes.

To access results, you may want to create a directory where the Docker container will bind store data like so:
```
mkdir -p /tmp/p4
```

To run P4, you need to access the docker container terminal by running
```
docker run -it --mount type=bind,source=/tmp/p4,target=/results p4:p4-dev
```


## Inference
From here on, we assume that you are working in the docker container terminal.
### CLI
You can check if the set-up works by running inference on some example data:
```
python3 p4/src/p4/p4.py --sys-name x264 --attribute Performance --method lasso --mcmc-samples 1000 --mcmc-tune 1000 --mcmc-cores 3 --t 1 --rnd 0
```

To run inference on your own data, add a new folder `mysys` to the measurement repository inside the Docker container at `/application/Distance-Based_Data/SupplementaryWebsite/MeasuredPerformanceValues`. Please refer to the existing folders for the required structure. 

Also add a description of your training sets by creating a folder `mysys` at `/application/Distance-Based_Data/SupplementaryWebsite/PerformancePredictions/Summary` inside the Docker container and create the files `twise_t1.txt`,`twise_t2.txt` and `twise_t3.txt`. 
Currently, P4 only supports this structure. SPL Conqueror may be used to create the t-wise sample files. You still may pass arbitrarily sampled training data to P4 by listing each configuration in a new line, matching the format of SPL Conqueror samples. (e.g., adding `prefix "root%;%optionA%;%optionB%;%` for a configuration with `optionA` and `optionB`)
Please note that, P4 will not infer interactions for t=1.

You may run inference with
```
python3 /application/p4/src/p4/p4.py --sys-name mysys --attribute Performance --method lasso --mcmc-samples 4000 --mcmc-tune 1000 --mcmc-cores 3 --t 1 --rnd 0 
```
Chosing 2 or 3 for `--t` will automatically switch the training set and allow interaction inference. For faster (but likely less accurate) inference you can try reducing `--mcmc-samples` to 1000.

### Python API (untested)
Using `LassoTracer` class in Python is an alternative to adopting the t-wise training data format. In our experiments, we only used this class to learn on t-wise sampled training sets; hence, using it with arbitrary training sets is untested.

To use this approach, you first need to create an instance of `LassoTracer` (see the [P4.py script](./p4/src/p4/p4.py) for reference). Then, you can call the `fit` method and pass your training data in `train_x, train_y`.

## Evaluating Inferred Models
The [eval-example.py](p4/src/p4/eval-example.py) script provides basic evaluations. By default, it uses the results it can find at `/results/last-inference/` inside the Docker container.

Run it like
```
python3 /application/p4/src/p4/eval-example.py
```
You can choose a specific number of predictive samples with the `--predictive-samples` option. 

