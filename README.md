# Mastering Uncertainty in Performance Estimations of Configurable Software Systems

## Supplementary Material
[md-mape]: mape/README.md
With this web site, we provide the following material:

- [descriptions of the **subject systems** we used][md-subject-systems]
- [**additional results** of our accuracy evaluation in RQ1 & RQ2 including the absolute model][md-mape]
- [more perspective on the **uncertainty calibration** of our models][md-calibration]
 <!--- the **data** we used for our evaluation-->

We also provide [data containing the accuracies of our models as well as predicted confidence interals][md-data] as well as [an implementation of our probabilistic programming approach](code/README.md) with an updated Dockerscript for an easier set-up. 

##Extended Supplementary Material
As a result of a more detailed analysis uncertainty within the influences of options and interactions (*terms*) on execution time and energy consumption, we additionally provide
 - [term influence confidence interval widths for models trained on T<sub>2</sub> vs models trained on T<sub>3</sub>][term-cis]  
 - [prediction confidence interval widths for models trained on T<sub>2</sub> vs models trained on T<sub>3</sub>][prediction-cis]

[md-mape]: mape/README.md
[md-subject-systems]: ./systems/README.md
[md-calibration]: calibration/README.md
[md-main]: ./README.md
[md-data]: ./data/eval.csv
[term-cis]: ./extension/term-cis/README.md
[prediction-cis]: ./extension/prediction-cis/README.md