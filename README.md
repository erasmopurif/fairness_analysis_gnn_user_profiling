# Fairness analysis of GNN-based models for behavioural user profiling

Analysis of **fairness** on state-of-the-art GNN-based models for behavioural user profiling.

## GNN-based models
The models considered (so far) are:

* ***CatGCN*** (TKDE '21)
    * [Paper](https://arxiv.org/abs/2009.05303)
    * [Code repository](https://github.com/TachiChan/CatGCN)
    * [Data process repository](https://github.com/TachiChan/Data_Process)
* ***RHGN*** (CIKM '21)
    * [Paper](https://arxiv.org/abs/2110.07181)
    * [Code repository](https://github.com/CRIPAC-DIG/RHGN)

## Datasets
The preprocessed files required for running each model are included as a zip file within the related folder.
All the files needed for running the models can be generated with the `*_data_process_*.ipynb` files.

The raw datasets are available at the following links:
* [Alibaba](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56)
* [JD](https://github.com/guyulongcs/IJCAI2019_HGAT)

## Fairness metrics
Every *fairness metrics* defined below considers:
* $y \in \{0,1\}$ as the binary target label;
* $\hat{y} \in \{0,1\}$ as the prediction of the user profiling model $f: x \rightarrow y$;
* $s \in \{0,1\}$ as the sensitive attribute.

In the metrics' descriptions, we also exploit the following notation which relates to classification properties: TPR, FPR, TNR and FNR, denoting *true positive rate*, *false positive rate*, *true negative rate* and *false negative rate*, respectively.

The *fairness metrics* that can be evaluated for each model are the following:

* ***Statistical parity*** (also known as *demographic parity*).
    It defines fairness as an equal probability for each group of being assigned to the positive predictive class, i.e. the predictions are independent with the sensitive attribute.
    $$
    P(\hat{y} = 1 | s = 0) = P(\hat{y} = 1 | s = 1)
    $$

* ***Equal opportunity***. It requires the probability of a subject in a positive class to be classified with the positive outcome should be equal for each group, i.e. TPR should be the same across groups.
    $$
    P(\hat{y} = 1 | y = 1, s = 0) = P(\hat{y} = 1 | y = 1, s = 1)
    $$

* ***Overall accuracy equality***. It considers the relative accuracy rates across different groups and defines fairness as the equal probability of a subject from either positive or negative class to be assigned to its respective class, i.e. each group should have the same prediction accuracy.
    $$
    P(\hat{y} = 0 | y = 0, s = 0) + P(\hat{y} = 1 | y = 1, s = 0) = P(\hat{y} = 0 | y = 0, s = 1) + P(\hat{y} = 1 | y = 1, s = 1)
    $$

* ***Treatment equality***. It requires the ratio of errors made by the classifier to be equal across different groups, i.e. each group should have the same ratio of *false negatives* (FNR) and *false positives* (FPR).
    $$
    \frac{P(\hat{y} = 1 | y = 0, s = 0)}{P(\hat{y} = 0 | y = 1, s = 0)} = \frac{P(\hat{y} = 1 | y = 0, s = 1)}{P(\hat{y} = 0 | y = 1, s = 1)}
    $$

## Track experimental results
Results of experiments are stored and tracked by using [Neptune.ai](https://neptune.ai/).
To make use of it, you need to specify the *project* (`--neptune-project`) and the *api token* (`--neptune-token`) as arguments when running the code.

Otherwise, comment the lines of code related to that for each model.

## Run the code
Test runs for each combination of model-dataset (args for tracking the results are omitted).

### CatGCN - Alibaba dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 0 --learning-rate 0.1 --weight-decay 1e-5 \
--dropout 0.1 --diag-probe 1 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input/ali_data/user_edge.csv \
--field-path ./input_ali_data/user_field.npy --target-path ./input_ali_data/user_gender.csv \
--labels-path ./input_ali_data/user_labels.csv --sens-attr bin_age --label gender 
```

### CatGCN - JD dataset
```
$ cd CatGCN
$ python3 main.py --seed 11 --gpu 0 --learning-rate 1e-2 --weight-decay 1e-5 \
--dropout 0.1 --diag-probe 39 --graph-refining agc --aggr-pooling mean --grn-units 64 \
--bi-interaction nfm --nfm-units none --graph-layer pna --gnn-hops 1 --gnn-units none \
--aggr-style sum --balance-ratio 0.7 --edge-path ./input_jd_data/user_edge.csv \
--field-path ./input_jd_data/user_field.npy --target-path ./input_jd_data/user_gender.csv \
--labels-path ./input_jd_data/user_labels.csv --sens-attr bin_age --label gender
```

### RHGN - Alibaba dataset
```
$ cd RHGN
$ python3 ali_main.py --seed 42 --gpu 0 --model RHGN --data_dir ./input_ali_data/ \
--graph G_new --max_lr 0.1 --n_hid 32 --clip 2 --n_epoch 100 \
--label gender --sens_attr bin_age
```

### RHGN - JD dataset
```
$ cd RHGN
$ python3 jd_main.py --seed 3 --gpu 0 --model RHGN --data_dir ./input_jd_data/ \
--graph G_new --max_lr 1e-3 --n_hid 64 --clip 1 --n_epoch 100 \
--label gender --sens_attr bin_age
```

## Contact
Erasmo Purificato (erasmo.purificato@ovgu.de)