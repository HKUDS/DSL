# Denoised Self-Augmented Learning for Social Recommendation (DSAL)

This is the PyTorch-based implementation for DSAL model proposed in this paper:

> Denoised Self-Augmented Learning for Social Recommendation

![model](https://p.ipic.vip/ikwc4z.png)

## Abstract

Social recommendation has been increasingly investigated in a broad spectrum of online applications (e.g., e-commerce, online streaming) to leverage social information for help user-item interaction modeling. Recently, Self-Supervised Learning (SSL) has been outstandingly successful in alleviating data sparsity with the augmented learning tasks. Inspired by this, recent attempts bring the benefits of SSL into social recommendation by supplementing the main supervised task with social-aware self-supervised signals. However, social information is unavoidably noisy for characterizing user preference, due to the ubiquitous presence of interest-irrelevant social connections, e.g., colleagues or classmates who do not share many common interests. To rectify this, we propose a new social recommender with a <u>D</u>enoised Cross-view <u>S</u>elf-<u>A</u>ugmented <u>L</u>earning paradigm (DSAL). It not only preserves the helpful social relations for enhancing user-item interaction modeling, but also allows the personalized cross-view knowledge transfer with adaptive semantic alignment in embedding space. Experimental results on various recommendation benchmarks verify the advantages of our DSAL over state-of-the-art methods.



## Environment

The implementation for DSAL is under the following development environment:

- python=3.8
- torch=1.12.1
- numpy=1.23.2
- scipy=1.9.1



## Datasets

Our experiments are conducted on three benchmark datasets collected from Ciao, Epinions and Yelp online platforms. In those sites, social connections can be established among users in addition to their observed implicit feedback (e.g., rating, click) over different items.

| Dataset  | # Users | # Items | # Interactions | Interaction Density | # Social Ties |
| :------: | :-----: | :-----: | :------------: | :-----------------: | :-----------: |
|   Ciao   |  6,672  | 98,875  |    198,181     |       0.0300%       |    109,503    |
| Epinions | 11,111  | 190,774 |    247,591     |       0.0117%       |    203,989    |
|   Yelp   | 161,305 | 114,852 |   1,118,645    |       0.0060%       |   2,142,242   |



## Usage

Please unzip the datasets first. Also you need to create the `History/` and the `Models/` directories. The command lines to train DSAL on the three datasets are as below. The un-specified hyperparameters in the commands are set as default.

- Ciao

  ```shell
  bash scripts/run_ciao.sh
  ```

- Epinions

  ```shell
  bash scripts/run_epinions.sh
  ```

- Yelp

  ```shell
  bash scripts/run_yelp.sh
  ```



### Important Arguments

- `gnn_layer`: It is the number of gnn layers, which is searched from `{1, 2, 3, 4}`.
- `reg`: It is the weight for weight-decay regularization. We tune this hyperparameter from the set `{1e-3, 1e-4, 1e-5, 1e-6, 1e-7}`.
- `uuPre_reg`: It is the weight for social graph prediction regularization, which is tuned from `{1e1, 1e0, 1e-1, 1e-2, 1e-3}`
- `sal_reg`: It is the weight for self-augmented regularization. We tune it from the set `{1e-3, 1e-4, 1e-5, 1e-6, 1e-7}`