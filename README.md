# LightSANs with Dynamic Top-k Interest Allocation
This fork introduces a dynamic interest allocation mechanism within LightSANs to further improve its adaptability and performance. This fork implemented as part of the SKKU Recommender Systems final project. The key enhancements include: The key enhancements include:

1. Complexity-Based Gating:
We implement a gating module (g(u)) that leverages user-specific complexity scores—derived from entropy and variance metrics—to determine which latent interests (k) should be activated for each user. By dynamically assigning a subset of interests, the model can better capture individual user preferences.

2. Thresholding and Sparsity Control:
A threshold is applied to the gating vector to ensure only interests above a certain activation level remain. This step promotes sparsity and prevents unnecessary complexity, helping to mitigate overfitting and focus on the most relevant interests.

3. Consistent Application Across Training, Validation, and Testing:
The dynamic k allocation logic (including complexity scoring and thresholding) is uniformly applied during training, validation, and testing. This ensures that the performance improvements observed during validation are more likely to generalize to the test set.

4. Regularization and Early Stopping:
To address potential overfitting that can arise from increased model complexity, we recommend fine-tuning hyperparameters and enabling early stopping (via stopping_step and adjusting valid_metric) to stabilize training and enhance generalization.

With these modifications, the model can flexibly adjust the number of active interests per user, potentially improving metrics like NDCG@10 and Hit@10, while maintaining a balance between model complexity and generalization performance.

<img src="./dynamick.png" width = "500px" align=center />

<br>

# Original Work : LightSANs
This is our Pytorch implementation for our SIGIR 2021 short paper:
> Xinyan Fan, Zheng Liu, Jianxun Lian, Wayne Xin Zhao, Xing Xie, and Ji-Rong Wen (2021). "Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation." In SIGIR 2021.
[PDF](https://www.microsoft.com/en-us/research/uploads/prod/2021/05/LighterandBetter_Low-RankDecomposedSelf-AttentionNetworksforNext-ItemRecommendation.pdf)

## Overview
We propose the low-rank decomposed self-attention networks **LightSANs** to improve the effectiveness and efficiency of SANs-based recommenders. Particularly, it projects user's historical items into a small constant number of latent interests, and leverages item-to-interest interaction to generate the user history representation. Besides, the decoupled position encoding is introduced, which expresses the items’ sequential relationships much more precisely. The overall framework of LightSANs is depicted bellow.

<img src="https://github.com/BELIEVEfxy/LightSANs/blob/main/model.png" width = "500px" align=center />

## Requirements
- Python 3.6
- Pytorch >= 1.3

Notice: For all sequencial recommendation models, we use the first version of RecBole v0.1.1 to do our experiments. The more details are on [RecBole](https://github.com/RUCAIBox/RecBole). For efficient Transformers([Synthesizer](https://github.com/leaderj1001/Synthesizer-Rethinking-Self-Attention-Transformer-Models), [LinTrans](https://linear-transformers.com), [Linformer](https://github.com/tatp22/linformer-pytorch), [Performer](https://github.com/lucidrains/performer-pytorch)), we implement them under RecBole Framework based on the source code, in order to ensure fair comparation. 

## Datasets
We use three real-world benchmark datasets, including Yelp, Amazon Books and ML-1M. The details about full version of these datasets are on [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets). For all datasets, we group the interaction records by users and sort them by the interaction timestamps ascendingly. 

Notice: all datasets should be saved in **dataset/**. For example, ml-1m dataset should saved in **dataset/ml-1m/ml-1m.inter**.

## Parameter Settings
We apply the leave-one-out strategy for evaluation, and employ HIT@k and NDCG@k to evaluate the performance. For fair evaluation, we pair each ground truth item in the test set with all items of dataset.

For all SANs-based models, 2 layers of self-attention are deployed, both of which have 2 attention heads. The hidden-dimension of embeddings are set to 64 uniformly. The maximum sequence length is 100, 150 and 200 and the parameter _k_interests_ of LightSANs is 10, 15 and 20 on Yelp, Books and ML-1M datasets, respectively. The dropout rate of turning off neurons is 0.2 for ML-1M and 0.5 for the other four datasets due to their sparsity. The low-rank projected dimension in Synthesizer, Linformer and Performer are set as the same as _k_interests_. We use the Adam optimizer with a learning rate of 0.003 on GPU (TITAN Xp), where the batch size is set as 1024 and 2048 in the training and the evaluation stage, respectively. 

Notice: More details about dataset settings are in .yaml files in **'recbole/properties/dataset'**, model settings are in **'recbole/properties/model/LightSANs.yaml'** and train/evaluation settings are in **'recbole/properties/overall.yaml'**.

## Run
You can use the sh command to run the model:
````
sh run_model.sh
````
You can also train the model directly:
````
python run_recbole.py --model=LightSANs --dataset=ml-1m
````
Main file is **'run_recbole.py'**, LightSANs model file is in **'recbole/model/sequential_recommender/lightsans.py'**.
Log files are in **'log/'**, and trained model(.pth) files are saved in **'saved/'**

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
````
@inproceedings{Fan-SIGIR-2021,
    title  = {Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation},
    author = {Xinyan Fan and
              Zheng Liu and
              Jianxun Lian and
              Wayne Xin Zhao and
              Xing Xie and 
              Ji{-}Rong Wen},
    booktitle = {{SIGIR} '21: The 44th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Virtual Event, Canada, July
               11-15, 2021},
    year = {2021},
    pages     = {1733--1737},
    publisher = {{ACM}},
    doi       = {10.1145/3404835.3462978}
}
````
If you have any questions for our paper or codes, please send an email to xinyanruc@126.com.
