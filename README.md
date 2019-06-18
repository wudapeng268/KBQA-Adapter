# KBQA-Adapter

This is the code and data for ACL 2019 long paper "Learning Representation Mapping for Relation Detection in Knowledge Base Question Answering". 

## Requirements

* Python3.5
* Tensorflow 1.7.0


## Re-organize SimpleQuestion to SimpleQuestion_Balance

As we discuss, for SimpleQuestion, 99% of the relations in the test set also exist in the training data. In order to evaluate unseen relation detection and seen relation detection fairly, we re-organize SimpleQuestion to SimpleQuestion_Balance(SQB), the dataset is released at [Data/SQB](Data/SQB) and the script for re-organize this dataset is [mix_dataset.py](qa+adapter/re-organize_dataset/mix_dataset.py).

## Reproduce Main Result
The main code for this paper is [qa+adapter](qa+adapter).

### Get JointNRE embedding for FB2M
Our relation embeddings are trained by JointNRE between FB2M and wikipedia, please see [this link](https://drive.google.com/open?id=137LGV3pYAU2lDR4TWSQf_kbOyptn-daz) for detail.

<!-- ### Train relation detection model -->


### Train Baseline and all model with adapter (including adapting JointNRE)
```
cd qa+adapter
bash script/run_baseline.sh
```

### Train adapter with only mapping and adapter without fine-tuning
```
cd qa+adapter
bash script/run-other.sh $card
```

## Reproduce KBQA result

### Entity Linking
We use FocusPrune to annotated the entity, please refer to <https://github.com/wudapeng268/KB-QA-baseline> for detail.


### Test for kbqa
```
cd qa+adapter
bash script/test-all-kbqa.sh $card_num
```

## Influence of Number of Relations for Training
Our data for this experiment at [Data/Number_relation_in_training](Data/Number_relation_in_training) created by [this script](qa+adapter/cut_train/keep_train_len.py). 

You can use following script to reproduce this result:
```
cd qa+adapter
bash script/run_tl.sh
```

## Citation

If you use our code or data, please kindly cite the paper about it!
```
@inproceedings{peng19acl,
    title = {Learning Representation Mapping for Relation Detection in Knowledge Base Question Answering},
    author = {Peng Wu, Shujian Huang, Rongxiang Weng, Zaixiang Zheng, Jianbing Zhang, Xiaohui Yan and Jiajun Chen},
    booktitle = {The 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
    address = {Florence, Italy},
    month = {July},
    year = {2019}
}
```

