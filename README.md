# Non-local Attention Learning on Large Heterogeneous Information Networks

## NLAH
This is a PyTorch implementation of the NLAH model for learning node embedding in heterogeneous information networks.

We provide a sample ACM dataset in "data" folder. It includes three types of non-local features: 2nd Order Proximity (2ndprox), Personalized PageRank (ppr) and Positive Pointwise Mutual Information (ppmi).

To start training on the provided ACM dataset, you need to create a folder named "/model/acm/{non-local features}" to store the trained model.

Run the following command as an example.
```
python3 src/main.py -dataset acm -nl_type 2ndprox -nhid 64 -nlayer 2
```

## Citation
```
@inproceedings{xiao2019nonlocal,
title={Non-local Attention Learning on Large Heterogeneous Information Networks},
author={Xiao, Yuxin and Zhang, Zecheng and Yang, Carl and Zhai, Chengxiang},
booktitle={2019 IEEE International Conference on Big Data (Big Data)},
year={2019},
organization={IEEE}
}
```