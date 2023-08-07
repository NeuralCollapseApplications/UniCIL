# Neural Collapse Terminus: A Unified Solution for Class Incremental Learning and Its Variants

[Yibo Yang](https://iboing.github.io)\*,
[Haobo Yuan](https://yuanhaobo.me)\*,
[Xiangtai Li](https://lxtgh.github.io),
[Jianlong Wu](https://jlwu1992.github.io/),
[Lefei Zhang](https://scholar.google.com/citations?user=BLKHwNwAAAAJ&hl=en),
[Zhouchen Lin](https://zhouchenlin.github.io/),
[Philip H.S. Torr](https://www.robots.ox.ac.uk/~phst/),
[Bernard Ghanem](https://www.bernardghanem.com/),
[Dacheng Tao](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/dacheng-tao.html).

[[pdf](https://arxiv.org/pdf/2308.01746)] [[arxiv](https://arxiv.org/abs/2308.01746)] [[code](https://github.com/NeuralCollapseApplications/UniCIL)]

## Environment

[Optional] To start, you need to install the environment with docker (in docker_env directory):

```
docker build -t ftc --network=host .
```

Note that we have published the pre-installed image and no need to run the above command if you network is well.

Then, you can start a new container to run our codes:

```commandline
DATALOC={YOUR DATA LOCATION} LOGLOC={YOUR LOG LOCATION} bash tools/docker.sh
```

## Preparing Data
You do not need to prepare CIFAR datasets.

For ImageNet datasets, please prepare and organize it as following:
```commandline
imagenet
├── train
│   ├── n01440764
│   │   ├── n01440764_18.JPEG
│   │   ├── ...
├── _val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ...
```

Please create a docker container and enter it (DATALOC and LOGLOC have default values, but they may not match your env):
```commandline
DATALOC=/path/to/data LOGLOC=/path/to/logger bash tools/docker.sh
```
Let's go for 🏃‍♀️running code.

## CIL
### CIFAR-100
25 steps
```commandline
bash tools/dist_train.sh configs/cifar/resnet12_cifar_dist_25.py 8 --seed 0 --deterministic --work-dir /opt/logger/cifar100_25t
```

### CIFAR100-LT
10 steps (shuffled)
```
bash tools/dist_train.sh configs/cifar_lt/resnet_cifar_shuffle_10.py 8 --seed 0 --deterministic --work-dir /opt/logger/cifar100_lt_10t_shuffle
```

### ImageNet100
25 steps
```commandline
bash tools/dist_train.sh configs/imagenet/resnet18_imagenet100_25t.py 8 --seed 0 --deterministic --work-dir /opt/logger/i100_25t
```

### ImageNet100-LT
10 steps (Shuffled)
```commandline
bash tools/dist_train.sh configs/imagenet_lt/resnet18_imagenet100_shuffle_10t.py 8 --seed 0 --deterministic --work-dir /opt/logger/i100_lt_10t_shuffle
```

## UniCIL
To conduct UniCIL, you need to run the base session first and run the incremental sessions beyond the base session checkpoint.

**Base Session:**
```commandline
bash tools_general/dist.sh train_base configs_general/cifar_general/resnet18_cifar_10.py 8 --seed 0 --deterministic --work-dir /opt/logger/general_cifar_10
```
**Incremental Sessions:**
```commandline
bash tools_general/dist.sh train_inc configs_general/cifar_general/resnet18_cifar_10.py 8 --seed 0 --deterministic --work-dir /opt/logger/general_cifar_10
```

## Results
You can cacluate the average of "[ACC_MEAN]" of each session to get the **average incremental accuracy**. Be carefult that "[ACC_MEAN]" is accuracy of a specific session rather than average incremental accuracy in the tables of our paper.

## Citation
If you find this work helpful in your research, please consider referring:
```bibtex
@article{UniCIL,
    author={Yibo Yang and Haobo Yuan and Xiangtai Li and Jianlong Wu and Lefei Zhang and Zhouchen Lin and Philip H.S. Torr and Bernard Ghanem and Dacheng Tao},
    title={Neural Collapse Terminus: A Unified Solution for Class Incremental Learning and Its Variants},
    journal={arXiv pre-print},
    year={2023}
  }
```
