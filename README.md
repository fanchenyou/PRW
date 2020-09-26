# "Projection Robust Wasserstein Distance and Riemannian Optimization." in NeurIPS'20


## Requirements
```
python 3.7+, Numpy, Scikit-learn
```

## Disclaimer
This repository is modified from https://github.com/francoispierrepaty/SubspaceRobustWasserstein
Please cite these papers if you use any part of the code

```
Paty, F. & Cuturi, M.. (2019). Subspace Robust Wasserstein Distances. 
Proceedings of the 36th International Conference on Machine Learning, in PMLR 97:5072-5081
```
```
Tianyi Lin, Chenyou Fan, Nhat Ho, Marco Cuturi, Michael I. Jordan. "Projection Robust Wasserstein Distance and Riemannian Optimization."
Conference on Neural Information Processing Systems 2020 (NeurIPS'20)
```

### Experiment 1, Hypercube:
```
python exp1_hypercube.py
python exp1_hypercube_dim_k.py
python exp1_hypercube_plan_visualize.py
```

### Experiment 2, Noise:
```
python exp2_noise.py
python exp2_noise_level.py
```

### Experiment 3, Word Embedding:
```
python exp3_cinema.py
python exp3_shakespeare.py
```

### Experiment 4, Computation Time:
Compute time of SRW and PRW
```
python exp4_computation_time.py
```

Compute time of PRW with different learning rates
```
python exp4_robust_lr_time.py
python exp4_robust_lr_time_2.py
```

### Experiment 5, MNIST projection:
Please install [Pytorch 1.4+](https://pytorch.org/) for this experiment.
```
python exp5_mnist.py
```