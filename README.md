# "Projection Robust Wasserstein Distance and Riemannian Optimization." in NeurIPS'20


## Requirements
```
python 3.7+, Numpy, Scikit-learn
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