# Optimization on Stiefel manifold (orthonormal constraints) 

This repository contains the code for the paper [Kong, Wang & Tao. Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport. ICLR 2023](https://arxiv.org/abs/2205.14173)

The code implements algorithms that optimize functions of matrices with orthonormal columns. This family of algorithms is derived from a variational approach and is both accurate and efficient.
## What is Stiefel optimization?
A Stiefel manifold $\mathsf{St}(n,m)$ is the set of all $n\text{-by-}m$ matrices satisfying $X^\top X=I$, where $n\ge m$. Stiefel optimization is the process of finding the minimum value of a function $f(X)$ on the Stiefel manifold, i.e.,
$$\min_{X \in \mathsf{St}(n,m)} f(X)=\min_{X \in \mathbb{R}^{n \times m}, s.t. X^\top X=I} f(X),\qquad n\ge m.$$
In other words, if you are trying to optimize a function under orthonormal constraints, then you are in the right place. Our optimizers `StiefelSGD` and `SteifelAdam` in `StiefelOptimizers.py` can be easily applied to your own idea.

![Demo](./demo.gif)

Here are some examples for Stiefel optimization.
### Example: Improve the performance of Transformer models with orthonormal constraints
Our experiments show that applying our Stiefel optimizer to a vanilla Transformer will let it outperform later, fancier model (Table 1 in the paper). To apply this technique to your own model, simply change the optimizer for each $W_i^Q$ and $W_i^K$ weight matrix to our corresponding Stiefel SGD/Adam and use the same hyperparameters as before. Please refer to Sec. 3.2 in our paper for details. 
```python
# put the Euclidean and Stiefel parameters into 2 different list
for name, param in net.named_parameters():
    if 'q.weight' in name or 'k.weight' in name:
        torch.nn.init.orthogonal_(param) # optional
        stiefel_parameters.append(param)
    else:
        euclidean_parameters.append(param)
optimizer_euclidean=torch.optim.Adam(model.parameters)
# apply our StiefelAdam algorithm
optimizer_stiefel=StiefelAdam(stiefel_parameters)
optimizer=CombinedOptimizer(optimizer_euclidean, optimizer_stiefel)
```
By modifying just the above few lines, you can improve your model **WITHOUT tuning any hyperparameters**! (please refer to Remark 1 in our paper)


### Example: Projection Robust Wasserstein (PRW) Distance
Our optimizer can enhance the great idea of Projection Robust Wasserstein (PRW) Distance. Instead of computing the costly  Wasserstein distance between two point clouds in high-dimensional spaces, PRW finds a Stiefel matrix that projects all the points to a lower dimensional subspace, where the Wasserstein distance is computed with significantly less cost. The Stiefel matrix is chosen such that after being projected, the Wasserstein distance is maximized. Thus, PRW involves a Stiefel optimization problem, and our powerful Stiefel optimizer can make this beautiful idea even better. Please refer to Sec. 3.1 in our paper for details.

In fact, our optimizer can be applied to a large class of optimization problems by pursuing solutions in a subspace.
### Generalization of the PRW Distance example: Subspace Pursue 
Subspace pursuing tries to find the best low-dimensional subspace for projection. Suppose we have a dataset $\lbrace x_i \rbrace_{i=1}^k$ with $x_i$ in $\mathbb{R}^n$. Instead of evaluating our function $f(\lbrace x_i\rbrace_{i=1}^k)$ directly, we consider the optimization problem $$\max_{U\in St(n,m)} f(\lbrace U^\top x_i\rbrace_{i=1}^k).$$ We take the maximum of $U$ in the sense that the information is preserved as much as we can with the column of $U$ being a set of the orthonormal basis of the subspace. If we choose $m\ll n$, then this can save many computational resources as well as reduce noise. The aforementioned great idea Projection Robust Wasserstein Distance can be viewed as a special case of subspace pursue by taking $f$ as the Wasserstein distance between 2 point clouds.


## Details of the optimizers
[StiefelOptimizers.py](StiefelOptimizers.py) is the implementation of our proposed Momentum (S)GD and Adam on St(n,m) (Algorithm 1 and 2 in our paper). They can also be used on $\mathsf{SO}(n)$. [utils_StiefelOptimizers.py](utils_StiefelOptimizers.py) contains some auxiliary code. Please put both files in your path when using.

### Momentum (S)GD on Stiefel manifold
The code for `StiefelSGD` corresponds to Algorithm 1, and can also be used for special case of $\mathsf{SO}(n)$ in Algorithm 4. Here are the details:
```python
class StiefelSGD(params, lr=required, momentum=0.9, dampening=0, expm_method='ForwardEuler', inner_prod='Canonical', max_inner_iter=100)
```
Parameters:
- **params**: the parameters to be optimized
- **lr**: learning rate
- **momentum** (float, optional): momentum factor (default: 0.9)
- **dampening** (float, optional): dampening for momentum (default: 0)
- **expm_method** (str in `['MatrixExp', 'Cayley', 'ForwardEuler']`, optional): the method to compute matrix exponential. (default: `'ForwardEuler'`)
- **inner_prod**: (float number less than 1 or string in `['Canonical', 'Euclidean']`, optional): the parameter in the canonical-type metric (defined in Definition 1 in the paper).
- **max_inner_iter** (int, optional): the maximum number of iterations when computing matrix root inversion. (default: 100)

### Adam on Stiefel manifold
The code for `StiefelAdam` corresponds to Algorithm 2, and can also be used for special case of $\mathsf{SO}(n)$ in Algorithm 5. Here are the details:
```python
class StiefelAdam(params, lr=0.001, betas=(0.9,0.99), epsilon=1e-5, expm_method='ForwardEuler', inner_prod='Canonical', max_inner_iter=100)
```
Parameters:
- **params**: the parameters to be optimized
- **lr** (float, optional): learning rate (default: 0.001)
- **betas** (Tuple[float, float], optional): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
- **expm_method**, **inner_prod** and **max_inner_iter** are the same as `StiefelSGD`

Note:
- The only package required to use these optimizers is *Pytorch*.
- Both optimizers inherit from `torch.optim.Optimizers` and have almost the same usage.
- There is no significant difference when further tuning **expm_method**, **inner_prod** and **max_inner_iter**. The default values are sufficient.
- We recommend using the same hyperparameters when the model contains both Euclidean parameters and Stiefel parameters. Please refer to Remark 1 in the paper for details.
- The matrices being optimized should have number of rows $\ge$ number of columns . Otherwise, the matrix will be transposed without warning. For tensors with more than 2 dimensions, all the dimensions will be flattened excepted the first dimension to create a matrix.
- No special orthonormal initialization for Stiefel matrices is required. Commonly used element-wise random Gaussian matrices will work and our optimizer will automatically project it onto the Stiefel manifold. However, explicit initialization using `torch.nn.init.orthogonal_` is still recommended.

## Instructions for reproducing the experiments in the paper
First install packages using the following code: 
```
pip install -r requirements.txt
```
### Projection Robust Wasserstein Distance
Please check the folder ProjectionRobustWasserstein. Run [test_mnist.py](test_mnist.py) and [test_shakespeare.py](test_shakespeare.py) to reproduce the results and use [plot.ipynb](plot.ipynb) to visualize. 
(Modified from [official implementation of Projection Robust Wasserstein Distance](https://github.com/fanchenyou/PRW))
### Vision Transformer (ViT)
Please check the folder ViT. Run [ViT_main.py](ViT_main.py) and use arguments `--label-smoothing` and `--autoaugment` for every optimizer, constraint and dataset. For example: 
```
python ViT_main.py --optim-method StiefelSGD --dataset c10 --constraint OnlyWithin
```

- `optim-method` should be chosen from `['SGD','Adam','RegularizerStiefelSGD', 'RegularizerStiefelAdam', 'ProjectedStiefelSGD', 'ProjectedStiefelAdam', 'StiefelSGD', 'StiefelAdam', 'MomentumlessStiefelSGD']`

- `constraint` should be chosen from `['Across', 'OnlyWithin', None]`

- `dataset` should be chosen from `['c10', 'c100']`

(Modified form the following repositary: [Training process](https://github.com/omihub777/ViT-CIFAR); [model implementation](https://github.com/lucidrains/vit-pytorch))
### Leading eigenvalue problem
Please run [LEV/LEV.ipynb](LEV/LEV.ipynb).


## Citation
Feel free to cite if you want to use these optimizers in your research!

	@inproceedings{kong2022momentum,
        title={Momentum Stiefel Optimizer, with Applications to Suitably-Orthogonal Attention, and Optimal Transport},
        author={Kong, Lingkai and Wang, Yuqing and Tao, Molei},
        booktitle={International Conference on Learning Representations},
        year={2023}
    }
