- ProjectedStiefelOptimizer: this constains SGD and Adam method on Stiefel manifold in [2] that is used and termed as `Projected Stiefel SGD/Adam' in our paper. Modified from the official implementation [3]
- StiefelRegularizer.py: this contains Stiefel SGD and Adam by regularizer.
- MomentumlessStiefelSGD.py: this constains (S)GD on Stiefel manifold in [1], which is termed as `Momentumless Stiefel SGD' in our paper.
- LiCombinedOptimizer.py: a modified version of ProjectedStiefelOptimizer.py in [2]. Applying our retraction to optimizer in [2], only used in leading eigenvalue test. See Fig.6 in our paper for details.

[1] [A feasible method for optimization with orthogonality constraints](https://link.springer.com/article/10.1007/s10107-012-0584-1)

[2] [Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform](https://arxiv.org/abs/2002.01113)

[3] https://github.com/JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform