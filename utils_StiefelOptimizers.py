import torch

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()


def matrix_square_root(mat_a, mat_a_size, iter_count=100, ridge_epsilon=1e-4):
  """
  Stable iterations for the matrix square root, Nicholas J. Higham
  Page 231, Eq 2.6b
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.6.8799&rep=rep1&type=pdf

  Modified from TensorFlow implementation of https://www.tensorflow.org/api_docs/python/tf/linalg/sqrtm
  """

  def _iter_body(i, mat_y, unused_old_mat_y, mat_z, unused_old_mat_z, err,
                 unused_old_err):
    current_iterate = 0.5 * (3.0 * identity - torch.matmul(mat_z, mat_y))
    current_mat_y = torch.matmul(mat_y, current_iterate)
    current_mat_z = torch.matmul(current_iterate, mat_z)
    # Compute the error in approximation.
    mat_sqrt_a = current_mat_y * torch.sqrt(norm)
    mat_a_approx = torch.matmul(mat_sqrt_a, mat_sqrt_a)
    residual = mat_a - mat_a_approx
    current_err = torch.norm(residual, p=2) / norm
    return i + 1, current_mat_y, mat_y, current_mat_z, mat_z, current_err, err

  identity = torch.eye(mat_a_size, device=mat_a.device, dtype=mat_a.dtype)
  mat_a = mat_a + ridge_epsilon * identity
  norm = torch.norm(mat_a, p=2)
  mat_init_y = mat_a / norm
  mat_init_z = identity
  init_err = norm

  func_input=[0, mat_init_y, mat_init_y, mat_init_z, mat_init_z, init_err, init_err + 1.0]
  for _ in range(iter_count):
    func_input=_iter_body(*func_input)
  return func_input[2] * torch.sqrt(norm), func_input[4] / torch.sqrt(norm)
def matrix_root(A):
  A_root, _ = matrix_square_root(A, A.shape[0], ridge_epsilon=0)
  return A_root

def matrix_root_inv(A, iter_count=100):
  _, A_root_inv = matrix_square_root(A, A.shape[0], ridge_epsilon=0, iter_count=iter_count)
  return A_root_inv

### Compute matrix root inversion by SVD. Super expensive. For debug only
def mat_root_inv_for_debug(A):
    D, U=torch.symeig(A, eigenvectors=True)
    return U@torch.diag(1/torch.sqrt(D))@U.t()

def cayley(Y, alpha=1.0):
    return torch.linalg.inv(torch.eye(Y.shape[0],device=Y.device, dtype=Y.dtype).add(Y, alpha=-alpha/2))@(torch.eye(Y.shape[0],device=Y.device, dtype=Y.dtype).add(Y, alpha=alpha/2))