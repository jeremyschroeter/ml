import numpy as np

def softmax(x: np.ndarray, temp: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.exp(x).sum()

def logsumexp(x: np.ndarray) -> np.ndarray:
    return np.log(np.exp(x).sum())

def quad_form(x: np.ndarray, A: np.ndarray, x_2: np.ndarray = None) -> np.ndarray:
    
    # if x.shape[1] != 1:
    #     raise ValueError(f'"x" cannot have shape {x.shape}')
    # elif x_2 is not None and x_2.shape[1] != 1:
    #     raise ValueError(f'"x_2" cannot have shape {x_2.shape}')

    if x_2 is None:
        return x.T @ A @ x
    else:
        return x.T @ A @ x_2

# def marginal_y(
#         A: np.ndarray,
#         x: np.ndarray,
#         mu: np.ndarray,
#         b: np.ndarray,
#         inv_lmda: np.ndarray,
#         inv_L: np.ndarray
# ) -> np.ndarray:
#     '''
#     returns the marginal p(y) of a gaussian joint distribution p(x, y) if
    
#     p(x) = N(mu, inv_lmda)
#     p(y | x) = N(Ax + b, inv_L)
    
#     see Bishop 2.3.3 for details
#     '''

