import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from spektral.layers import XENetConv, XENetDenseConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor

def test_sparse_model_sizes():
    """
    This is a sanity check to make sure we have the same number of operations that we intend to have
    """
    N = 5
    F = 4
    S = 3
    X_in = Input(shape=(F,), name="X_in")
    A_in = Input(shape=(None,), name="A_in", sparse=True)
    E_in = Input(shape=(S,), name="E_in")

    x = np.ones(shape=(N, F))
    a = np.ones(shape=(N, N, N))

    #          1  2
    a_dimi = [ 0, 4, ]
    a_dimj = [ 1, 2, ]
    a_dimk = [ 2, 3, ]

    a_val  = [
        1, # 1
        1, # 2
    ]

    a = sp.COO((a_val, (a_dimi, a_dimj, a_dimk)), shape=(N,N,N))

    e_vals = [
        [ 1, 2, 3 ], # 1
        [ 3, 2, 1 ], # 2
    ]

    a = sp_matrix_to_sp_tensor(a)
    e = np.ones(shape=(N * N, S))

    def assert_n_params(inp, out, expected_size):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="mean_squared_error")
        print(model.count_params())
        assert model.count_params() == expected_size
        # for test coverage:
        model([x, a, e])

    X, E = XENetConv([5], 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # int vs list: 5 vs [5]
    X, E = XENetConv(5, 10, 20, False)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 350)
    # t = (4+4+3+3+1)*5 =  75    # Stack Conv
    # x = (4+5+5+1)*10  = 150    # Node reduce
    # e = (5+1)*20      = 120    # Edge reduce
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 350

    X, E = XENetConv(5, 10, 20, True)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 362)
    # t = (4+4+3+3+1)*5 =  75
    # a = (5+1)*1   *2  =  12    # Attention
    # x = (4+5+5+1)*10  = 150
    # e = (5+1)*20      = 120
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 362

    X, E = XENetConv([50, 5], 10, 20, True)([X_in, A_in, E_in])
    assert_n_params([X_in, A_in, E_in], [X, E], 1292)
    # t1 = (4+4+3+3+1)*50   =  750
    # t2 = (50+1)*5         =  255
    # a = (5+1)*1   *2      =   12    # Attention
    # x = (4+5+5+1)*10      =  150
    # e = (5+1)*20          =  120
    # p                     =    5    # Prelu
    # total = t+x+e+p       = 1292


if __name__ == "__main__":
    test_sparse_model_sizes()
