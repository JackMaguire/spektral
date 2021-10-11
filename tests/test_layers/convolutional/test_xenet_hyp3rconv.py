import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import tensorflow as tf

from spektral.layers import XENetHyp3rConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor

def test_sparse_model_sizes():
    """
    This is a sanity check to make sure we have the same number of operations that we intend to have
    """
    N = 5
    F = 4
    S = 3
    X_in = Input(shape=(None,F,), name="X_in")
    A3_in = Input(shape=(None,None,), name="A3_in", sparse=True)
    E3_in = Input(shape=(None,S,), name="E3_in")

    x = np.ones(shape=(1, N, F))
    #x = tf.Variable( x, shape=x.shape, dtype='float32' )

    e_vals = [ [
        [ 1, 2, 3 ], # 1
        [ 3, 2, 1 ], # 2
    ], ]

    a = tf.sparse.SparseTensor(indices=[ [0, 0, 1, 2], [0, 4, 2, 3] ], values=[1, 1], dense_shape=[1, 5, 5, 5])
    print( "a.indices.shape", a.indices.shape )
    print( "a.shape", a.shape )
    #a = tf.sparse.SparseTensor(indices=[ [0, 4], [1, 2], [2, 3] ], values=[1, 1], dense_shape=[5, 5, 5])
    #e = tf.Variable( e_vals, shape=(2,S), dtype='float32' )
    e = np.asarray( e_vals )
    print( "e:", e )

    #Ensure compilability
    test_model = Model(inputs=[X_in,E3_in], outputs=[Dense(1)(X_in),Dense(1)(E3_in)])
    test_model.compile(optimizer="adam", loss="mean_squared_error")

    print( type(x), x.shape )
    print( type(e), e.shape )

    test_pred = test_model.predict( [x, e] )
    print( "test_pred:", test_pred )

    def assert_n_params(inp, out, expected_size):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer="adam", loss="mean_squared_error")
        print(model.count_params())
        assert model.count_params() == expected_size
        # for test coverage:
        pred = model.predict([x, a, e ])
        print( pred )

    print( "!!! 1 !!!" )
    X, E = XENetHyp3rConv([5], 10, 20, only_update_i=False)([X_in, A3_in, E3_in])
    assert_n_params([X_in, A3_in, E3_in], [X, E], 423)
    # int vs list: 5 vs [5]
    print( "!!! 2 !!!" )
    X, E = XENetHyp3rConv(5, 10, 20, only_update_i=False)([X_in, A3_in, E3_in])
    assert_n_params([X_in, A3_in, E3_in], [X, E], 423)
    # t = (4+4+4+3+1)*5 =  80    # Stack Conv
    # x = (4+5+5+5+1)*10= 200    # Node reduce
    # e = (5+1)*20      = 120    # Edge reduce
    # a = (5+1)*1   *3  =  18    # Attention
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 423

    print( "!!! 3 !!!" )
    X, E = XENetHyp3rConv([50, 5], 10, 20)([X_in, A3_in, E3_in])
    assert_n_params([X_in, A3_in, E3_in], [X, E], 1398)
    # t1 = (4+4+4+3+1)*50   =  800
    # t2 = (50+1)*5         =  255
    # a = (5+1)*1   *3      =   18    # Attention
    # x = (4+5+5+5+1)*10    =  200
    # e = (5+1)*20          =  120
    # p                     =    5    # Prelu
    # total = t+x+e+p       = 1398

    X, E = XENetHyp3rConv(5, 10, 20, only_update_i=True)([X_in, A3_in, E3_in])
    assert_n_params([X_in, A3_in, E3_in], [X, E], 311)
    # t = (4+4+4+3+1)*5 =  80    # Stack Conv
    # x = (4+5+1)*10    = 100    # Node reduce
    # e = (5+1)*20      = 120    # Edge reduce
    # a = (5+1)*1   *1  =   6    # Attention
    # p                 =   5    # Prelu
    # total = t+x+e+p   = 311

if __name__ == "__main__":
    test_sparse_model_sizes()
