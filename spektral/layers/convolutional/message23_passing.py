import inspect

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from spektral.layers.ops.scatter import deserialize_scatter, serialize_scatter
from spektral.utils.keras import (
    deserialize_kwarg,
    is_keras_kwarg,
    is_layer_kwarg,
    serialize_kwarg,
)


class Message3Passing(Layer):
    def __init__(self, aggregate="sum", **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters
        self.agg = deserialize_scatter(aggregate)

    def call(self, inputs, **kwargs):
        x, a2, e2, a3, e3 = self.get_inputs(inputs)
        return self.propagate(x, a2, e2, a3, e3)

    def build(self, input_shape):
        self.built = True

    def propagate(self, x, a2, e2, a3, e3, **kwargs):
        self.n_nodes = tf.shape(x)[-2]
        self.index_i = a3.indices[:, 0]
        self.index_j = a3.indices[:, 1]
        self.index_k = a3.indices[:, 2]

        # Message
        msg_kwargs = self.get_kwargs(x, a2, e2, a3, e3, self.msg_signature, kwargs)
        messages = self.message(x, **msg_kwargs)

        # Aggregate
        agg_kwargs = self.get_kwargs(x, a2, e2, a3, e3, self.agg_signature, kwargs)
        embeddings = self.aggregate(messages, **agg_kwargs)

        # Update
        upd_kwargs = self.get_kwargs(x, a2, e2, a3, e3, self.upd_signature, kwargs)
        output = self.update(embeddings, **upd_kwargs)

        return output

    def message(self, x, **kwargs):
        raise NotImplementedError()

    def aggregate(self, messages, **kwargs):
        raise NotImplementedError()

    def update(self, embeddings, **kwargs):
        return embeddings

    def get_i(self, x):
        return tf.gather(x, self.index_i, axis=-2)

    def get_j(self, x):
        return tf.gather(x, self.index_j, axis=-2)

    def get_k(self, x):
        return tf.gather(x, self.index_k, axis=-2)

    def get_ij(self, e2):
        #TODO
        return tf.gather_nd(e2, self.index_i, axis=-2)

    def get_kwargs(self, x, a2, e2, a3, e3, signature, kwargs):
        output = {}
        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a2":
                output[k] = a2
            elif k == "e2":
                output[k] = e2
            elif k == "a3":
                output[k] = a3
            elif k == "e3":
                output[k] = e3
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError("Missing key {} for signature {}".format(k, signature))

        return output

    @staticmethod
    def get_inputs(inputs):
        """
        Parses the inputs lists and returns a tuple (x, a, e) with node features,
        adjacency matrix and edge features. In the inputs only contain x and a, then
        e=None is returned.
        """
        if len(inputs) == 5:
            x, a2, e2, a3, e3 = inputs
            assert K.ndim(e2) in (2, 3), "E2 must have rank 2 or 3"
            assert K.ndim(e3) in (2, 3), "E3 must have rank 2 or 3" #TODO is this still true?
        else:
            raise ValueError(
                "Expected 5 inputs tensors (X, A2, E2, A3, E3), got {}.".format(len(inputs))
            )
        assert K.ndim(x) in (2, 3), "X must have rank 2 or 3"
        assert K.is_sparse(a2), "A must be a SparseTensor"
        assert K.ndim(a2) == 2, "A must have rank 2"
        assert K.is_sparse(a3), "A must be a SparseTensor"
        assert K.ndim(a3) == 2, "A must have rank 2" #TODO is this still true?

        return x, a2, e2, a3, e3

    def get_config(self):
        mp_config = {"aggregate": serialize_scatter(self.agg)}
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        base_config = super().get_config()

        return {**base_config, **keras_config, **mp_config, **self.config}

    @property
    def config(self):
        return {}

    #@staticmethod
    #def preprocess(a):
    #    return a
