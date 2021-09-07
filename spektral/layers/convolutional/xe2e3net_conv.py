from collections.abc import Iterable

import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Multiply, PReLU, ReLU, Reshape
from tensorflow.python.ops import gen_sparse_ops

from spektral.layers.convolutional.conv import Conv
from spektral.layers.convolutional.message_passing import MessagePassing


class XE2E3NetConv(MessagePassing):
    def __init__(
        self,
        stack_channels,
        node_channels,
        edge_channels,
        node_activation=None,
        edge_activation=None,
        aggregate: str = "sum",
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super().__init__(
            aggregate=aggregate,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )
        self.stack_channels = stack_channels
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.node_activation = node_activation
        self.edge_activation = edge_activation

    def build(self, input_shape):
        assert len(input_shape) == 3  # X, A, E, right?
        layer_kwargs = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        )

        self.stack_models = []
        self.stack_model_acts = []

        if isinstance(self.stack_channels, Iterable):
            assert len(self.stack_channels) > 0

            for count, value in enumerate(self.stack_channels):
                self.stack_models.append(Dense(value, **layer_kwargs))
                if count != len(self.stack_channels) - 1:
                    self.stack_model_acts.append(ReLU())
                else:
                    self.stack_model_acts.append(PReLU())
        else:
            self.stack_models.append(Dense(self.stack_channels, **layer_kwargs))
            self.stack_model_acts.append(PReLU())

        self.node_model = Dense(
            self.node_channels, activation=self.node_activation, **layer_kwargs
        )
        self.edge_model = Dense(
            self.edge_channels, activation=self.edge_activation, **layer_kwargs
        )

        #attention
        self.e2_pos1_att_sigmoid = Dense(1, activation="sigmoid")
        self.e2_pos1_att_multiply = Multiply()
        self.e2_pos2_att_sigmoid = Dense(1, activation="sigmoid")
        self.e2_pos2_att_multiply = Multiply()

        self.e3_pos1_att_sigmoid = Dense(1, activation="sigmoid")
        self.e3_pos1_att_multiply = Multiply()
        self.e3_pos2_att_sigmoid = Dense(1, activation="sigmoid")
        self.e3_pos2_att_multiply = Multiply()
        self.e3_pos3_att_sigmoid = Dense(1, activation="sigmoid")
        self.e3_pos3_att_multiply = Multiply()

        self.built = True

    def call(self, inputs, **kwargs):
        # TODO we can just use the base class here?
        x, a2, e2, a3, e3 = self.get_inputs(inputs)
        x_out, e2_out, e3_out = self.propagate(x, a2, e2, a3, e3)

        return x_out, e2_out, e3_out

    def message(self, x, e2, e3):
        x_2i = self.get2_i(x)  # Features of self
        x_2j = self.get2_j(x)  # Features of neighbours

        x_3i = self.get3_i(x)  # Features of self
        x_3j = self.get3_j(x)  # Features of neighbours
        x_3k = self.get3_k(x)  # Features of neighbours

        # Features of outgoing edges are simply the edge features
        e2_ij  = e2
        e3_ijk = e3

        # Features of incoming edges j -> i are obtained by transposing the edge features.
        # Since TF does not allow transposing sparse matrices with rank > 2, we instead
        # re-order a tf.range(n_edges) and use it as indices to re-order the edge
        # features.
        # The following two instructions are the sparse equivalent of
        #     tf.transpose(E, perm=(1, 0, 2))
        # where E has shape (N, N, S).
        reorder_idx = gen_sparse_ops.sparse_reorder(
            tf.stack([self.index2_i, self.index2_j], axis=-1),
            tf.range(tf.shape(e2)[0]),
            (self.n_nodes, self.n_nodes),
        )[1]
        e2_ji = tf.gather(e2, reorder_idx)


        # Concatenate the features and feed to first MLP
        stack_ijk = tf.concat(
            [x_i, x_j, x_k, e2_ij, e2_ji, e_ijk], axis=-1
        )  # Shape: (n_edges, F + F + S + S)

        for stack_conv in range(0, len(self.stack_models)):
            stack_ijk = self.stack_models[stack_conv](stack_ijk)
            stack_ijk = self.stack_model_acts[stack_conv](stack_ijk)

        return stack_ijk

    def aggregate(self, messages, x):
        pos1_att = self.pos1_att_sigmoid(messages)
        pos1 = self.pos1_att_multiply([pos1_att, messages])
        pos1 = self.agg(pos1, self.index_i, self.n_nodes)

        pos2_att = self.pos2_att_sigmoid(messages)
        pos2 = self.pos2_att_multiply([pos2_att, messages])
        pos2 = self.agg(pos2, self.index_j, self.n_nodes)

        pos3_att = self.pos3_att_sigmoid(messages)
        pos3 = self.pos3_att_multiply([pos3_att, messages])
        pos3 = self.agg(pos3, self.index_k, self.n_nodes)

        return tf.concat([x, pos1, pos2, pos3], axis=-1), messages

    def update(self, embeddings):
        x_new, stack_ijk = embeddings

        return self.node_model(x_new), self.edge_model(stack_ijk)

    @property
    def config(self):
        return {
            "stack_channels": self.stack_channels,
            "node_channels": self.node_channels,
            "edge_channels": self.edge_channels,
            "node_activation": self.node_activation,
            "edge_activation": self.edge_activation,
        }


