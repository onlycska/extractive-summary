from typing import Dict

import torch
from torch.nn.functional import sigmoid, relu, elu, tanh
from torch.nn import Module, LSTM, RNN, GRU, Linear, Sequential, Dropout


class RecurrentClassifier(Module):
    def __init__(self, config: Dict, emb_matrix: torch.Tensor):
        """
        Baseline classifier, hyperparameters are passed in `config`.
        Consists of recurrent part and a classifier (Multilayer Perceptron) part
        Keys are:
            - freeze: whether word embeddings should be frozen
            - cell_type: one of: RNN, GRU, LSTM, which recurrent cell model should use
            - hidden_size: size of hidden state for recurrent cell
            - num_layers: amount of recurrent cells in the model
            - cell_dropout: dropout rate between recurrent cells (not applied if model has only one cell!)
            - bidirectional: boolean, whether to use unidirectional of bidirectional model
            - out_activation: one of: "sigmoid", "tanh", "relu", "elu". Activation in classifier part
            - out_dropout: dropout rate in classifier part
            - out_sizes: List[int], hidden size of each layer in classifier part. Empty list means that final
                layer is attached directly to recurrent part output
        :param config: configuration of model
        :param emb_matrix: embeddings matrix from `prepare_emb_matrix`
        """
        super().__init__()
        self.config = config
        cell_types = {
            "RNN": RNN,
            "GRU": GRU,
            "LSTM": LSTM}
        cell_class = cell_types[config["cell_type"]]
        self.cell = cell_class(input_size=emb_matrix.size(1),
                               batch_first=True,
                               hidden_size=config["hidden_size"],
                               num_layers=config["num_layers"],
                               dropout=config["cell_dropout"],
                               bidirectional=config["bidirectional"],
                               )
        activation_types = {
            "sigmoid": sigmoid,
            "tanh": tanh,
            "relu": relu,
            "elu": elu,
        }
        self.out_activation = activation_types[config["out_activation"]]
        self.out_dropout = Dropout(config["out_dropout"])
        cur_out_size = config["hidden_size"] * config["num_layers"]
        if config["bidirectional"]:
            cur_out_size *= 2
        out_layers = []
        for cur_hidden_size in config["out_sizes"]:
            out_layers.append(Linear(cur_out_size, cur_hidden_size))
            cur_out_size = cur_hidden_size
        out_layers.append(Linear(cur_out_size, 2))
        self.out_proj = Sequential(*out_layers)

    def forward(self, input):
        batch_emb_matrix = input.data.reshape(input.batch_sizes[0], 1, 768)
        _, last_state = self.cell(batch_emb_matrix)
        if isinstance(last_state, tuple):
            last_state = last_state[0]
        last_state = last_state.transpose(0, 1)
        last_state = last_state.reshape(last_state.size(0), -1)
        return self.out_proj(last_state)

