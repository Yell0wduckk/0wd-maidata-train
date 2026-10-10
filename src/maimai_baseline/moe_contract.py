"""Dense, token-gated FFNs; widths count main matrices, not gate/bias overhead."""

def identity(width, dense_hidden, activation, dropout):
    return dict(kind='dense-token-moe', width=width, experts=4,
                expert_hidden=dense_hidden // 2, gate='linear-softmax-fp32',
                gate_init='zeros', activation=activation, dropout=dropout,
                routing='all-valid-tokens-all-experts', auxiliary_loss=None)


def parameter_delta(width, dense_hidden):
    expert_hidden = dense_hidden // 2
    dense = 2 * width * dense_hidden + dense_hidden + width
    experts = 4 * (2 * width * expert_hidden + expert_hidden + width)
    return experts + width * 4 + 4 - dense
