class EXAMPLE_PARAMS(object):
    vocab_size = 33945

    encoder_num_layers = 2
    decoder_num_layers = 2

    hidden_size = 216
    ff_size = 512
    num_heads = 8
    keep_prob = 0.8



    # Default prediction params
    extra_decode_length=10
    beam_size=2
    alpha=0.6 # used to calculate length normalization in beam search


class tiny_PARAMS(object):
    vocab_size = 20

    encoder_num_layers = 1
    decoder_num_layers = 1

    hidden_size = 64
    ff_size = 16
    num_heads = 4
    keep_prob = 0.9



    # Default prediction params
    extra_decode_length=5
    beam_size=2
    alpha=0.6 # used to calculate length normalization in beam search
