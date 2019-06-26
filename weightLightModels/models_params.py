class TINY_PARAMS(object):
    filter_number = 64
    n_channels = 2
    n_units = 128
    H = 2


    light_filter_size=(1,3)
    filter_size = light_filter_size[-1]
    vocab_size = 50

    encoder_num_layers = 2
    decoder_num_layers = 2

    hidden_size = 64
    ff_size = 16
    num_heads = 4
    keep_prob = 0.9


    # Default prediction params
    extra_decode_length=5
    beam_size=2
    alpha=0.6 # used to calculate length normalization in beam search
