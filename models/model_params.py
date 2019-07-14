class EXAMPLE_PARAMS(object):
    vocab_size = 33945

    encoder_num_layers = 6
    decoder_num_layers = 6
    hidden_size = 512
    ff_size = 2048
    num_heads = 8
    keep_prob = 0.9

    # Default prediction params
    extra_decode_length=50
    beam_size=4
    alpha=0.6 # used to calculate length normalization in beam search

    label_smoothing=0.1
    learning_rate=2.0
    learning_rate_decay_rate=1.0
    learning_rate_warmup_steps=16000

    # Optimizer params
    optimizer_adam_beta1=0.9
    optimizer_adam_beta2=0.997
    optimizer_adam_epsilon=1e-09




class TINY_PARAMS(object):
    vocab_size = 50
    encoder_num_layers = 4
    decoder_num_layers = 4
    filter_number = 256
    R1 = 4
    R2 = 8
    n_channels = 2
    n_units = 128
    H = 32
    light_filter_size=(1,3)
    filter_size = light_filter_size[-1]
    hidden_size = 64
    ff_size = 16
    num_heads = 4
    keep_prob = 0.9



    # Default prediction params
    extra_decode_length=5
    beam_size=2
    alpha=0.6 # used to calculate length normalization in beam search



class EXAMPLE_PARAMS_v4(object):
    vocab_size = 33945

    filter_number = 256
    n_units = 1024
    H = 4

    light_filter_size=(1,16)
    filter_size = light_filter_size[-1]
    encoder_num_layers = 2
    decoder_num_layers = 2

    hidden_size = 256
    ff_size = 512
    num_heads = 8
    keep_prob = 0.8

    # Default prediction params
    extra_decode_length=50
    beam_size=4
    alpha=0.6 # used to calculate length normalization in beam search

    label_smoothing=0.1
    learning_rate=2.0
    learning_rate_decay_rate=1.0
    learning_rate_warmup_steps=16000

    # Optimizer params
    optimizer_adam_beta1=0.9
    optimizer_adam_beta2=0.997
    optimizer_adam_epsilon=1e-09




