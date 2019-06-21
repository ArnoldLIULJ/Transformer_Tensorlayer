import tensorlayer as tl
import tensorflow as tf
from models import embedding_layer
from models.attention_layer import SelfAttentionLayer, MultiHeadAttentionLayer
from models.feedforward_layer import FeedForwardLayer
from models.model_utils import get_input_mask, get_target_mask, positional_encoding
import models.beam_search as beam_search


class Transformer(tl.models.Model):
    """
    Transormer model.

    Parameters
    ----------
    params: a parameter object, containing hyper-parameter values to construct model
        refer to ../transformer/utils/model_params for details

    Methods
    ----------
    __init__()
        Initializing the model, constructing all essential components
    forward()
        forward pass of the model
    """

    def __init__(self, params):
        super(Transformer, self).__init__()

        self.params = params

        self.embedding_layer = embedding_layer.EmbeddingLayer(
            params.vocab_size, params.hidden_size)
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)
        # self.output_linear = OutputLinear(params)

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = positional_encoding(
            max_decode_length + 1, self.params.hidden_size)
        decoder_self_attention_bias = get_target_mask(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
            ids: Current decoded sequences. int tensor with shape [batch_size *
                beam_size, i + 1]
            i: Loop index
            cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
            Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_layer(decoder_input)
            decoder_input += timing_signal[0][i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            self.decoder_stack.eval()
            # print("decoder_inputs= ", decoder_input.shape)
            # print("encoder_outputs= ", cache["encoder_outputs"].shape)
            
            decoder_outputs = self.decoder_stack(
                inputs=decoder_input,
                features=cache.get("encoder_outputs"),
                input_mask=cache.get("encoder_decoder_attention_bias"),
                target_mask=self_attention_bias,
                cache=cache)
            self.decoder_stack.train()
            # print("decoder_outputs = ", decoder_outputs.shape)
            logits = self.embedding_layer(decoder_outputs)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """Return predicted sequence."""
        batch_size = tf.shape(encoder_outputs)[0]
        input_length = tf.shape(encoder_outputs)[1]
        max_decode_length = input_length + self.params.extra_decode_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(
            max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = tf.zeros([batch_size], dtype=tf.int32)
        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": tf.zeros([batch_size, 0, self.params.hidden_size]),
                "v": tf.zeros([batch_size, 0, self.params.hidden_size]),
            } for layer in range(self.params.encoder_num_layers)}
            # pylint: enable=g-complex-comprehension

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
        # print("encoder_outputs = , bias = ", encoder_outputs.shape, encoder_decoder_attention_bias.shape)
        EOS_ID = 1
        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.params.vocab_size,
            beam_size=self.params.beam_size,
            alpha=self.params.alpha,
            max_decode_length=max_decode_length,
            eos_id=EOS_ID)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}


    def forward(self, inputs, targets=None):
        length = tf.shape(inputs)[1]
        # get padding mask
        input_mask = get_input_mask(inputs)
        

        # Generate continuous representation for inputs.
        inputs = self.embedding_layer(inputs)
        inputs += positional_encoding(length, self.params.hidden_size)
        inputs = tf.nn.dropout(inputs, rate=1 - self.params.keep_prob)
        features = self.encoder_stack(inputs, input_mask=input_mask)
        encoder_outputs = features

        
        if self.is_train:
            targets = self.embedding_layer(targets)
            # shift targets to right
            targets = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
            length = tf.shape(targets)[1]
            target_mask = get_target_mask(length)
            # add positional_encoding
            targets += positional_encoding(length, self.params.hidden_size)
            targets = tf.nn.dropout(targets, rate=1 - self.params.keep_prob)
            
            outputs = self.decoder_stack(inputs=targets, features=features, input_mask=input_mask, target_mask=target_mask)
            outputs = self.embedding_layer(outputs)
        else:
            outputs = self.predict(encoder_outputs, input_mask)

        return outputs


class OutputLinear(tl.layers.Layer):

    def __init__(self, params):
        super(OutputLinear, self).__init__()
        self.params = params

        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        self.W = self._get_weights("W", (self.params.hidden_size, self.params.vocab_size))

    def forward(self, inputs):
        inputs = tf.tensordot(inputs, self.W, axes=[[2], [0]])
        inputs = tf.nn.softmax(inputs)
        return inputs

    def __repr__(self):
        return "output linear layer"





class LayerNormalization(tl.layers.Layer):
    """
    Layer normalization

    Parameters
    ----------
    hidden_size:
        hidden size of features
    epsilon:
        value to prevent division by zero

    """

    def __init__(self, hidden_size, epsilon=1e-6):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        self.scale = self._get_weights('scale', shape=(self.hidden_size), init=tl.initializers.Ones())
        self.bias = self._get_weights('bias', shape=(self.hidden_size), init=tl.initializers.Zeros())

    def forward(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[-1], keepdims=True)
        var = tf.reduce_mean(tf.square(inputs - mean), axis=[-1], keepdims=True)
        norm_inputs = (inputs - mean) * tf.math.rsqrt(var + self.epsilon)
        return norm_inputs * self.scale + self.bias

    def __repr__(self):
        return "layer normalization"


class SublayerWrapper(tl.models.Model):
    """
    wrapper for sublayer(attention, feedforward)
    contains no parameters, so is not a tl layer
    """

    def __init__(self, layer, params):
        super(SublayerWrapper, self).__init__()
        self.params = params

        self.layer = layer
        self.layer_norm = LayerNormalization(params.hidden_size)

    def forward(self, inputs, *args, **kwargs):
        outputs = self.layer(inputs, *args, **kwargs)
        if self.is_train:
            outputs = tf.nn.dropout(outputs, rate=1 - self.params.keep_prob)
        # residual connection
        return self.layer_norm(inputs + outputs)



class EncoderStack(tl.models.Model):
    """
    Encoder stack
    Encoder is made up of self-attn and feed forward

    Parameters
    ----------
    params: a parameter object
        refer to ../transformer/utils/model_params for details

    """

    def __init__(self, params):
        super(EncoderStack, self).__init__()

        self.sublayers = []
        for _ in range(params.encoder_num_layers):
            self.sublayers.append([
                SublayerWrapper(SelfAttentionLayer(params.num_heads, params.hidden_size, params.keep_prob),
                                params),
                SublayerWrapper(FeedForwardLayer(params.hidden_size, params.ff_size, params.keep_prob),
                                params)])

        self.layer_norm = LayerNormalization(params.hidden_size)

    def forward(self, inputs, input_mask):
        """
        Parameters
        ----------
        inputs:
            inputs to the Encoder, shape=(batch_size, length, hidden_size)
        input_mask:
            mask for padding

        Return
        -------
            encoded features, shape=(batch_size, length, hidden_size)
        """
        for sublayer in self.sublayers:
            inputs = sublayer[0](inputs=inputs, mask=input_mask)
            inputs = sublayer[1](inputs)
        inputs = self.layer_norm(inputs)
        return inputs


class DecoderStack(tl.models.Model):
    """
    Decoder stack
    Decoder is made of self-attn, src-attn, and feed forward

    Parameters
    ----------
    params: a parameter object
        refer to ../transformer/utils/model_params for details

    """

    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params

        self.sublayers = []
        for _ in range(self.params.decoder_num_layers):
            self.sublayers.append([
                SublayerWrapper(
                    SelfAttentionLayer(self.params.num_heads, self.params.hidden_size,
                                       self.params.keep_prob), self.params),
                SublayerWrapper(
                    MultiHeadAttentionLayer(self.params.num_heads, self.params.hidden_size,
                                            self.params.keep_prob), self.params),
                SublayerWrapper(
                    FeedForwardLayer(self.params.hidden_size, self.params.ff_size,
                                     self.params.keep_prob), self.params)])
        self.layer_norm = LayerNormalization(self.params.hidden_size)

    def forward(self, inputs, features, input_mask, target_mask, cache=None):
        for n, sublayer in enumerate(self.sublayers):
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            inputs = sublayer[0](inputs, mask=target_mask, cache=layer_cache)
            inputs = sublayer[1](inputs, y=features, mask=input_mask)
            inputs = sublayer[2](inputs)
        inputs = self.layer_norm(inputs)
        return inputs

