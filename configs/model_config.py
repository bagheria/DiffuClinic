from transformers import PretrainedConfig

class CustomTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        num_layers=32,
        num_heads=32,
        prediction_chunk=256,
        dropout=0,
        max_position_embeddings=4096,
        masking_type="bidirectional",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.prediction_chunk = prediction_chunk
        self.max_position_embeddings = max_position_embeddings
        self.input_size = prediction_chunk  # alias
        self.masking_type = masking_type
