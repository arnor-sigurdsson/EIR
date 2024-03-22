.. _external-sequence-models:

Sequence Models
===============

This page contains the list of external sequence models that can be used with EIR, coming from the excellent `Transformers <https://huggingface.co/docs/transformers/index>`__ library.

There are 3 ways to use these models:

- Configure and train specific architectures (e.g. BERT with chosen number of layers) from scratch.
- Train a specific architecture (e.g. ``bert-base-uncased``) from scratch.
- Use a pre-trained model (e.g. ``bert-base-uncased``) and fine-tune it.

Please refer to `this page <https://huggingface.co/models>`__ for a complete list of pre-defined architectures, with the option of using pre-trained weights.

Configurable Models
-------------------

The following models can be configured and trained from scratch.

The model type is specified in the ``model_type`` field of the configuration, while the model specific configuration is specified in the ``model_init_config`` field.

For example, the ``LongFormer`` architecture includes the ``num_attention_heads`` and ``num_hidden_layers`` parameters, and can be configured as follows:

.. literalinclude:: ../tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial/04_imdb_input_longformer.yaml
    :language: yaml
    :caption: input_configurable_sequence_model.yaml

**Pretrained Models**

We can also fine-tune or train a specific architecture from scratch. For example, a ``tiny-bert`` model like so:

.. literalinclude:: ../tutorials/tutorial_files/a_using_eir/04_pretrained_sequence_tutorial/04_imdb_input_tiny-bert.yaml
    :language: yaml
    :caption: input_pre_trained_sequence_model.yaml

Below is a list of the configurable models that can be used with EIR.

.. class:: transformers.models.albert.configuration_albert.AlbertConfig(vocab_size=30000, embedding_size=128, hidden_size=4096, num_hidden_layers=12, num_hidden_groups=1, num_attention_heads=64, intermediate_size=16384, inner_group_num=1, hidden_act='gelu_new', hidden_dropout_prob=0, attention_probs_dropout_prob=0, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, classifier_dropout_prob=0.1, position_embedding_type='absolute', pad_token_id=0, bos_token_id=2, eos_token_id=3, **kwargs)

The ALBERT model was proposed in `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations <https://arxiv.org/abs/1909.11942>`__ by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma,
Radu Soricut. It presents two parameter-reduction techniques to lower memory consumption and increase the training
speed of BERT:

- Splitting the embedding matrix into two smaller matrices.
- Using repeating layers split among groups.

The abstract from the paper is the following:

*Increasing model size when pretraining natural language representations often results in improved performance on
downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations,
longer training times, and unexpected model degradation. To address these problems, we present two parameter-reduction
techniques to lower memory consumption and increase the training speed of BERT. Comprehensive empirical evidence shows
that our proposed methods lead to models that scale much better compared to the original BERT. We also use a
self-supervised loss that focuses on modeling inter-sentence coherence, and show it consistently helps downstream tasks
with multi-sentence inputs. As a result, our best model establishes new state-of-the-art results on the GLUE, RACE, and
SQuAD benchmarks while having fewer parameters compared to BERT-large.*

Tips:

- ALBERT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather
  than the left.
- ALBERT uses repeating layers which results in a small memory footprint, however the computational cost remains
  similar to a BERT-like architecture with the same number of hidden layers as it has to iterate through the same
  number of (repeating) layers.
- Embedding size E is different from hidden size H justified because the embeddings are context independent (one embedding vector represents one token), whereas hidden states are context dependent (one hidden state represents a sequence of tokens) so it's more logical to have H >> E. Also, the embedding matrix is large since it's V x E (V being the vocab size). If E < H, it has less parameters.
- Layers are split in groups that share parameters (to save memory).
Next sentence prediction is replaced by a sentence ordering prediction: in the inputs, we have two sentences A and B (that are consecutive) and we either feed A followed by B or B followed by A. The model must predict if they have been swapped or not.


This model was contributed by `lysandre <https://huggingface.co/lysandre>`__. This model jax version was contributed by
`kamalkraj <https://huggingface.co/kamalkraj>`__. The original code can be found `here <https://github.com/google-research/ALBERT>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30000):
        Vocabulary size of the ALBERT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``AlbertModel`` or ``TFAlbertModel``.
    embedding_size (`int`, *optional*, defaults to 128):
        Dimensionality of vocabulary embeddings.
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_hidden_groups (`int`, *optional*, defaults to 1):
        Number of groups for the hidden layers, parameters in the same group are shared.
    num_attention_heads (`int`, *optional*, defaults to 64):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 16384):
        The dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    inner_group_num (`int`, *optional*, defaults to 1):
        The number of inner repetition of attention and ffn.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``AlbertModel`` or ``TFAlbertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for attached classifiers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    pad_token_id (`int`, *optional*, defaults to 0):
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 2):
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 3):
        End of stream token id.

.. class:: transformers.models.bart.configuration_bart.BartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False, use_cache=True, num_labels=3, pad_token_id=1, bos_token_id=0, eos_token_id=2, is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2, **kwargs)

The Bart model was proposed in `BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation,
Translation, and Comprehension <https://arxiv.org/abs/1910.13461>`__ by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan
Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.

According to the abstract,

- Bart uses a standard seq2seq/machine translation architecture with a bidirectional encoder (like BERT) and a
  left-to-right decoder (like GPT).
- The pretraining task involves randomly shuffling the order of the original sentences and a novel in-filling scheme,
  where spans of text are replaced with a single mask token.
- BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. It
  matches the performance of RoBERTa with comparable training resources on GLUE and SQuAD, achieves new
  state-of-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains
  of up to 6 ROUGE.

Tips:

- BART is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- Sequence-to-sequence model with an encoder and a decoder. Encoder is fed a corrupted version of the tokens, decoder is fed the original tokens (but has a mask to hide the future words like a regular transformers decoder). A composition of the following transformations are applied on the pretraining tasks for the encoder:

  * mask random tokens (like in BERT)
  * delete random tokens
  * mask a span of k tokens with a single mask token (a span of 0 tokens is an insertion of a mask token)
  * permute sentences
  * rotate the document to make it start at a specific token

This model was contributed by `sshleifer <https://huggingface.co/sshleifer>`__. The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/bart>`__.


#Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the BART model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``BartModel`` or ``TFBartModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    num_labels (`int`, *optional*, defaults to 3):
        The number of labels to use in ``BartForSequenceClassification``.
    forced_eos_token_id (`int`, *optional*, defaults to 2):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.

.. class:: transformers.models.bert.configuration_bert.BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The BERT model was proposed in `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a
bidirectional transformer pretrained using a combination of masked language modeling objective and next sentence
prediction on a large corpus comprising the Toronto Book Corpus and Wikipedia.

The abstract from the paper is the following:

*We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations
from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional
representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result,
the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models
for a wide range of tasks, such as question answering and language inference, without substantial task-specific
architecture modifications.*

*BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural
language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI
accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute
improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).*

Tips:

- BERT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- BERT was trained with the masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is
  efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation.
- Corrupts the inputs by using random masking, more precisely, during pretraining, a given percentage of tokens (usually 15%) is masked by:

    * a special mask token with probability 0.8
    * a random token different from the one masked with probability 0.1
    * the same token with probability 0.1
    
- The model must predict the original sentence, but has a second objective: inputs are two sentences A and B (with a separation token in between). With probability 50%, the sentences are consecutive in the corpus, in the remaining 50% they are not related. The model has to predict if the sentences are consecutive or not.



This model was contributed by `thomwolf <https://huggingface.co/thomwolf>`__. The original code can be found `here <https://github.com/google-research/bert>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``BertModel`` or ``TFBertModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``BertModel`` or ``TFBertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.bert_generation.configuration_bert_generation.BertGenerationConfig(vocab_size=50358, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, bos_token_id=2, eos_token_id=1, position_embedding_type='absolute', use_cache=True, **kwargs)

The BertGeneration model is a BERT model that can be leveraged for sequence-to-sequence tasks using
``EncoderDecoderModel`` as proposed in `Leveraging Pre-trained Checkpoints for Sequence Generation
Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

The abstract from the paper is the following:

*Unsupervised pretraining of large neural models has recently revolutionized Natural Language Processing. By
warm-starting from the publicly released checkpoints, NLP practitioners have pushed the state-of-the-art on multiple
benchmarks while saving significant amounts of compute time. So far the focus has been mainly on the Natural Language
Understanding tasks. In this paper, we demonstrate the efficacy of pre-trained checkpoints for Sequence Generation. We
developed a Transformer-based sequence-to-sequence model that is compatible with publicly available pre-trained BERT,
GPT-2 and RoBERTa checkpoints and conducted an extensive empirical study on the utility of initializing our model, both
encoder and decoder, with these checkpoints. Our models result in new state-of-the-art results on Machine Translation,
Text Summarization, Sentence Splitting, and Sentence Fusion.*

.. class:: transformers.models.big_bird.configuration_big_bird.BigBirdConfig(vocab_size=50358, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu_new', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=4096, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2, sep_token_id=66, attention_type='block_sparse', use_bias=True, rescale_embeddings=False, block_size=64, num_random_blocks=3, classifier_dropout=None, **kwargs)

The BigBird model was proposed in `Big Bird: Transformers for Longer Sequences <https://arxiv.org/abs/2007.14062>`__ by
Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon,
Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others. BigBird, is a sparse-attention
based transformer which extends Transformer based models, such as BERT to much longer sequences. In addition to sparse
attention, BigBird also applies global attention as well as random attention to the input sequence. Theoretically, it
has been shown that applying sparse, global, and random attention approximates full attention, while being
computationally much more efficient for longer sequences. As a consequence of the capability to handle longer context,
BigBird has shown improved performance on various long document NLP tasks, such as question answering and
summarization, compared to BERT or RoBERTa.

The abstract from the paper is the following:

*Transformers-based models, such as BERT, have been one of the most successful deep learning models for NLP.
Unfortunately, one of their core limitations is the quadratic dependency (mainly in terms of memory) on the sequence
length due to their full attention mechanism. To remedy this, we propose, BigBird, a sparse attention mechanism that
reduces this quadratic dependency to linear. We show that BigBird is a universal approximator of sequence functions and
is Turing complete, thereby preserving these properties of the quadratic, full attention model. Along the way, our
theoretical analysis reveals some of the benefits of having O(1) global tokens (such as CLS), that attend to the entire
sequence as part of the sparse attention mechanism. The proposed sparse attention can handle sequences of length up to
8x of what was previously possible using similar hardware. As a consequence of the capability to handle longer context,
BigBird drastically improves performance on various NLP tasks such as question answering and summarization. We also
propose novel applications to genomics data.*

Tips:

- For an in-detail explanation on how BigBird's attention works, see `this blog post <https://huggingface.co/blog/big-bird>`__.
- BigBird comes with 2 implementations: **original_full** & **block_sparse**. For the sequence length < 1024, using
  **original_full** is advised as there is no benefit in using **block_sparse** attention.
- The code currently uses window size of 3 blocks and 2 global blocks.
- Sequence length must be divisible by block size.
- Current implementation supports only **ITC**.
- Current implementation doesn't support **num_random_blocks = 0**
- BigBird is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.

This model was contributed by `vasudevgupta <https://huggingface.co/vasudevgupta>`__. The original code can be found
`here <https://github.com/google-research/bigbird>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50358):
        Vocabulary size of the BigBird model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``BigBirdModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 4096):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 1024 or 2048 or 4096).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``BigBirdModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    attention_type (`str`, *optional*, defaults to `"block_sparse"`)
        Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
        layer (with n^2 complexity). Possible values are `"original_full"` and `"block_sparse"`.
    use_bias (`bool`, *optional*, defaults to `True`)
        Whether to use bias in query, key, value.
    rescale_embeddings (`bool`, *optional*, defaults to `False`)
        Whether to rescale embeddings with (hidden_size ** 0.5).
    block_size (`int`, *optional*, defaults to 64)
        Size of each block. Useful only when `attention_type == "block_sparse"`.
    num_random_blocks (`int`, *optional*, defaults to 3)
        Each query is going to attend these many number of random blocks. Useful only when `attention_type ==
        "block_sparse"`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.bigbird_pegasus.configuration_bigbird_pegasus.BigBirdPegasusConfig(vocab_size=96103, max_position_embeddings=4096, encoder_layers=16, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=16, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu_new', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=2, classifier_dropout=0.0, scale_embedding=True, pad_token_id=0, bos_token_id=2, eos_token_id=1, attention_type='block_sparse', block_size=64, num_random_blocks=3, use_bias=False, **kwargs)

The BigBird model was proposed in `Big Bird: Transformers for Longer Sequences <https://arxiv.org/abs/2007.14062>`__ by
Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon,
Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others. BigBird, is a sparse-attention
based transformer which extends Transformer based models, such as BERT to much longer sequences. In addition to sparse
attention, BigBird also applies global attention as well as random attention to the input sequence. Theoretically, it
has been shown that applying sparse, global, and random attention approximates full attention, while being
computationally much more efficient for longer sequences. As a consequence of the capability to handle longer context,
BigBird has shown improved performance on various long document NLP tasks, such as question answering and
summarization, compared to BERT or RoBERTa.

The abstract from the paper is the following:

*Transformers-based models, such as BERT, have been one of the most successful deep learning models for NLP.
Unfortunately, one of their core limitations is the quadratic dependency (mainly in terms of memory) on the sequence
length due to their full attention mechanism. To remedy this, we propose, BigBird, a sparse attention mechanism that
reduces this quadratic dependency to linear. We show that BigBird is a universal approximator of sequence functions and
is Turing complete, thereby preserving these properties of the quadratic, full attention model. Along the way, our
theoretical analysis reveals some of the benefits of having O(1) global tokens (such as CLS), that attend to the entire
sequence as part of the sparse attention mechanism. The proposed sparse attention can handle sequences of length up to
8x of what was previously possible using similar hardware. As a consequence of the capability to handle longer context,
BigBird drastically improves performance on various NLP tasks such as question answering and summarization. We also
propose novel applications to genomics data.*

Tips:

- For an in-detail explanation on how BigBird's attention works, see `this blog post <https://huggingface.co/blog/big-bird>`__.
- BigBird comes with 2 implementations: **original_full** & **block_sparse**. For the sequence length < 1024, using
  **original_full** is advised as there is no benefit in using **block_sparse** attention.
- The code currently uses window size of 3 blocks and 2 global blocks.
- Sequence length must be divisible by block size.
- Current implementation supports only **ITC**.
- Current implementation doesn't support **num_random_blocks = 0**.
- BigBirdPegasus uses the `PegasusTokenizer <https://github.com/huggingface/transformers/blob/main/src/transformers/models/pegasus/tokenization_pegasus.py>`__.
- BigBird is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.

The original code can be found `here <https://github.com/google-research/bigbird>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 96103):
        Vocabulary size of the BigBirdPegasus model. Defines the number of different tokens that can be represented
        by the `inputs_ids` passed when calling ``BigBirdPegasusModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimension of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 16):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 16):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimension of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimension of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_position_embeddings (`int`, *optional*, defaults to 4096):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 1024 or 2048 or 4096).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    attention_type (`str`, *optional*, defaults to `"block_sparse"`)
        Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
        layer (with n^2 complexity) in encoder. Possible values are `"original_full"` and `"block_sparse"`.
    use_bias (`bool`, *optional*, defaults to `False`)
        Whether to use bias in query, key, value.
    block_size (`int`, *optional*, defaults to 64)
        Size of each block. Useful only when `attention_type == "block_sparse"`.
    num_random_blocks (`int`, *optional*, defaults to 3)
        Each query is going to attend these many number of random blocks. Useful only when `attention_type ==
        "block_sparse"`.
    scale_embeddings (`bool`, *optional*, defaults to `True`)
        Whether to rescale embeddings with (hidden_size ** 0.5).

.. class:: transformers.models.biogpt.configuration_biogpt.BioGptConfig(vocab_size=42384, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=1024, initializer_range=0.02, layer_norm_eps=1e-12, scale_embedding=True, use_cache=True, layerdrop=0.0, activation_dropout=0.0, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The BioGPT model was proposed in `BioGPT: generative pre-trained transformer for biomedical text generation and mining
 <https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9>`__ by Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon and Tie-Yan Liu. BioGPT is a domain-specific generative pre-trained Transformer language model for biomedical text generation and mining. BioGPT follows the Transformer language model backbone, and is pre-trained on 15M PubMed abstracts from scratch.

The abstract from the paper is the following:

*Pre-trained language models have attracted increasing attention in the biomedical domain, inspired by their great success in the general natural language domain. Among the two main branches of pre-trained language models in the general language domain, i.e. BERT (and its variants) and GPT (and its variants), the first one has been extensively studied in the biomedical domain, such as BioBERT and PubMedBERT. While they have achieved great success on a variety of discriminative downstream biomedical tasks, the lack of generation ability constrains their application scope. In this paper, we propose BioGPT, a domain-specific generative Transformer language model pre-trained on large-scale biomedical literature. We evaluate BioGPT on six biomedical natural language processing tasks and demonstrate that our model outperforms previous models on most tasks. Especially, we get 44.98%, 38.42% and 40.76% F1 score on BC5CDR, KD-DTI and DDI end-to-end relation extraction tasks, respectively, and 78.2% accuracy on PubMedQA, creating a new record. Our case study on text generation further demonstrates the advantage of BioGPT on biomedical literature to generate fluent descriptions for biomedical terms.*

Tips:

- BioGPT is a model with absolute position embeddings so it’s usually advised to pad the inputs on the right rather than the left.
- BioGPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next token in a sequence. Leveraging this feature allows BioGPT to generate syntactically coherent text as it can be observed in the run_generation.py example script.
- The model can take the `past_key_values` (for PyTorch) as input, which is the previously computed key/value attention pairs. Using this (past_key_values or past) value prevents the model from re-computing pre-computed values in the context of text generation. For PyTorch, see past_key_values argument of the BioGptForCausalLM.forward() method for more information on its usage.

This model was contributed by `kamalkraj <https://huggingface.co/kamalkraj>`__. The original code can be found `here <https://github.com/microsoft/BioGPT>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 42384):
        Vocabulary size of the BioGPT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``BioGptModel``.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 4096):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    scale_embedding (`bool`, *optional*, defaults to `True`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    layerdrop (`float`, *optional*, defaults to 0.0):
        Please refer to the paper about LayerDrop: https://arxiv.org/abs/1909.11556 for further details
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    pad_token_id (`int`, *optional*, defaults to 1):
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 0):
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 2):
        End of stream token id.

.. class:: transformers.models.blenderbot.configuration_blenderbot.BlenderbotConfig(vocab_size=8008, max_position_embeddings=128, encoder_layers=2, encoder_ffn_dim=10240, encoder_attention_heads=32, decoder_layers=24, decoder_ffn_dim=10240, decoder_attention_heads=32, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=2560, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=1, scale_embedding=False, pad_token_id=0, bos_token_id=1, eos_token_id=2, encoder_no_repeat_ngram_size=3, forced_eos_token_id=2, **kwargs)

The Blender chatbot model was proposed in `Recipes for building an open-domain chatbot <https://arxiv.org/pdf/2004.13637.pdf>`__ Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.

The abstract of the paper is the following:

*Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.*

Tips:

- Blenderbot is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.

This model was contributed by `sshleifer <https://huggingface.co/sshleifer>`__. The authors' code can be found `here <https://github.com/facebookresearch/ParlAI>`__ .


Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the Blenderbot model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``BlenderbotModel`` or ``TFBlenderbotModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    max_position_embeddings (`int`, *optional*, defaults to 128):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)
    forced_eos_token_id (`int`, *optional*, defaults to 2):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.

.. class:: transformers.models.blenderbot_small.configuration_blenderbot_small.BlenderbotSmallConfig(vocab_size=50265, max_position_embeddings=512, encoder_layers=8, encoder_ffn_dim=2048, encoder_attention_heads=16, decoder_layers=8, decoder_ffn_dim=2048, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=512, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=1, scale_embedding=False, pad_token_id=0, bos_token_id=1, eos_token_id=2, forced_eos_token_id=2, **kwargs)

The Blender chatbot model was proposed in `Recipes for building an open-domain chatbot <https://arxiv.org/pdf/2004.13637.pdf>`__ Stephen Roller, Emily Dinan, Naman Goyal, Da Ju, Mary Williamson, Yinhan Liu,
Jing Xu, Myle Ott, Kurt Shuster, Eric M. Smith, Y-Lan Boureau, Jason Weston on 30 Apr 2020.

The abstract of the paper is the following:

*Building open-domain chatbots is a challenging area for machine learning research. While prior work has shown that
scaling neural models in the number of parameters and the size of the data they are trained on gives improved results,
we show that other ingredients are important for a high-performing chatbot. Good conversation requires a number of
skills that an expert conversationalist blends in a seamless way: providing engaging talking points and listening to
their partners, and displaying knowledge, empathy and personality appropriately, while maintaining a consistent
persona. We show that large scale models can learn these skills when given appropriate training data and choice of
generation strategy. We build variants of these recipes with 90M, 2.7B and 9.4B parameter models, and make our models
and code publicly available. Human evaluations show our best models are superior to existing approaches in multi-turn
dialogue in terms of engagingness and humanness measurements. We then discuss the limitations of this work by analyzing
failure cases of our models.*

Tips:

- Blenderbot Small is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.

This model was contributed by `patrickvonplaten <https://huggingface.co/patrickvonplaten>`__. The authors' code can be
found `here <https://github.com/facebookresearch/ParlAI>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the BlenderbotSmall model. Defines the number of different tokens that can be
        represented by the `inputs_ids` passed when calling ``BlenderbotSmallModel`` or ``TFBlenderbotSmallModel``.
    d_model (`int`, *optional*, defaults to 512):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 8):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 8):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)
    forced_eos_token_id (`int`, *optional*, defaults to 2):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.

.. class:: transformers.models.bloom.configuration_bloom.BloomConfig(vocab_size=250880, hidden_size=64, n_layer=2, n_head=8, layer_norm_epsilon=1e-05, initializer_range=0.02, use_cache=True, bos_token_id=1, eos_token_id=2, apply_residual_connection_post_layernorm=False, hidden_dropout=0.0, attention_dropout=0.0, pretraining_tp=1, slow_but_exact=False, **kwargs)

The BLOOM model has been proposed with its various versions through the `BigScience Workshop <https://bigscience.huggingface.co/>`__. BigScience is inspired by other open science initiatives where researchers have pooled their time and resources to collectively achieve a higher impact.
The architecture of BLOOM is essentially similar to GPT3 (auto-regressive model for next token prediction), but has been trained on 46 different languages and 13 programming languages.
Several smaller versions of the models have been trained on the same dataset. BLOOM is available in the following versions:

- `bloom-560m <https://huggingface.co/bigscience/bloom-560m>`__
- `bloom-1b1 <https://huggingface.co/bigscience/bloom-1b1>`__
- `bloom-1b7 <https://huggingface.co/bigscience/bloom-1b7>`__
- `bloom-3b <https://huggingface.co/bigscience/bloom-3b>`__
- `bloom-7b1 <https://huggingface.co/bigscience/bloom-7b1>`__
- `bloom <https://huggingface.co/bigscience/bloom>`__ (176B parameters)

Args:
    vocab_size (`int`, *optional*, defaults to 250880):
        Vocabulary size of the Bloom model. Defines the maximum number of different tokens that can be represented
        by the `inputs_ids` passed when calling ``BloomModel``. Check `this
        discussion <https://huggingface.co/bigscience/bloom/discussions/120#633d28389addb8530b406c2a>`__ on how the
        `vocab_size` has been defined.
    hidden_size (`int`, *optional*, defaults to 64):
        Dimensionality of the embeddings and hidden states.
    n_layer (`int`, *optional*, defaults to 2):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the Transformer encoder.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
        The epsilon to use in the layer normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
        If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
    hidden_dropout (`float`, *optional*, defaults to 0.1):
        Dropout rate of the dropout function on the bias dropout.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        Dropout rate applied to the attention probs
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    pretraining_tp (`int`, *optional*, defaults to `1`):
        Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to `this
        document <https://huggingface.co/docs/transformers/parallelism>`__ to understand more about it. This value is
        necessary to ensure exact reproducibility of the pretraining results. Please refer to `this
        issue <https://github.com/pytorch/pytorch/issues/76232>`__. Note also that this is enabled only when
        `slow_but_exact=True`.
    slow_but_exact (`bool`, *optional*, defaults to `False`):
        Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
        merging the TP rank tensors, due to slicing operations the results may be slightly different between the
        model trained on Megatron and our model. Please refer to `this
        issue <https://github.com/pytorch/pytorch/issues/76232>`__. A solution to obtain more accurate results is to
        enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
        resolved in the future once the main model has been fine-tuned with TP_rank=1.

.. class:: transformers.models.camembert.configuration_camembert.CamembertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The CamemBERT model was proposed in `CamemBERT: a Tasty French Language Model <https://arxiv.org/abs/1911.03894>`__ by
Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la
Clergerie, Djamé Seddah, and Benoît Sagot. It is based on Facebook's RoBERTa model released in 2019. It is a model
trained on 138GB of French text.

The abstract from the paper is the following:

*Pretrained language models are now ubiquitous in Natural Language Processing. Despite their success, most available
models have either been trained on English data or on the concatenation of data in multiple languages. This makes
practical use of such models --in all languages except English-- very limited. Aiming to address this issue for French,
we release CamemBERT, a French version of the Bi-directional Encoders for Transformers (BERT). We measure the
performance of CamemBERT compared to multilingual models in multiple downstream tasks, namely part-of-speech tagging,
dependency parsing, named-entity recognition, and natural language inference. CamemBERT improves the state of the art
for most of the tasks considered. We release the pretrained model for CamemBERT hoping to foster research and
downstream applications for French NLP.*

Tips:

- This implementation is the same as RoBERTa. Refer to the `documentation of RoBERTa <roberta>`__ for usage examples
  as well as the information relative to the inputs and outputs.

This model was contributed by `camembert <https://huggingface.co/camembert>`__. The original code can be found `here <https://camembert-model.fr/>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``CamembertModel`` or ``TFCamembertModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``CamembertModel`` or ``TFCamembertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.llama.configuration_llama.LlamaConfig(vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None, hidden_act='silu', max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=1e-06, use_cache=True, pad_token_id=None, bos_token_id=1, eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, rope_theta=10000.0, rope_scaling=None, attention_bias=False, attention_dropout=0.0, **kwargs)

The Code Llama model was proposed in `Code Llama: Open Foundation Models for Code <https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/>`__ by Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve.

The abstract from the paper is the following:

*We release Code Llama, a family of large language models for code based on Llama 2 providing state-of-the-art performance among open models, infilling capabilities, support for large input contexts, and zero-shot instruction following ability for programming tasks. We provide multiple flavors to cover a wide range of applications: foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models (Code Llama - Instruct) with 7B, 13B and 34B parameters each. All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens. 7B and 13B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama reaches state-of-the-art performance among open models on several code benchmarks, with scores of up to 53% and 55% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every other publicly available model on MultiPL-E. We release Code Llama under a permissive license that allows for both research and commercial use.*

Check out all Code Llama models `here <https://huggingface.co/models?search=code_llama>`__ and the officially released ones in the `codellama org <https://huggingface.co/codellama>`__.

<Tip warning={true}>

The `Llama2` family models, on which Code Llama is based, were trained using `bfloat16`, but the original inference uses `float16`. Let's look at the different precisions:

* `float32`: PyTorch convention on model initialization is to load models in `float32`, no matter with which `dtype` the model weights were stored. `transformers` also follows this convention for consistency with PyTorch. This will be picked by default. If you want the `AutoModel` API to cast the load the checkpoints with the storage weights type, you must specify `torch_dtype="auto"`, e.g. `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`.
* `bfloat16`: Code Llama was trained with this precision, so we recommend using it for further training or fine-tuning.
* `float16`: We recommend running inference using this precision, as it's usually faster than `bfloat16`, and evaluation metrics show no discernible degradation with respect to `bfloat16`. You can also run inference using `bfloat16`, and we recommend you check inference results with both `float16` and `bfloat16` after fine-tuning.

As mentioned above, the `dtype` of the storage weights is mostly irrelevant unless you are using `torch_dtype="auto"` when initializing a model using. The reason is that the model will first be downloaded (using the `dtype` of the checkpoints online) and then will be casted to the default `dtype` of `torch` (becomes `torch.float32`). If there is a specified `torch_dtype`, it will be used instead.

</Tip>

Tips:

- These models have the same architecture as the `Llama2` models
- The infilling task is supported out of the box. You should be using the `tokenizer.fill_token` where you want your input to be filled.
- The model conversion script is the same as for the `Llama2` family:

Here is a sample usage
```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```
Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).

- After conversion, the model and tokenizer can be loaded via:

::

>>> from transformers import LlamaForCausalLM, CodeLlamaTokenizer

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = tokenizer(PROMPT, return_tensors="pt")`"input_ids"]
>>> generated_ids = model.generate(input_ids, max_new_tokens=128)

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result

If you only want the infilled part:
::

>>> from transformers import pipeline
>>> import torch

>>> generator = pipeline("text-generation",model="codellama/CodeLlama-7b-hf",torch_dtype=torch.float16, device_map="auto")
>>> generator('def remove_non_ascii(s: str) -> str:\n    """ <FILL_ME>\n    return result', max_new_tokens = 128, return_type = 1)

Under the hood, the tokenizer [automatically splits by `<FILL_ME>` <https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.fill_token>`__ to create a formatted input string that follows `the original training pattern <https://github.com/facebookresearch/codellama/blob/cb51c14ec761370ba2e2bc351374a79265d0465e/llama/generation.py#L402>`__. This is more robust than preparing the pattern yourself: it avoids pitfalls, such as token glueing, that are very hard to debug.  To see how much CPU and GPU memory you need for this model or others, try `this calculator <https://huggingface.co/spaces/hf-accelerate/model-memory-usage>`__ which can help determine that value.

- The LLaMA tokenizer is a BPE model based on `sentencepiece <https://github.com/google/sentencepiece>`__. One quirk of sentencepiece is that when decoding a sequence, if the first token is the start of the word (e.g. "Banana"), the tokenizer does not prepend the prefix space to the string.

This model was contributed by `ArthurZucker <https://huggingface.co/ArthurZ>`__. The original code of the authors can be found `here <https://github.com/facebookresearch/llama>`__.


Args:
    vocab_size (`int`, *optional*, defaults to 32000):
        Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``LlamaModel``
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 11008):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer decoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer decoder.
    num_key_value_heads (`int`, *optional*):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to
        `num_attention_heads`.
    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
        Llama 2 up to 4096, CodeLlama up to 16384.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    rms_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon used by the rms normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    pad_token_id (`int`, *optional*):
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 1):
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 2):
        End of stream token id.
    pretraining_tp (`int`, *optional*, defaults to 1):
        Experimental feature. Tensor parallelism rank used during pretraining. Please refer to `this
        document <https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism>`__ to understand more about it. This value is
        necessary to ensure exact reproducibility of the pretraining results. Please refer to `this
        issue <https://github.com/pytorch/pytorch/issues/76232>`__.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to tie weight embeddings
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
        strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
        `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
        experimental feature, subject to breaking API changes in future versions.
    attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
        Whether to use a bias in the query, key, value and output projection layers during self-attention.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.

::

>>> from transformers import LlamaModel, LlamaConfig

>>> # Initializing a LLaMA llama-7b style configuration
>>> configuration = LlamaConfig()

>>> # Initializing a model from the llama-7b style configuration
>>> model = LlamaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

.. class:: transformers.models.codegen.configuration_codegen.CodeGenConfig(vocab_size=50400, n_positions=2048, n_ctx=2048, n_embd=4096, n_layer=28, n_head=16, rotary_dim=64, n_inner=None, activation_function='gelu_new', resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-05, initializer_range=0.02, use_cache=True, bos_token_id=50256, eos_token_id=50256, tie_word_embeddings=False, **kwargs)

The CodeGen model was proposed in `A Conversational Paradigm for Program Synthesis <https://arxiv.org/abs/2203.13474>`__ by Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong.

CodeGen is an autoregressive language model for program synthesis trained sequentially on `The Pile <https://pile.eleuther.ai/>`__, BigQuery, and BigPython.

The abstract from the paper is the following:

*Program synthesis strives to generate a computer program as a solution to a given problem specification. We propose a conversational program synthesis approach via large language models, which addresses the challenges of searching over a vast program space and user intent specification faced in prior approaches. Our new approach casts the process of writing a specification and program as a multi-turn conversation between a user and a system. It treats program synthesis as a sequence prediction problem, in which the specification is expressed in natural language and the desired program is conditionally sampled. We train a family of large language models, called CodeGen, on natural language and programming language data. With weak supervision in the data and the scaling up of data size and model size, conversational capacities emerge from the simple autoregressive language modeling. To study the model behavior on conversational program synthesis, we develop a multi-turn programming benchmark (MTPB), where solving each problem requires multi-step synthesis via multi-turn conversation between the user and the model. Our findings show the emergence of conversational capabilities and the effectiveness of the proposed conversational program synthesis paradigm. In addition, our model CodeGen (with up to 16B parameters trained on TPU-v4) outperforms OpenAI's Codex on the HumanEval benchmark. We make the training library JaxFormer including checkpoints available as open source contribution: `this https URL <https://github.com/salesforce/codegen>`__.* 

This model was contributed by `Hiroaki Hayashi <https://huggingface.co/rooa>`__.
The original code can be found `here <https://github.com/salesforce/codegen>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50400):
        Vocabulary size of the CodeGen model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``CodeGenModel``.
    n_positions (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_ctx (`int`, *optional*, defaults to 2048):
        This attribute is used in `CodeGenModel.__init__` without any real effect.
    n_embd (`int`, *optional*, defaults to 4096):
        Dimensionality of the embeddings and hidden states.
    n_layer (`int`, *optional*, defaults to 28):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    rotary_dim (`int`, *optional*, defaults to 64):
        Number of dimensions in the embedding that Rotary Position Embedding is applied to.
    n_inner (`int`, *optional*):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
    activation_function (`str`, *optional*, defaults to `"gelu_new"`):
        Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    resid_pdrop (`float`, *optional*, defaults to 0.0):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (`int`, *optional*, defaults to 0.0):
        The dropout ratio for the embeddings.
    attn_pdrop (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    bos_token_id (`int`, *optional*, defaults to 50256):
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 50256):
        End of stream token id.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
        model has a output word embedding layer.

.. class:: transformers.models.cohere.configuration_cohere.CohereConfig(vocab_size=256000, hidden_size=8192, intermediate_size=22528, logit_scale=0.0625, num_hidden_layers=40, num_attention_heads=64, num_key_value_heads=None, hidden_act='silu', max_position_embeddings=8192, initializer_range=0.02, layer_norm_eps=1e-05, use_cache=True, pad_token_id=0, bos_token_id=5, eos_token_id=255001, tie_word_embeddings=True, rope_theta=10000.0, attention_bias=False, attention_dropout=0.0, **kwargs)

The Cohere Command-R model was proposed in the blogpost `Command-R: Retrieval Augmented Generation at Production Scale <https://txt.cohere.com/command-r/>`__ by the Cohere Team.

The abstract from the paper is the following:

*Command-R is a scalable generative model targeting RAG and Tool Use to enable production-scale AI for enterprise. Today, we are introducing Command-R, a new LLM aimed at large-scale production workloads. Command-R targets the emerging “scalable” category of models that balance high efficiency with strong accuracy, enabling companies to move beyond proof of concept, and into production.*

*Command-R is a generative model optimized for long context tasks such as retrieval augmented generation (RAG) and using external APIs and tools. It is designed to work in concert with our industry-leading Embed and Rerank models to provide best-in-class integration for RAG applications and excel at enterprise use cases. As a model built for companies to implement at scale, Command-R boasts:
- Strong accuracy on RAG and Tool Use
- Low latency, and high throughput
- Longer 128k context and lower pricing
- Strong capabilities across 10 key languages
- Model weights available on HuggingFace for research and evaluation

Checkout model checkpoints `here <https://huggingface.co/CohereForAI/c4ai-command-r-v01>`__.
This model was contributed by `Saurabh Dash <https://huggingface.co/saurabhdash>`__ and `Ahmet Üstün <https://huggingface.co/ahmetustun>`__. The code of the implementation in Hugging Face is based on GPT-NeoX `here <https://github.com/EleutherAI/gpt-neox>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 256000):
        Vocabulary size of the Cohere model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``CohereModel``
    hidden_size (`int`, *optional*, defaults to 8192):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 22528):
        Dimension of the MLP representations.
    logit_scale (`float`, *optional*, defaults to 0.0625):
        The scaling factor for the output logits.
    num_hidden_layers (`int`, *optional*, defaults to 40):
        Number of hidden layers in the Transformer decoder.
    num_attention_heads (`int`, *optional*, defaults to 64):
        Number of attention heads for each attention layer in the Transformer decoder.
    num_key_value_heads (`int`, *optional*):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to
        `num_attention_heads`.
    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to 8192):
        The maximum sequence length that this model might ever be used with.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the layer normalization.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    pad_token_id (`int`, *optional*, defaults to 0):
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 5):
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 255001):
        End of stream token id.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie weight embeddings
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
        Whether to use a bias in the query, key, value and output projection layers during self-attention.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.

::

>>> from transformers import CohereModel, CohereConfig

>>> # Initializing a Cohere model configuration
>>> configuration = CohereConfig()

>>> # Initializing a model from the Cohere configuration
>>> model = CohereModel(configuration) # doctest: +SKIP

>>> # Accessing the model configuration
>>> configuration = model.config # doctest: +SKIP

.. class:: transformers.models.ctrl.configuration_ctrl.CTRLConfig(vocab_size=246534, n_positions=256, n_embd=1280, dff=8192, n_layer=48, n_head=16, resid_pdrop=0.1, embd_pdrop=0.1, layer_norm_epsilon=1e-06, initializer_range=0.02, use_cache=True, **kwargs)

CTRL model was proposed in `CTRL: A Conditional Transformer Language Model for Controllable Generation <https://arxiv.org/abs/1909.05858>`__ by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and
Richard Socher. It's a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus
of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

The abstract from the paper is the following:

*Large-scale language models show promising text generation capabilities, but users cannot easily control particular
aspects of the generated text. We release CTRL, a 1.63 billion-parameter conditional transformer language model,
trained to condition on control codes that govern style, content, and task-specific behavior. Control codes were
derived from structure that naturally co-occurs with raw text, preserving the advantages of unsupervised learning while
providing more explicit control over text generation. These codes also allow CTRL to predict which parts of the
training data are most likely given a sequence. This provides a potential method for analyzing large amounts of data
via model-based source attribution.*

Tips:

- CTRL makes use of control codes to generate text: it requires generations to be started by certain words, sentences
  or links to generate coherent text. Refer to the `original implementation <https://github.com/salesforce/ctrl>`__ for
  more information.
- CTRL is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text as it can be
  observed in the *run_generation.py* example script.
- The PyTorch models can take the `past_key_values` as input, which is the previously computed key/value attention pairs.
  TensorFlow models accepts `past` as input. Using the `past_key_values` value prevents the model from re-computing
  pre-computed values in the context of text generation. See the ``forward``(model_doc/ctrl#transformers.CTRLModel.forward)
  method for more information on the usage of this argument.

This model was contributed by `keskarnitishr <https://huggingface.co/keskarnitishr>`__. The original code can be found
`here <https://github.com/salesforce/ctrl>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 246534):
        Vocabulary size of the CTRL model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``CTRLModel`` or ``TFCTRLModel``.
    n_positions (`int`, *optional*, defaults to 256):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_embd (`int`, *optional*, defaults to 1280):
        Dimensionality of the embeddings and hidden states.
    dff (`int`, *optional*, defaults to 8192):
        Dimensionality of the inner dimension of the feed forward networks (FFN).
    n_layer (`int`, *optional*, defaults to 48):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    resid_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (`int`, *optional*, defaults to 0.1):
        The dropout ratio for the embeddings.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-06):
        The epsilon to use in the layer normalization layers
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.data2vec.configuration_data2vec_text.Data2VecTextConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

This is the configuration class to store the configuration of a ``Data2VecTextModel`` and ``Data2VecTextModel``. It
is used to instantiate a Data2VecText model according to the specified arguments, defining the model architecture.
Instantiating a configuration with the defaults will yield a similar configuration to that of the Data2VecText
`facebook/data2vec-text-base <https://huggingface.co/facebook/data2vec-text-base>`__ architecture.

Configuration objects inherit from ``PretrainedConfig`` and can be used to control the model outputs. Read the
documentation from ``PretrainedConfig`` for more information.


Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the DATA2VEC model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``Data2VecModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``Data2VecModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.deberta.configuration_deberta.DebertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=0, initializer_range=0.02, layer_norm_eps=1e-07, relative_attention=False, max_relative_positions=-1, pad_token_id=0, position_biased_input=True, pos_att_type=None, pooler_dropout=0, pooler_hidden_act='gelu', **kwargs)

The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention <https://arxiv.org/abs/2006.03654>`__ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen It is based on Google's
BERT model released in 2018 and Facebook's RoBERTa model released in 2019.

It builds on RoBERTa with disentangled attention and enhanced mask decoder training with half of the data used in
RoBERTa.

The abstract from the paper is the following:

*Recent progress in pre-trained neural language models has significantly improved the performance of many natural
language processing (NLP) tasks. In this paper we propose a new model architecture DeBERTa (Decoding-enhanced BERT with
disentangled attention) that improves the BERT and RoBERTa models using two novel techniques. The first is the
disentangled attention mechanism, where each word is represented using two vectors that encode its content and
position, respectively, and the attention weights among words are computed using disentangled matrices on their
contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to
predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency
of model pretraining and performance of downstream tasks. Compared to RoBERTa-Large, a DeBERTa model trained on half of
the training data performs consistently better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9%
(90.2% vs. 91.1%), on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%). The DeBERTa code and
pre-trained models will be made publicly available at https://github.com/microsoft/DeBERTa.*


This model was contributed by `DeBERTa <https://huggingface.co/DeBERTa>`__. This model TF 2.0 implementation was
contributed by `kamalkraj <https://huggingface.co/kamalkraj>`__ . The original code can be found `here <https://github.com/microsoft/DeBERTa>`__.

Arguments:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the DeBERTa model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``DebertaModel`` or ``TFDebertaModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
        are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``DebertaModel`` or ``TFDebertaModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    relative_attention (`bool`, *optional*, defaults to `False`):
        Whether use relative position encoding.
    max_relative_positions (`int`, *optional*, defaults to 1):
        The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
        as `max_position_embeddings`.
    pad_token_id (`int`, *optional*, defaults to 0):
        The value used to pad input_ids.
    position_biased_input (`bool`, *optional*, defaults to `True`):
        Whether add absolute position embedding to content embedding.
    pos_att_type (`List[str]`, *optional*):
        The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
        `["p2c", "c2p"]`.
    layer_norm_eps (`float`, optional, defaults to 1e-12):
        The epsilon used by the layer normalization layers.

.. class:: transformers.models.deberta_v2.configuration_deberta_v2.DebertaV2Config(vocab_size=128100, hidden_size=1536, num_hidden_layers=24, num_attention_heads=24, intermediate_size=6144, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=0, initializer_range=0.02, layer_norm_eps=1e-07, relative_attention=False, max_relative_positions=-1, pad_token_id=0, position_biased_input=True, pos_att_type=None, pooler_dropout=0, pooler_hidden_act='gelu', **kwargs)

The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention <https://arxiv.org/abs/2006.03654>`__ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen It is based on Google's
BERT model released in 2018 and Facebook's RoBERTa model released in 2019.

It builds on RoBERTa with disentangled attention and enhanced mask decoder training with half of the data used in
RoBERTa.

The abstract from the paper is the following:

*Recent progress in pre-trained neural language models has significantly improved the performance of many natural
language processing (NLP) tasks. In this paper we propose a new model architecture DeBERTa (Decoding-enhanced BERT with
disentangled attention) that improves the BERT and RoBERTa models using two novel techniques. The first is the
disentangled attention mechanism, where each word is represented using two vectors that encode its content and
position, respectively, and the attention weights among words are computed using disentangled matrices on their
contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to
predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency
of model pretraining and performance of downstream tasks. Compared to RoBERTa-Large, a DeBERTa model trained on half of
the training data performs consistently better on a wide range of NLP tasks, achieving improvements on MNLI by +0.9%
(90.2% vs. 91.1%), on SQuAD v2.0 by +2.3% (88.4% vs. 90.7%) and RACE by +3.6% (83.2% vs. 86.8%). The DeBERTa code and
pre-trained models will be made publicly available at https://github.com/microsoft/DeBERTa.*


The following information is visible directly on the `original implementation
repository <https://github.com/microsoft/DeBERTa>`__. DeBERTa v2 is the second version of the DeBERTa model. It includes
the 1.5B model used for the SuperGLUE single-model submission and achieving 89.9, versus human baseline 89.8. You can
find more details about this submission in the authors'
`blog <https://www.microsoft.com/en-us/research/blog/microsoft-deberta-surpasses-human-performance-on-the-superglue-benchmark/>`__

New in v2:

- **Vocabulary** In v2 the tokenizer is changed to use a new vocabulary of size 128K built from the training data.
  Instead of a GPT2-based tokenizer, the tokenizer is now
  `sentencepiece-based <https://github.com/google/sentencepiece>`__ tokenizer.
- **nGiE(nGram Induced Input Encoding)** The DeBERTa-v2 model uses an additional convolution layer aside with the first
  transformer layer to better learn the local dependency of input tokens.
- **Sharing position projection matrix with content projection matrix in attention layer** Based on previous
  experiments, this can save parameters without affecting the performance.
- **Apply bucket to encode relative positions** The DeBERTa-v2 model uses log bucket to encode relative positions
  similar to T5.
- **900M model & 1.5B model** Two additional model sizes are available: 900M and 1.5B, which significantly improves the
  performance of downstream tasks.

This model was contributed by `DeBERTa <https://huggingface.co/DeBERTa>`__. This model TF 2.0 implementation was
contributed by `kamalkraj <https://huggingface.co/kamalkraj>`__. The original code can be found `here <https://github.com/microsoft/DeBERTa>`__.

Arguments:
    vocab_size (`int`, *optional*, defaults to 128100):
        Vocabulary size of the DeBERTa-v2 model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``DebertaV2Model``.
    hidden_size (`int`, *optional*, defaults to 1536):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 24):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 6144):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"`, `"gelu"`, `"tanh"`, `"gelu_fast"`, `"mish"`, `"linear"`, `"sigmoid"` and `"gelu_new"`
        are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 0):
        The vocabulary size of the `token_type_ids` passed when calling ``DebertaModel`` or ``TFDebertaModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-7):
        The epsilon used by the layer normalization layers.
    relative_attention (`bool`, *optional*, defaults to `True`):
        Whether use relative position encoding.
    max_relative_positions (`int`, *optional*, defaults to -1):
        The range of relative positions `[-max_position_embeddings, max_position_embeddings]`. Use the same value
        as `max_position_embeddings`.
    pad_token_id (`int`, *optional*, defaults to 0):
        The value used to pad input_ids.
    position_biased_input (`bool`, *optional*, defaults to `False`):
        Whether add absolute position embedding to content embedding.
    pos_att_type (`List[str]`, *optional*):
        The type of relative position attention, it can be a combination of `["p2c", "c2p"]`, e.g. `["p2c"]`,
        `["p2c", "c2p"]`, `["p2c", "c2p"]`.
    layer_norm_eps (`float`, optional, defaults to 1e-12):
        The epsilon used by the layer normalization layers.

.. class:: transformers.models.distilbert.configuration_distilbert.DistilBertConfig(vocab_size=30522, max_position_embeddings=512, sinusoidal_pos_embds=False, n_layers=6, n_heads=12, dim=768, hidden_dim=3072, dropout=0.1, attention_dropout=0.1, activation='gelu', initializer_range=0.02, qa_dropout=0.1, seq_classif_dropout=0.2, pad_token_id=0, **kwargs)

The DistilBERT model was proposed in the blog post `Smaller, faster, cheaper, lighter: Introducing DistilBERT, a
distilled version of BERT <https://medium.com/huggingface/distilbert-8cf3380435b5>`__, and the paper `DistilBERT, a
distilled version of BERT: smaller, faster, cheaper and lighter <https://arxiv.org/papers/1910.01108>`__. DistilBERT is a
small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than
*bert-base-uncased*, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language
understanding benchmark.

The abstract from the paper is the following:

*As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP),
operating these large models in on-the-edge and/or under constrained computational training or inference budgets
remains challenging. In this work, we propose a method to pre-train a smaller general-purpose language representation
model, called DistilBERT, which can then be fine-tuned with good performances on a wide range of tasks like its larger
counterparts. While most prior work investigated the use of distillation for building task-specific models, we leverage
knowledge distillation during the pretraining phase and show that it is possible to reduce the size of a BERT model by
40%, while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive
biases learned by larger models during pretraining, we introduce a triple loss combining language modeling,
distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we
demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device
study.*

Tips:

- DistilBERT doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just
  separate your segments with the separation token `tokenizer.sep_token` (or ``SEP]`).
- DistilBERT doesn't have options to select the input positions (`position_ids` input). This could be added if
  necessary though, just let us know if you need this option.
- Same as BERT but smaller. Trained by distillation of the pretrained BERT model, meaning it’s been trained to predict the same probabilities as the larger model. The actual objective is a combination of:

    * finding the same probabilities as the teacher model
    * predicting the masked tokens correctly (but no next-sentence objective)
    * a cosine similarity between the hidden states of the student and the teacher model

This model was contributed by [victorsanh <https://huggingface.co/victorsanh>`__. This model jax version was
contributed by `kamalkraj <https://huggingface.co/kamalkraj>`__. The original code can be found `here <https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the DistilBERT model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``DistilBertModel`` or ``TFDistilBertModel``.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    sinusoidal_pos_embds (`boolean`, *optional*, defaults to `False`):
        Whether to use sinusoidal positional embeddings.
    n_layers (`int`, *optional*, defaults to 6):
        Number of hidden layers in the Transformer encoder.
    n_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    dim (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    hidden_dim (`int`, *optional*, defaults to 3072):
        The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    qa_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probabilities used in the question answering model ``DistilBertForQuestionAnswering``.
    seq_classif_dropout (`float`, *optional*, defaults to 0.2):
        The dropout probabilities used in the sequence classification and the multiple choice model
        ``DistilBertForSequenceClassification``.

.. class:: transformers.models.electra.configuration_electra.ElectraConfig(vocab_size=30522, embedding_size=128, hidden_size=256, num_hidden_layers=12, num_attention_heads=4, intermediate_size=1024, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, summary_type='first', summary_use_proj=True, summary_activation='gelu', summary_last_dropout=0.1, pad_token_id=0, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The ELECTRA model was proposed in the paper `ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
Generators <https://openreview.net/pdf?id=r1xMH1BtvB>`__. ELECTRA is a new pretraining approach which trains two
transformer models: the generator and the discriminator. The generator's role is to replace tokens in a sequence, and
is therefore trained as a masked language model. The discriminator, which is the model we're interested in, tries to
identify which tokens were replaced by the generator in the sequence.

The abstract from the paper is the following:

*Masked language modeling (MLM) pretraining methods such as BERT corrupt the input by replacing some tokens with `MASK]
and then train a model to reconstruct the original tokens. While they produce good results when transferred to
downstream NLP tasks, they generally require large amounts of compute to be effective. As an alternative, we propose a
more sample-efficient pretraining task called replaced token detection. Instead of masking the input, our approach
corrupts it by replacing some tokens with plausible alternatives sampled from a small generator network. Then, instead
of training a model that predicts the original identities of the corrupted tokens, we train a discriminative model that
predicts whether each token in the corrupted input was replaced by a generator sample or not. Thorough experiments
demonstrate this new pretraining task is more efficient than MLM because the task is defined over all input tokens
rather than just the small subset that was masked out. As a result, the contextual representations learned by our
approach substantially outperform the ones learned by BERT given the same model size, data, and compute. The gains are
particularly strong for small models; for example, we train a model on one GPU for 4 days that outperforms GPT (trained
using 30x more compute) on the GLUE natural language understanding benchmark. Our approach also works well at scale,
where it performs comparably to RoBERTa and XLNet while using less than 1/4 of their compute and outperforms them when
using the same amount of compute.*

Tips:

- ELECTRA is the pretraining approach, therefore there is nearly no changes done to the underlying model: BERT. The
  only change is the separation of the embedding size and the hidden size: the embedding size is generally smaller,
  while the hidden size is larger. An additional projection layer (linear) is used to project the embeddings from their
  embedding size to the hidden size. In the case where the embedding size is the same as the hidden size, no projection
  layer is used.
- ELECTRA is a transformer model pretrained with the use of another (small) masked language model. The inputs are corrupted by that language model, which takes an input text that is randomly masked and outputs a text in which ELECTRA has to predict which token is an original and which one has been replaced. Like for GAN training, the small language model is trained for a few steps (but with the original texts as objective, not to fool the ELECTRA model like in a traditional GAN setting) then the ELECTRA model is trained for a few steps.
- The ELECTRA checkpoints saved using [Google Research's implementation <https://github.com/google-research/electra>`__
  contain both the generator and discriminator. The conversion script requires the user to name which model to export
  into the correct architecture. Once converted to the HuggingFace format, these checkpoints may be loaded into all
  available ELECTRA models, however. This means that the discriminator may be loaded in the
  ``ElectraForMaskedLM`` model, and the generator may be loaded in the
  ``ElectraForPreTraining`` model (the classification head will be randomly initialized as it
  doesn't exist in the generator).

This model was contributed by `lysandre <https://huggingface.co/lysandre>`__. The original code can be found `here <https://github.com/google-research/electra>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the ELECTRA model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``ElectraModel`` or ``TFElectraModel``.
    embedding_size (`int`, *optional*, defaults to 128):
        Dimensionality of the encoder layers and the pooler layer.
    hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``ElectraModel`` or ``TFElectraModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    summary_type (`str`, *optional*, defaults to `"first"`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Has to be one of the following options:

            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Pass `"gelu"` for a gelu activation to the output, any other value will result in no activation.
    summary_last_dropout (`float`, *optional*, defaults to 0.0):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        The dropout ratio to be used after the projection and activation.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.ernie.configuration_ernie.ErnieConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, task_type_vocab_size=3, use_task_id=False, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

ERNIE is a series of powerful models proposed by baidu, especially in Chinese tasks,
including `ERNIE1.0 <https://arxiv.org/abs/1904.09223>`__, `ERNIE2.0 <https://ojs.aaai.org/index.php/AAAI/article/view/6428>`__,
`ERNIE3.0 <https://arxiv.org/abs/2107.02137>`__, `ERNIE-Gram <https://arxiv.org/abs/2010.12148>`__, `ERNIE-health <https://arxiv.org/abs/2110.07244>`__, etc.

These models are contributed by `nghuyong <https://huggingface.co/nghuyong>`__ and the official code can be found in `PaddleNLP <https://github.com/PaddlePaddle/PaddleNLP>`__ (in PaddlePaddle).

#Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the ERNIE model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``ErnieModel`` or ``TFErnieModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``ErnieModel`` or ``TFErnieModel``.
    task_type_vocab_size (`int`, *optional*, defaults to 3):
        The vocabulary size of the `task_type_ids` for ERNIE2.0/ERNIE3.0 model
    use_task_id (`bool`, *optional*, defaults to `False`):
        Whether or not the model support `task_type_ids`
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    pad_token_id (`int`, *optional*, defaults to 0):
        Padding token id.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.falcon.configuration_falcon.FalconConfig(vocab_size=65024, hidden_size=4544, num_hidden_layers=32, num_attention_heads=71, layer_norm_epsilon=1e-05, initializer_range=0.02, use_cache=True, hidden_dropout=0.0, attention_dropout=0.0, num_kv_heads=None, alibi=False, new_decoder_architecture=False, multi_query=True, parallel_attn=True, bias=False, max_position_embeddings=2048, rope_theta=10000.0, rope_scaling=None, bos_token_id=11, eos_token_id=11, **kwargs)

Falcon is a class of causal decoder-only models built by `TII <https://www.tii.ae/>`__. The largest Falcon checkpoints
have been trained on >=1T tokens of text, with a particular emphasis on the `RefinedWeb <https://arxiv.org/abs/2306.01116>`__
corpus. They are made available under the Apache 2.0 license.


Falcon's architecture is modern and optimized for inference, with multi-query attention and support for efficient
attention variants like `FlashAttention`. Both 'base' models trained only as causal language models as well as
'instruct' models that have received further fine-tuning are available.


Falcon models are (as of 2023) some of the largest and most powerful open-source language models,
and consistently rank highly in the `OpenLLM leaderboard <https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 65024):
        Vocabulary size of the Falcon model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``FalconModel``
    hidden_size (`int`, *optional*, defaults to 4544):
        Dimension of the hidden representations.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer decoder.
    num_attention_heads (`int`, *optional*, defaults to 71):
        Number of attention heads for each attention layer in the Transformer encoder.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the layer normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether the model should return the last key/values attentions (not used by all models). Only relevant if
        `config.is_decoder=True`.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for MLP layers.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability for attention layers.
    num_kv_heads (`int`, *optional*):
        Number of key-value heads to use per attention layer. If unset, defaults to the same value as
        `num_attention_heads`.
    alibi (`bool`, *optional*, defaults to `False`):
        Whether to use ALiBi positional biases during self-attention.
    new_decoder_architecture (`bool`, *optional*, defaults to `False`):
        Whether to use the new (Falcon-40B) decoder architecture. If `True`, the `multi_query` and `parallel_attn`
        arguments are ignored, as the new decoder always uses parallel attention.
    multi_query (`bool`, *optional*, defaults to `True`):
        Whether to use multi-query attention in the decoder. Ignored when `new_decoder_architecture` is `True`.
    parallel_attn (`bool`, *optional*, defaults to `True`):
        Whether to compute attention in parallel with the feedforward layer. If False, they are consecutive
        instead, as in the original Transformer architecture. Ignored when `new_decoder_architecture` is `True`.
    bias (`bool`, *optional*, defaults to `False`):
        Whether to use bias on Linear layers.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with, when `alibi` is `False`. Pretrained
        Falcon models with RoPE support up to 2048 tokens.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
        strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
        `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
        experimental feature, subject to breaking API changes in future versions.
    bos_token_id (`int`, *optional*, defaults to 11):
        The id of the "beginning-of-sequence" token.
    eos_token_id (`int`, *optional*, defaults to 11):
        The id of the "end-of-sequence" token.

.. class:: transformers.models.flaubert.configuration_flaubert.FlaubertConfig(pre_norm=False, layerdrop=0.0, vocab_size=30145, emb_dim=2048, n_layers=12, n_heads=16, dropout=0.1, attention_dropout=0.1, gelu_activation=True, sinusoidal_embeddings=False, causal=False, asm=False, n_langs=1, use_lang_emb=True, max_position_embeddings=512, embed_init_std=0.02209708691207961, layer_norm_eps=1e-12, init_std=0.02, bos_index=0, eos_index=1, pad_index=2, unk_index=3, mask_index=5, is_encoder=True, summary_type='first', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, start_n_top=5, end_n_top=5, mask_token_id=0, lang_id=0, pad_token_id=2, bos_token_id=0, **kwargs)

The FlauBERT model was proposed in the paper `FlauBERT: Unsupervised Language Model Pre-training for French <https://arxiv.org/abs/1912.05372>`__ by Hang Le et al. It's a transformer model pretrained using a masked language
modeling (MLM) objective (like BERT).

The abstract from the paper is the following:

*Language models have become a key step to achieve state-of-the art results in many different Natural Language
Processing (NLP) tasks. Leveraging the huge amount of unlabeled texts nowadays available, they provide an efficient way
to pre-train continuous word representations that can be fine-tuned for a downstream task, along with their
contextualization at the sentence level. This has been widely demonstrated for English using contextualized
representations (Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al.,
2019; Yang et al., 2019b). In this paper, we introduce and share FlauBERT, a model learned on a very large and
heterogeneous French corpus. Models of different sizes are trained using the new CNRS (French National Centre for
Scientific Research) Jean Zay supercomputer. We apply our French language models to diverse NLP tasks (text
classification, paraphrasing, natural language inference, parsing, word sense disambiguation) and show that most of the
time they outperform other pretraining approaches. Different versions of FlauBERT as well as a unified evaluation
protocol for the downstream tasks, called FLUE (French Language Understanding Evaluation), are shared to the research
community for further reproducible experiments in French NLP.*

This model was contributed by `formiel <https://huggingface.co/formiel>`__. The original code can be found `here <https://github.com/getalp/Flaubert>`__.

Tips:
- Like RoBERTa, without the sentence ordering prediction (so just trained on the MLM objective).

Args:
    pre_norm (`bool`, *optional*, defaults to `False`):
        Whether to apply the layer normalization before or after the feed forward layer following the attention in
        each layer (Vaswani et al., Tensor2Tensor for Neural Machine Translation. 2018)
    layerdrop (`float`, *optional*, defaults to 0.0):
        Probability to drop layers during training (Fan et al., Reducing Transformer Depth on Demand with
        Structured Dropout. ICLR 2020)
    vocab_size (`int`, *optional*, defaults to 30145):
        Vocabulary size of the FlauBERT model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``FlaubertModel`` or ``TFFlaubertModel``.
    emb_dim (`int`, *optional*, defaults to 2048):
        Dimensionality of the encoder layers and the pooler layer.
    n_layer (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for the attention mechanism
    gelu_activation (`bool`, *optional*, defaults to `True`):
        Whether or not to use a *gelu* activation instead of *relu*.
    sinusoidal_embeddings (`bool`, *optional*, defaults to `False`):
        Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
    causal (`bool`, *optional*, defaults to `False`):
        Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
        order to only attend to the left-side context instead if a bidirectional context.
    asm (`bool`, *optional*, defaults to `False`):
        Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
        layer.
    n_langs (`int`, *optional*, defaults to 1):
        The number of languages the model handles. Set to 1 for monolingual models.
    use_lang_emb (`bool`, *optional*, defaults to `True`)
        Whether to use language embeddings. Some models use additional language embeddings, see `the multilingual
        models page <http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings>`__ for information
        on how to use them.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    embed_init_std (`float`, *optional*, defaults to 2048^-0.5):
        The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.
    init_std (`int`, *optional*, defaults to 50257):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the
        embedding matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    bos_index (`int`, *optional*, defaults to 0):
        The index of the beginning of sentence token in the vocabulary.
    eos_index (`int`, *optional*, defaults to 1):
        The index of the end of sentence token in the vocabulary.
    pad_index (`int`, *optional*, defaults to 2):
        The index of the padding token in the vocabulary.
    unk_index (`int`, *optional*, defaults to 3):
        The index of the unknown token in the vocabulary.
    mask_index (`int`, *optional*, defaults to 5):
        The index of the masking token in the vocabulary.
    is_encoder(`bool`, *optional*, defaults to `True`):
        Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
    summary_type (`string`, *optional*, defaults to "first"):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Has to be one of the following options:

            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Used in the sequence classification and multiple choice models.

        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Used in the sequence classification and multiple choice models.

        The dropout ratio to be used after the projection and activation.
    start_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    end_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    mask_token_id (`int`, *optional*, defaults to 0):
        Model agnostic parameter to identify masked tokens when generating text in an MLM context.
    lang_id (`int`, *optional*, defaults to 1):
        The ID of the language used by the model. This parameter is used when generating text in a given language.

.. class:: transformers.models.fnet.configuration_fnet.FNetConfig(vocab_size=32000, hidden_size=768, num_hidden_layers=12, intermediate_size=3072, hidden_act='gelu_new', hidden_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=4, initializer_range=0.02, layer_norm_eps=1e-12, use_tpu_fourier_optimizations=False, tpu_short_seq_length=512, pad_token_id=3, bos_token_id=1, eos_token_id=2, **kwargs)

The FNet model was proposed in `FNet: Mixing Tokens with Fourier Transforms <https://arxiv.org/abs/2105.03824>`__ by
James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon. The model replaces the self-attention layer in a BERT
model with a fourier transform which returns only the real parts of the transform. The model is significantly faster
than the BERT model because it has fewer parameters and is more memory efficient. The model achieves about 92-97%
accuracy of BERT counterparts on GLUE benchmark, and trains much faster than the BERT model. The abstract from the
paper is the following:

*We show that Transformer encoder architectures can be sped up, with limited accuracy costs, by replacing the
self-attention sublayers with simple linear transformations that "mix" input tokens. These linear mixers, along with
standard nonlinearities in feed-forward layers, prove competent at modeling semantic relationships in several text
classification tasks. Most surprisingly, we find that replacing the self-attention sublayer in a Transformer encoder
with a standard, unparameterized Fourier Transform achieves 92-97% of the accuracy of BERT counterparts on the GLUE
benchmark, but trains 80% faster on GPUs and 70% faster on TPUs at standard 512 input lengths. At longer input lengths,
our FNet model is significantly faster: when compared to the "efficient" Transformers on the Long Range Arena
benchmark, FNet matches the accuracy of the most accurate models, while outpacing the fastest models across all
sequence lengths on GPUs (and across relatively shorter lengths on TPUs). Finally, FNet has a light memory footprint
and is particularly efficient at smaller model sizes; for a fixed speed and accuracy budget, small FNet models
outperform Transformer counterparts.*

Tips on usage:

- The model was trained without an attention mask as it is based on Fourier Transform. The model was trained with
  maximum sequence length 512 which includes pad tokens. Hence, it is highly recommended to use the same maximum
  sequence length for fine-tuning and inference.

This model was contributed by `gchhablani <https://huggingface.co/gchhablani>`__. The original code can be found `here <https://github.com/google-research/google-research/tree/master/f_net>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 32000):
        Vocabulary size of the FNet model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``FNetModel`` or ``TFFNetModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 4):
        The vocabulary size of the `token_type_ids` passed when calling ``FNetModel`` or ``TFFNetModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    use_tpu_fourier_optimizations (`bool`, *optional*, defaults to `False`):
        Determines whether to use TPU optimized FFTs. If `True`, the model will favor axis-wise FFTs transforms.
        Set to `False` for GPU/CPU hardware, in which case n-dimensional FFTs are used.
    tpu_short_seq_length (`int`, *optional*, defaults to 512):
        The sequence length that is expected by the model when using TPUs. This will be used to initialize the DFT
        matrix only when *use_tpu_fourier_optimizations* is set to `True` and the input sequence is shorter than or
        equal to 4096 tokens.

.. class:: transformers.models.gemma.configuration_gemma.GemmaConfig(vocab_size=256000, hidden_size=3072, intermediate_size=24576, num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=16, head_dim=256, hidden_act='gelu_pytorch_tanh', hidden_activation=None, max_position_embeddings=8192, initializer_range=0.02, rms_norm_eps=1e-06, use_cache=True, pad_token_id=0, eos_token_id=1, bos_token_id=2, tie_word_embeddings=True, rope_theta=10000.0, attention_bias=False, attention_dropout=0.0, **kwargs)

The Gemma model was proposed in `Gemma: Open Models Based on Gemini Technology and Research <https://blog.google/technology/developers/gemma-open-models/>`__ by Gemma Team, Google.
Gemma models are trained on 6T tokens, and released with 2 versions, 2b and 7b.

The abstract from the paper is the following:

*This work introduces Gemma, a new family of open language models demonstrating strong performance across academic benchmarks for language understanding, reasoning, and safety. We release two sizes of models (2 billion and 7 billion parameters), and provide both pretrained and fine-tuned checkpoints. Gemma outperforms similarly sized open models on 11 out of 18 text-based tasks, and we present comprehensive evaluations of safety and responsibility aspects of the models, alongside a detailed description of our model development. We believe the responsible release of LLMs is critical for improving the safety of frontier models, and for enabling the next wave of LLM innovations*

Tips:

- The original checkpoints can be converted using the conversion script `src/transformers/models/gemma/convert_gemma_weights_to_hf.py` 

This model was contributed by `Arthur Zucker <https://huggingface.co/ArthurZ>`__, `Younes Belkada <https://huggingface.co/ybelkada>`__, `Sanchit Gandhi <https://huggingface.co/sanchit-gandhi>`__, `Pedro Cuenca <https://huggingface.co/pcuenq>`__.


Args:
    vocab_size (`int`, *optional*, defaults to 256000):
        Vocabulary size of the Gemma model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``GemmaModel``
    hidden_size (`int`, *optional*, defaults to 3072):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 24576):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 28):
        Number of hidden layers in the Transformer decoder.
    num_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    num_key_value_heads (`int`, *optional*, defaults to 16):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to
        `num_attention_heads`.
    head_dim (`int`, *optional*, defaults to 256):
        The attention head dimension.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
        The legacy activation function. It is overwritten by the `hidden_activation`.
    hidden_activation (`str` or `function`, *optional*):
        The non-linear activation function (function or string) in the decoder. Will default to `"gelu_pytorch_tanh"`
        if not specified. `"gelu_pytorch_tanh"` uses an approximation of the `"gelu"` activation function.
    max_position_embeddings (`int`, *optional*, defaults to 8192):
        The maximum sequence length that this model might ever be used with.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    rms_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon used by the rms normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    pad_token_id (`int`, *optional*, defaults to 0):
        Padding token id.
    eos_token_id (`int`, *optional*, defaults to 1):
        End of stream token id.
    bos_token_id (`int`, *optional*, defaults to 2):
        Beginning of stream token id.
    tie_word_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to tie weight embeddings
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
        Whether to use a bias in the query, key, value and output projection layers during self-attention.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.

::

>>> from transformers import GemmaModel, GemmaConfig

>>> # Initializing a Gemma gemma-7b style configuration
>>> configuration = GemmaConfig()

>>> # Initializing a model from the gemma-7b style configuration
>>> model = GemmaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

.. class:: transformers.models.git.configuration_git.GitConfig(vision_config=None, vocab_size=30522, hidden_size=768, num_hidden_layers=6, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=1024, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type='absolute', use_cache=True, tie_word_embeddings=False, bos_token_id=101, eos_token_id=102, num_image_with_embedding=None, **kwargs)

The GIT model was proposed in `GIT: A Generative Image-to-text Transformer for Vision and Language <https://arxiv.org/abs/2205.14100>`__ by
Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, Lijuan Wang. GIT is a decoder-only Transformer
that leverages `CLIP <clip>`__'s vision encoder to condition the model on vision inputs besides text. The model obtains state-of-the-art results on
image captioning and visual question answering benchmarks.

The abstract from the paper is the following:

*In this paper, we design and train a Generative Image-to-text Transformer, GIT, to unify vision-language tasks such as image/video captioning and question answering. While generative models provide a consistent network architecture between pre-training and fine-tuning, existing work typically contains complex structures (uni/multi-modal encoder/decoder) and depends on external modules such as object detectors/taggers and optical character recognition (OCR). In GIT, we simplify the architecture as one image encoder and one text decoder under a single language modeling task. We also scale up the pre-training data and the model size to boost the model performance. Without bells and whistles, our GIT establishes new state of the arts on 12 challenging benchmarks with a large margin. For instance, our model surpasses the human performance for the first time on TextCaps (138.2 vs. 125.5 in CIDEr). Furthermore, we present a new scheme of generation-based image classification and scene text recognition, achieving decent performance on standard benchmarks.*

Tips:

- GIT is implemented in a very similar way to GPT-2, the only difference being that the model is also conditioned on `pixel_values`.
- One can use ``GitProcessor`` to prepare images for the model, and the `generate` method for autoregressive generation.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/git_architecture.jpg"
alt="drawing" width="600"/>

<small> GIT architecture. Taken from the <a href="https://arxiv.org/abs/2205.14100" target="_blank">original paper</a>. </small>

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__.
The original code can be found `here <https://github.com/microsoft/GenerativeImage2Text>`__.

Args:
    vision_config (`dict`, *optional*):
        Dictionary of configuration options used to initialize ``GitVisionConfig``.
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the GIT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``GitModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 6):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    num_image_with_embedding (`int`, *optional*):
        The number of temporal embeddings to add, in case the model is used for video captioning/VQA.

.. class:: transformers.models.gpt2.configuration_gpt2.GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, scale_attn_weights=True, use_cache=True, bos_token_id=50256, eos_token_id=50256, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False, **kwargs)

The GPT-Sw3 model was first proposed in
`Lessons Learned from GPT-SW3: Building the First Large-Scale Generative Language Model for Swedish <http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.376.pdf>`__
by Ariel Ekgren, Amaru Cuba Gyllensten, Evangelia Gogoulou, Alice Heiman, Severine Verlinden, Joey Öhman,
Fredrik Carlsson, Magnus Sahlgren.

Since that first paper the authors have extended their work and trained new models on their new 1.2TB corpora named The Nordic Pile.

GPT-Sw3 is a collection of large decoder-only pretrained transformer language models that were developed by AI Sweden
in collaboration with RISE and the WASP WARA for Media and Language. GPT-Sw3 has been trained on a dataset containing
320B tokens in Swedish, Norwegian, Danish, Icelandic, English, and programming code. The model was pretrained using a
causal language modeling (CLM) objective utilizing the NeMo Megatron GPT implementation.

This model was contributed by `AI Sweden <https://huggingface.co/AI-Sweden>`__.

The implementation uses the `GPT2Model <https://huggingface.co/docs/transformers/model_doc/gpt2>`__ coupled
with our `GPTSw3Tokenizer`. This means that `AutoTokenizer` and `AutoModelForCausalLM` map to our tokenizer
implementation and the corresponding GPT2 model implementation respectively.
*Note that sentencepiece is required to use our tokenizer and can be installed with:* `pip install transformers[sentencepiece]` or `pip install sentencepiece`

.. class:: transformers.models.gpt2.configuration_gpt2.GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None, activation_function='gelu_new', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, scale_attn_weights=True, use_cache=True, bos_token_id=50256, eos_token_id=50256, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False, **kwargs)

OpenAI GPT-2 model was proposed in `Language Models are Unsupervised Multitask Learners <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`__ by Alec
Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever from `OpenAI <https://huggingface.co/openai>`__. It's a causal (unidirectional)
transformer pretrained using language modeling on a very large corpus of ~40 GB of text data.

The abstract from the paper is the following:

*GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset`1] of 8 million
web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some
text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks
across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than
10X the amount of data.*

Tips:

- GPT-2 is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- GPT-2 was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as it can be
  observed in the *run_generation.py* example script.
- The model can take the *past_key_values* (for PyTorch) or *past* (for TF) as input, which is the previously computed
  key/value attention pairs. Using this (*past_key_values* or *past*) value prevents the model from re-computing
  pre-computed values in the context of text generation. For PyTorch, see *past_key_values* argument of the
  ``GPT2Model.forward`` method, or for TF the *past* argument of the
  ``TFGPT2Model.call`` method for more information on its usage.
- Enabling the *scale_attn_by_inverse_layer_idx* and *reorder_and_upcast_attn* flags will apply the training stability
  improvements from [Mistral <https://github.com/stanford-crfm/mistral/>`__ (for PyTorch only).

`Write With Transformer <https://transformer.huggingface.co/doc/gpt2-large>`__ is a webapp created and hosted by
Hugging Face showcasing the generative capabilities of several models. GPT-2 is one of them and is available in five
different sizes: small, medium, large, xl and a distilled version of the small checkpoint: *distilgpt-2*.

This model was contributed by `thomwolf <https://huggingface.co/thomwolf>`__. The original code can be found `here <https://openai.com/blog/better-language-models/>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50257):
        Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``GPT2Model`` or ``TFGPT2Model``.
    n_positions (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_embd (`int`, *optional*, defaults to 768):
        Dimensionality of the embeddings and hidden states.
    n_layer (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    n_inner (`int`, *optional*):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
    activation_function (`str`, *optional*, defaults to `"gelu_new"`):
        Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    resid_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the embeddings.
    attn_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    summary_type (`string`, *optional*, defaults to `"cls_index"`):
        Argument used when doing sequence summary, used in the models ``GPT2DoubleHeadsModel`` and
        ``TFGPT2DoubleHeadsModel``.

        Has to be one of the following options:

            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models ``GPT2DoubleHeadsModel`` and
        ``TFGPT2DoubleHeadsModel``.

        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in for the multiple choice head in
        ``GPT2DoubleHeadsModel``.

        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models ``GPT2DoubleHeadsModel`` and
        ``TFGPT2DoubleHeadsModel``.

        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Argument used when doing sequence summary, used in the models ``GPT2DoubleHeadsModel`` and
        ``TFGPT2DoubleHeadsModel``.

        The dropout ratio to be used after the projection and activation.
    scale_attn_weights (`bool`, *optional*, defaults to `True`):
        Scale attention weights by dividing by sqrt(hidden_size)..
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    bos_token_id (`int`, *optional*, defaults to 50256):
        Id of the beginning of sentence token in the vocabulary.
    eos_token_id (`int`, *optional*, defaults to 50256):
        Id of the end of sentence token in the vocabulary.
    scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
        Whether to additionally scale attention weights by `1 / layer_idx + 1`.
    reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
        Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
        dot-product/softmax to float() when training with mixed precision.

.. class:: transformers.models.gpt_bigcode.configuration_gpt_bigcode.GPTBigCodeConfig(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None, activation_function='gelu_pytorch_tanh', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, scale_attn_weights=True, use_cache=True, bos_token_id=50256, eos_token_id=50256, attention_softmax_in_fp32=True, scale_attention_softmax_in_fp32=True, multi_query=True, **kwargs)

The GPTBigCode model was proposed in `SantaCoder: don't reach for the stars! <https://arxiv.org/abs/2301.03988>`__ by BigCode. The listed authors are: Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, Logesh Kumar Umapathi, Carolyn Jane Anderson, Yangtian Zi, Joel Lamy Poirier, Hailey Schoelkopf, Sergey Troshin, Dmitry Abulkhanov, Manuel Romero, Michael Lappert, Francesco De Toni, Bernardo García del Río, Qian Liu, Shamik Bose, Urvashi Bhattacharyya, Terry Yue Zhuo, Ian Yu, Paulo Villegas, Marco Zocca, Sourab Mangrulkar, David Lansky, Huu Nguyen, Danish Contractor, Luis Villa, Jia Li, Dzmitry Bahdanau, Yacine Jernite, Sean Hughes, Daniel Fried, Arjun Guha, Harm de Vries, Leandro von Werra.

The abstract from the paper is the following:uery

*The BigCode project is an open-scientific collaboration working on the responsible development of large language models for code. This tech report describes the progress of the collaboration until December 2022, outlining the current state of the Personally Identifiable Information (PII) redaction pipeline, the experiments conducted to de-risk the model architecture, and the experiments investigating better preprocessing methods for the training data. We train 1.1B parameter models on the Java, JavaScript, and Python subsets of The Stack and evaluate them on the MultiPL-E text-to-code benchmark. We find that more aggressive filtering of near-duplicates can further boost performance and, surprisingly, that selecting files from repositories with 5+ GitHub stars deteriorates performance significantly. Our best model outperforms previous open-source multilingual code generation models (InCoder-6.7B and CodeGen-Multi-2.7B) in both left-to-right generation and infilling on the Java, JavaScript, and Python portions of MultiPL-E, despite being a substantially smaller model. All models are released under an OpenRAIL license at `this https URL. <https://huggingface.co/bigcode>`__*

The model is a an optimized `GPT2 model <https://huggingface.co/docs/transformers/model_doc/gpt2>`__ with support for Multi-Query Attention.

Args:
    vocab_size (`int`, *optional*, defaults to 50257):
        Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``GPTBigCodeModel``.
    n_positions (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_embd (`int`, *optional*, defaults to 768):
        Dimensionality of the embeddings and hidden states.
    n_layer (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    n_inner (`int`, *optional*, defaults to None):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
    activation_function (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
        Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new",
        "gelu_pytorch_tanh"]`.
    resid_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the embeddings.
    attn_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
        The epsilon to use in the layer normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    scale_attn_weights (`bool`, *optional*, defaults to `True`):
        Scale attention weights by dividing by sqrt(hidden_size)..
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether to call the fused softmax in float32.
    scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether to scale the attention softmax in float32.
    attention_type (`bool`, *optional*, defaults to `True`):
        Whether to use Multi-Query Attion (`True`) or Multi-Head Attention (`False`).

.. class:: transformers.models.gpt_neox.configuration_gpt_neox.GPTNeoXConfig(vocab_size=50432, hidden_size=6144, num_hidden_layers=44, num_attention_heads=64, intermediate_size=24576, hidden_act='gelu', rotary_pct=0.25, rotary_emb_base=10000, attention_dropout=0.0, hidden_dropout=0.0, classifier_dropout=0.1, max_position_embeddings=2048, initializer_range=0.02, layer_norm_eps=1e-05, use_cache=True, bos_token_id=0, eos_token_id=2, tie_word_embeddings=False, use_parallel_residual=True, rope_scaling=None, attention_bias=True, **kwargs)

We introduce GPT-NeoX-20B, a 20 billion parameter autoregressive language model trained on the Pile, whose weights will
be made freely and openly available to the public through a permissive license. It is, to the best of our knowledge,
the largest dense autoregressive model that has publicly available weights at the time of submission. In this work,
we describe GPT-NeoX-20B's architecture and training and evaluate its performance on a range of language-understanding,
mathematics, and knowledge-based tasks. We find that GPT-NeoX-20B is a particularly powerful few-shot reasoner and
gains far more in performance when evaluated five-shot than similarly sized GPT-3 and FairSeq models. We open-source
the training and evaluation code, as well as the model weights, at `https://github.com/EleutherAI/gpt-neox <https://github.com/EleutherAI/gpt-neox>`__.

Development of the model was led by Sid Black, Stella Biderman and Eric Hallahan, and the model was trained with
generous the support of `CoreWeave <https://www.coreweave.com/>`__.

GPT-NeoX-20B was trained with fp16, thus it is recommended to initialize the model as follows:

::

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b").half().cuda()

GPT-NeoX-20B also has a different tokenizer from the one used in GPT-J-6B and GPT-Neo. The new tokenizer allocates
additional tokens to whitespace characters, making the model more suitable for certain tasks like code generation.

#Args:
    vocab_size (`int`, *optional*, defaults to 50432):
        Vocabulary size of the GPTNeoX model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``GPTNeoXModel``.
    hidden_size (`int`, *optional*, defaults to 6144):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 44):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 64):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 24576):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    rotary_pct (`float`, *optional*, defaults to 0.25):
        percentage of hidden dimensions to allocate to rotary embeddings
    rotary_emb_base (`int`, *optional*, defaults to 10000)
        base for computing rotary embeddings frequency
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio probability of the attention score.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio of (1) the word embeddings, (2) the post-attention hidden states, and (3) the post-mlp
        hidden states.
    classifier_dropout (`float`, *optional*, defaults to 0.1):
        Argument used when doing token classification, used in the model ``GPTNeoXForTokenClassification``.

        The dropout ratio for the hidden layer.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    initializer_range (`float`, *optional*, defaults to 1e-5):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    use_parallel_residual (`bool`, *optional*, defaults to `True`):
        Whether to use a "parallel" formulation in each Transformer layer, which can provide a slight training
        speedup at large scales (e.g. 20B).
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
        strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
        `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
        experimental feature, subject to breaking API changes in future versions.
    attention_bias (`bool`, *optional*, defaults to `True`):
        Whether to use a bias in the query, key, value and output projection layers during self-attention.

.. class:: transformers.models.gpt_neox_japanese.configuration_gpt_neox_japanese.GPTNeoXJapaneseConfig(vocab_size=32000, hidden_size=2560, num_hidden_layers=32, num_attention_heads=32, intermediate_multiple_size=4, hidden_act='gelu', rotary_pct=1.0, rotary_emb_base=10000, max_position_embeddings=2048, initializer_range=0.02, layer_norm_eps=1e-05, use_cache=True, bos_token_id=31996, eos_token_id=31999, attention_dropout=0.1, hidden_dropout=0.0, **kwargs)

We introduce GPT-NeoX-Japanese, which is an autoregressive language model for Japanese, trained on top of `https://github.com/EleutherAI/gpt-neox <https://github.com/EleutherAI/gpt-neox>`__.
Japanese is a unique language with its large vocabulary and a combination of hiragana, katakana, and kanji writing scripts.
To address this distinct structure of the Japanese language, we use a `special sub-word tokenizer <https://github.com/tanreinama/Japanese-BPEEncoder_V2>`__. We are very grateful to *tanreinama* for open-sourcing this incredibly helpful tokenizer.
Following the recommendations from Google's research on `PaLM <https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html>`__, we have removed bias parameters from transformer blocks, achieving better model performance. Please refer `this article <https://medium.com/ml-abeja/training-a-better-gpt-2-93b157662ae4>`__ in detail.

Development of the model was led by `Shinya Otani <https://github.com/SO0529>`__, `Takayoshi Makabe <https://github.com/spider-man-tm>`__, `Anuj Arora <https://github.com/Anuj040>`__, and `Kyo Hattori <https://github.com/go5paopao>`__ from `ABEJA, Inc. <https://www.abejainc.com/>`__. For more information on this model-building activity, please refer `here (ja) <https://tech-blog.abeja.asia/entry/abeja-gpt-project-202207>`__.

#Args:
    vocab_size (`int`, *optional*, defaults to 32000):
        Vocabulary size of the GPTNeoXJapanese model. Defines the number of different tokens that can be
        represented by the `inputs_ids` passed when calling ``GPTNeoXJapanese``.
    hidden_size (`int`, *optional*, defaults to 2560):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_multiple_size (`int`, *optional*, defaults to 4):
        Dimension of the "intermediate" layer in the Transformer encoder is calculated by hidden_size *
        intermediate_multiple_size.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler.
    rotary_pct (`float`, *optional*, defaults to 1.00):
        percentage of hidden dimensions to allocate to rotary embeddings
    rotary_emb_base (`int`, *optional*, defaults to 10000)
        base for computing rotary embeddings frequency
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        The epsilon used by the layer normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the hidden layer.

.. class:: transformers.models.gptj.configuration_gptj.GPTJConfig(vocab_size=50400, n_positions=2048, n_embd=4096, n_layer=28, n_head=16, rotary_dim=64, n_inner=None, activation_function='gelu_new', resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, layer_norm_epsilon=1e-05, initializer_range=0.02, use_cache=True, bos_token_id=50256, eos_token_id=50256, tie_word_embeddings=False, **kwargs)

The GPT-J model was released in the `kingoflolz/mesh-transformer-jax <https://github.com/kingoflolz/mesh-transformer-jax>`__ repository by Ben Wang and Aran Komatsuzaki. It is a GPT-2-like
causal language model trained on `the Pile <https://pile.eleuther.ai/>`__ dataset.

This model was contributed by `Stella Biderman <https://huggingface.co/stellaathena>`__.

Tips:

- To load `GPT-J <https://huggingface.co/EleutherAI/gpt-j-6B>`__ in float32 one would need at least 2x model size
  RAM: 1x for initial weights and another 1x to load the checkpoint. So for GPT-J it would take at least 48GB
  RAM to just load the model. To reduce the RAM usage there are a few options. The `torch_dtype` argument can be
  used to initialize the model in half-precision on a CUDA device only. There is also a fp16 branch which stores the fp16 weights,
  which could be used to further minimize the RAM usage:

::

>>> from transformers import GPTJForCausalLM
>>> import torch

>>> device = "cuda"
>>> model = GPTJForCausalLM.from_pretrained(
...     "EleutherAI/gpt-j-6B",
...     revision="float16",
...     torch_dtype=torch.float16,
... ).to(device)

- The model should fit on 16GB GPU for inference. For training/fine-tuning it would take much more GPU RAM. Adam
  optimizer for example makes four copies of the model: model, gradients, average and squared average of the gradients.
  So it would need at least 4x model size GPU memory, even with mixed precision as gradient updates are in fp32. This
  is not including the activations and data batches, which would again require some more GPU RAM. So one should explore
  solutions such as DeepSpeed, to train/fine-tune the model. Another option is to use the original codebase to
  train/fine-tune the model on TPU and then convert the model to Transformers format for inference. Instructions for
  that could be found `here <https://github.com/kingoflolz/mesh-transformer-jax/blob/master/howto_finetune.md>`__

- Although the embedding matrix has a size of 50400, only 50257 entries are used by the GPT-2 tokenizer. These extra
  tokens are added for the sake of efficiency on TPUs. To avoid the mismatch between embedding matrix size and vocab
  size, the tokenizer for `GPT-J <https://huggingface.co/EleutherAI/gpt-j-6B>`__ contains 143 extra tokens
  `<|extratoken_1|>... <|extratoken_143|>`, so the `vocab_size` of tokenizer also becomes 50400.

#Args:
    vocab_size (`int`, *optional*, defaults to 50400):
        Vocabulary size of the GPT-J model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``GPTJModel``.
    n_positions (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_embd (`int`, *optional*, defaults to 4096):
        Dimensionality of the embeddings and hidden states.
    n_layer (`int`, *optional*, defaults to 28):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    rotary_dim (`int`, *optional*, defaults to 64):
        Number of dimensions in the embedding that Rotary Position Embedding is applied to.
    n_inner (`int`, *optional*, defaults to None):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
    activation_function (`str`, *optional*, defaults to `"gelu_new"`):
        Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
    resid_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (`int`, *optional*, defaults to 0.1):
        The dropout ratio for the embeddings.
    attn_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
        The epsilon to use in the layer normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.ibert.configuration_ibert.IBertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type='absolute', quant_mode=False, force_dequant='none', **kwargs)

The I-BERT model was proposed in `I-BERT: Integer-only BERT Quantization <https://arxiv.org/abs/2101.01321>`__ by
Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney and Kurt Keutzer. It's a quantized version of RoBERTa running
inference up to four times faster.

The abstract from the paper is the following:

*Transformer based models, like BERT and RoBERTa, have achieved state-of-the-art results in many Natural Language
Processing tasks. However, their memory footprint, inference latency, and power consumption are prohibitive for
efficient inference at the edge, and even at the data center. While quantization can be a viable solution for this,
previous work on quantizing Transformer based models use floating-point arithmetic during inference, which cannot
efficiently utilize integer-only logical units such as the recent Turing Tensor Cores, or traditional integer-only ARM
processors. In this work, we propose I-BERT, a novel quantization scheme for Transformer based models that quantizes
the entire inference with integer-only arithmetic. Based on lightweight integer-only approximation methods for
nonlinear operations, e.g., GELU, Softmax, and Layer Normalization, I-BERT performs an end-to-end integer-only BERT
inference without any floating point calculation. We evaluate our approach on GLUE downstream tasks using
RoBERTa-Base/Large. We show that for both cases, I-BERT achieves similar (and slightly higher) accuracy as compared to
the full-precision baseline. Furthermore, our preliminary implementation of I-BERT shows a speedup of 2.4 - 4.0x for
INT8 inference on a T4 GPU system as compared to FP32 inference. The framework has been developed in PyTorch and has
been open-sourced.*

This model was contributed by `kssteven <https://huggingface.co/kssteven>`__. The original code can be found `here <https://github.com/kssteven418/I-BERT>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the I-BERT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``IBertModel``
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``IBertModel``
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    quant_mode (`bool`, *optional*, defaults to `False`):
        Whether to quantize the model or not.
    force_dequant (`str`, *optional*, defaults to `"none"`):
        Force dequantize specific nonlinear layer. Dequatized layers are then executed with full precision.
        `"none"`, `"gelu"`, `"softmax"`, `"layernorm"` and `"nonlinear"` are supported. As deafult, it is set as
        `"none"`, which does not dequantize any layers. Please specify `"gelu"`, `"softmax"`, or `"layernorm"` to
        dequantize GELU, Softmax, or LayerNorm, respectively. `"nonlinear"` will dequantize all nonlinear layers,
        i.e., GELU, Softmax, and LayerNorm.

.. class:: transformers.models.imagegpt.configuration_imagegpt.ImageGPTConfig(vocab_size=513, n_positions=1024, n_embd=512, n_layer=24, n_head=8, n_inner=None, activation_function='quick_gelu', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, scale_attn_weights=True, use_cache=True, tie_word_embeddings=False, scale_attn_by_inverse_layer_idx=False, reorder_and_upcast_attn=False, **kwargs)

The ImageGPT model was proposed in `Generative Pretraining from Pixels <https://openai.com/blog/image-gpt>`__ by Mark
Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, Ilya Sutskever. ImageGPT (iGPT) is a GPT-2-like
model trained to predict the next pixel value, allowing for both unconditional and conditional image generation.

The abstract from the paper is the following:

*Inspired by progress in unsupervised representation learning for natural language, we examine whether similar models
can learn useful representations for images. We train a sequence Transformer to auto-regressively predict pixels,
without incorporating knowledge of the 2D input structure. Despite training on low-resolution ImageNet without labels,
we find that a GPT-2 scale model learns strong image representations as measured by linear probing, fine-tuning, and
low-data classification. On CIFAR-10, we achieve 96.3% accuracy with a linear probe, outperforming a supervised Wide
ResNet, and 99.0% accuracy with full fine-tuning, matching the top supervised pre-trained models. We are also
competitive with self-supervised benchmarks on ImageNet when substituting pixels for a VQVAE encoding, achieving 69.0%
top-1 accuracy on a linear probe of our features.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/imagegpt_architecture.png"
alt="drawing" width="600"/>

<small> Summary of the approach. Taken from the `original paper <https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf>`__. </small>

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__, based on `this issue <https://github.com/openai/image-gpt/issues/7>`__. The original code can be found
`here <https://github.com/openai/image-gpt>`__.

Tips:

- ImageGPT is almost exactly the same as `GPT-2 <gpt2>`__, with the exception that a different activation
  function is used (namely "quick gelu"), and the layer normalization layers don't mean center the inputs. ImageGPT
  also doesn't have tied input- and output embeddings.
- As the time- and memory requirements of the attention mechanism of Transformers scales quadratically in the sequence
  length, the authors pre-trained ImageGPT on smaller input resolutions, such as 32x32 and 64x64. However, feeding a
  sequence of 32x32x3=3072 tokens from 0..255 into a Transformer is still prohibitively large. Therefore, the authors
  applied k-means clustering to the (R,G,B) pixel values with k=512. This way, we only have a 32*32 = 1024-long
  sequence, but now of integers in the range 0..511. So we are shrinking the sequence length at the cost of a bigger
  embedding matrix. In other words, the vocabulary size of ImageGPT is 512, + 1 for a special "start of sentence" (SOS)
  token, used at the beginning of every sequence. One can use ``ImageGPTImageProcessor`` to prepare
  images for the model.
- Despite being pre-trained entirely unsupervised (i.e. without the use of any labels), ImageGPT produces fairly
  performant image features useful for downstream tasks, such as image classification. The authors showed that the
  features in the middle of the network are the most performant, and can be used as-is to train a linear model (such as
  a sklearn logistic regression model for example). This is also referred to as "linear probing". Features can be
  easily obtained by first forwarding the image through the model, then specifying `output_hidden_states=True`, and
  then average-pool the hidden states at whatever layer you like.
- Alternatively, one can further fine-tune the entire model on a downstream dataset, similar to BERT. For this, you can
  use ``ImageGPTForImageClassification``.
- ImageGPT comes in different sizes: there's ImageGPT-small, ImageGPT-medium and ImageGPT-large. The authors did also
  train an XL variant, which they didn't release. The differences in size are summarized in the following table:

| **Model variant** | **Depths** | **Hidden sizes** | **Decoder hidden size** | **Params (M)** | **ImageNet-1k Top 1** |
|---|---|---|---|---|---|
| MiT-b0 | [2, 2, 2, 2] | [32, 64, 160, 256] | 256 | 3.7 | 70.5 |
| MiT-b1 | [2, 2, 2, 2] | [64, 128, 320, 512] | 256 | 14.0 | 78.7 |
| MiT-b2 | [3, 4, 6, 3] | [64, 128, 320, 512] | 768 | 25.4 | 81.6 |
| MiT-b3 | [3, 4, 18, 3] | [64, 128, 320, 512] | 768 | 45.2 | 83.1 |
| MiT-b4 | [3, 8, 27, 3] | [64, 128, 320, 512] | 768 | 62.6 | 83.6 |
| MiT-b5 | [3, 6, 40, 3] | [64, 128, 320, 512] | 768 | 82.0 | 83.8 |

Args:
    vocab_size (`int`, *optional*, defaults to 512):
        Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``ImageGPTModel`` or ``TFImageGPTModel``.
    n_positions (`int`, *optional*, defaults to 32*32):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_embd (`int`, *optional*, defaults to 512):
        Dimensionality of the embeddings and hidden states.
    n_layer (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the Transformer encoder.
    n_inner (`int`, *optional*, defaults to None):
        Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
    activation_function (`str`, *optional*, defaults to `"quick_gelu"`):
        Activation function (can be one of the activation functions defined in src/transformers/activations.py).
        Defaults to "quick_gelu".
    resid_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (`int`, *optional*, defaults to 0.1):
        The dropout ratio for the embeddings.
    attn_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
        The epsilon to use in the layer normalization layers.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    scale_attn_weights (`bool`, *optional*, defaults to `True`):
        Scale attention weights by dividing by sqrt(hidden_size)..
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
        Whether to additionally scale attention weights by `1 / layer_idx + 1`.
    reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
        Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
        dot-product/softmax to float() when training with mixed precision.

.. class:: transformers.models.layoutlm.configuration_layoutlm.LayoutLMConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type='absolute', use_cache=True, max_2d_position_embeddings=1024, **kwargs)

The LayoutLM model was proposed in the paper `LayoutLM: Pre-training of Text and Layout for Document Image
Understanding <https://arxiv.org/abs/1912.13318>`__ by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and
Ming Zhou. It's a simple but effective pretraining method of text and layout for document image understanding and
information extraction tasks, such as form understanding and receipt understanding. It obtains state-of-the-art results
on several downstream tasks:

- form understanding: the `FUNSD <https://guillaumejaume.github.io/FUNSD/>`__ dataset (a collection of 199 annotated
  forms comprising more than 30,000 words).
- receipt understanding: the `SROIE <https://rrc.cvc.uab.es/?ch=13>`__ dataset (a collection of 626 receipts for
  training and 347 receipts for testing).
- document image classification: the `RVL-CDIP <https://www.cs.cmu.edu/~aharley/rvl-cdip/>`__ dataset (a collection of
  400,000 images belonging to one of 16 classes).

The abstract from the paper is the following:

*Pre-training techniques have been verified successfully in a variety of NLP tasks in recent years. Despite the
widespread use of pretraining models for NLP applications, they almost exclusively focus on text-level manipulation,
while neglecting layout and style information that is vital for document image understanding. In this paper, we propose
the LayoutLM to jointly model interactions between text and layout information across scanned document images, which is
beneficial for a great number of real-world document image understanding tasks such as information extraction from
scanned documents. Furthermore, we also leverage image features to incorporate words' visual information into LayoutLM.
To the best of our knowledge, this is the first time that text and layout are jointly learned in a single framework for
document-level pretraining. It achieves new state-of-the-art results in several downstream tasks, including form
understanding (from 70.72 to 79.27), receipt understanding (from 94.02 to 95.24) and document image classification
(from 93.07 to 94.42).*

Tips:

- In addition to *input_ids*, ``~transformers.LayoutLMModel.forward`` also expects the input `bbox`, which are
  the bounding boxes (i.e. 2D-positions) of the input tokens. These can be obtained using an external OCR engine such
  as Google's `Tesseract <https://github.com/tesseract-ocr/tesseract>`__ (there's a `Python wrapper <https://pypi.org/project/pytesseract/>`__ available). Each bounding box should be in (x0, y0, x1, y1) format, where
  (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1, y1) represents the
  position of the lower right corner. Note that one first needs to normalize the bounding boxes to be on a 0-1000
  scale. To normalize, you can use the following function:

::

def normalize_bbox(bbox, width, height):
    return `
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]

Here, `width` and `height` correspond to the width and height of the original document in which the token
occurs. Those can be obtained using the Python Image Library (PIL) library for example, as follows:

::

from PIL import Image

# Document can be a png, jpg, etc. PDFs must be converted to images.
image = Image.open(name_of_your_document).convert("RGB")

width, height = image.size

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the LayoutLM model. Defines the different tokens that can be represented by the
        *inputs_ids* passed to the forward method of ``LayoutLMModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed into ``LayoutLMModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    pad_token_id (`int`, *optional*, defaults to 0):
        The value used to pad input_ids.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        [Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    max_2d_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the 2D position embedding might ever used. Typically set this to something large
        just in case (e.g., 1024).

.. class:: transformers.models.led.configuration_led.LEDConfig(vocab_size=50265, max_encoder_position_embeddings=16384, max_decoder_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=2, classifier_dropout=0.0, pad_token_id=1, bos_token_id=0, eos_token_id=2, attention_window: Union[List[int], int] = 512, **kwargs)

The LED model was proposed in `Longformer: The Long-Document Transformer <https://arxiv.org/abs/2004.05150>`__ by Iz
Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales
quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention
mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or
longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local
windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we
evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In
contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our
pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on
WikiHop and TriviaQA. We finally introduce the Longformer-Encoder-Decoder (LED), a Longformer variant for supporting
long document generative sequence-to-sequence tasks, and demonstrate its effectiveness on the arXiv summarization
dataset.*

Tips:

- ``LEDForConditionalGeneration`` is an extension of
  ``BartForConditionalGeneration`` exchanging the traditional *self-attention* layer with
  *Longformer*'s *chunked self-attention* layer. ``LEDTokenizer`` is an alias of
  ``BartTokenizer``.
- LED works very well on long-range *sequence-to-sequence* tasks where the `input_ids` largely exceed a length of
  1024 tokens.
- LED pads the `input_ids` to be a multiple of `config.attention_window` if required. Therefore a small speed-up is
  gained, when ``LEDTokenizer`` is used with the `pad_to_multiple_of` argument.
- LED makes use of *global attention* by means of the `global_attention_mask` (see
  ``LongformerModel``). For summarization, it is advised to put *global attention* only on the first
  `<s>` token. For question answering, it is advised to put *global attention* on all tokens of the question.
- To fine-tune LED on all 16384, *gradient checkpointing* can be enabled in case training leads to out-of-memory (OOM)
  errors. This can be done by executing `model.gradient_checkpointing_enable()`. 
 Moreover, the `use_cache=False`
  flag can be used to disable the caching mechanism to save memory.
- A notebook showing how to evaluate LED, can be accessed `here <https://colab.research.google.com/drive/12INTTR6n64TzS4RrXZxMSXfrOd9Xzamo?usp=sharing>`__.
- A notebook showing how to fine-tune LED, can be accessed `here <https://colab.research.google.com/drive/12LjJazBl7Gam0XBPy_y0CTOJZeZ34c2v?usp=sharing>`__.
- LED is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.

This model was contributed by `patrickvonplaten <https://huggingface.co/patrickvonplaten>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the LED model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``LEDModel`` or ``TFLEDModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_encoder_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that the encoder might ever be used with.
    max_decoder_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that the decoder might ever be used with.
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)

.. class:: transformers.models.llama.configuration_llama.LlamaConfig(vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None, hidden_act='silu', max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=1e-06, use_cache=True, pad_token_id=None, bos_token_id=1, eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, rope_theta=10000.0, rope_scaling=None, attention_bias=False, attention_dropout=0.0, **kwargs)

The LLaMA model was proposed in `LLaMA: Open and Efficient Foundation Language Models <https://arxiv.org/abs/2302.13971>`__ by Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. It is a collection of foundation language models ranging from 7B to 65B parameters.

The abstract from the paper is the following:

*We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community. *

Tips:

- Weights for the LLaMA models can be obtained from by filling out `this form <https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form>`__
- After downloading the weights, they will need to be converted to the Hugging Face Transformers format using the `conversion script <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py>`__. The script can be called with the following (example) command:

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- After conversion, the model and tokenizer can be loaded via:

::

from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("/output/path")
model = LlamaForCausalLM.from_pretrained("/output/path")

Note that executing the script requires enough CPU RAM to host the whole model in float16 precision (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM). For the 65B model, it's thus 130GB of RAM needed.

- The LLaMA tokenizer is a BPE model based on `sentencepiece <https://github.com/google/sentencepiece>`__. One quirk of sentencepiece is that when decoding a sequence, if the first token is the start of the word (e.g. "Banana"), the tokenizer does not prepend the prefix space to the string.

This model was contributed by `zphang <https://huggingface.co/zphang>`__ with contributions from `BlackSamorez <https://huggingface.co/BlackSamorez>`__. The code of the implementation in Hugging Face is based on GPT-NeoX `here <https://github.com/EleutherAI/gpt-neox>`__. The original code of the authors can be found `here <https://github.com/facebookresearch/llama>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 32000):
        Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``LlamaModel``
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 11008):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer decoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer decoder.
    num_key_value_heads (`int`, *optional*):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to
        `num_attention_heads`.
    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
        Llama 2 up to 4096, CodeLlama up to 16384.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    rms_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon used by the rms normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    pad_token_id (`int`, *optional*):
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 1):
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 2):
        End of stream token id.
    pretraining_tp (`int`, *optional*, defaults to 1):
        Experimental feature. Tensor parallelism rank used during pretraining. Please refer to `this
        document <https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism>`__ to understand more about it. This value is
        necessary to ensure exact reproducibility of the pretraining results. Please refer to `this
        issue <https://github.com/pytorch/pytorch/issues/76232>`__.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to tie weight embeddings
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
        strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
        `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
        experimental feature, subject to breaking API changes in future versions.
    attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
        Whether to use a bias in the query, key, value and output projection layers during self-attention.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.

::

>>> from transformers import LlamaModel, LlamaConfig

>>> # Initializing a LLaMA llama-7b style configuration
>>> configuration = LlamaConfig()

>>> # Initializing a model from the llama-7b style configuration
>>> model = LlamaModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

.. class:: transformers.models.longformer.configuration_longformer.LongformerConfig(attention_window: Union[List[int], int] = 512, sep_token_id: int = 2, pad_token_id: int = 1, bos_token_id: int = 0, eos_token_id: int = 2, vocab_size: int = 30522, hidden_size: int = 768, num_hidden_layers: int = 12, num_attention_heads: int = 12, intermediate_size: int = 3072, hidden_act: str = 'gelu', hidden_dropout_prob: float = 0.1, attention_probs_dropout_prob: float = 0.1, max_position_embeddings: int = 512, type_vocab_size: int = 2, initializer_range: float = 0.02, layer_norm_eps: float = 1e-12, onnx_export: bool = False, **kwargs)

The Longformer model was presented in `Longformer: The Long-Document Transformer <https://arxiv.org/pdf/2004.05150.pdf>`__ by Iz Beltagy, Matthew E. Peters, Arman Cohan.

The abstract from the paper is the following:

*Transformer-based models are unable to process long sequences due to their self-attention operation, which scales
quadratically with the sequence length. To address this limitation, we introduce the Longformer with an attention
mechanism that scales linearly with sequence length, making it easy to process documents of thousands of tokens or
longer. Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local
windowed attention with a task motivated global attention. Following prior work on long-sequence transformers, we
evaluate Longformer on character-level language modeling and achieve state-of-the-art results on text8 and enwik8. In
contrast to most prior work, we also pretrain Longformer and finetune it on a variety of downstream tasks. Our
pretrained Longformer consistently outperforms RoBERTa on long document tasks and sets new state-of-the-art results on
WikiHop and TriviaQA.*

Tips:

- Since the Longformer is based on RoBERTa, it doesn't have `token_type_ids`. You don't need to indicate which
  token belongs to which segment. Just separate your segments with the separation token `tokenizer.sep_token` (or
  `</s>`).
- A transformer model replacing the attention matrices by sparse matrices to go faster. Often, the local context (e.g., what are the two tokens left and right?) is enough to take action for a given token. Some preselected input tokens are still given global attention, but the attention matrix has way less parameters, resulting in a speed-up. See the local attention section for more information.

This model was contributed by `beltagy <https://huggingface.co/beltagy>`__. The Authors' code can be found `here <https://github.com/allenai/longformer>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the Longformer model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``LongformerModel`` or ``TFLongformerModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``LongformerModel`` or
        ``TFLongformerModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    attention_window (`int` or `List[int]`, *optional*, defaults to 512):
        Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a
        different window size for each layer, use a `List[int]` where `len(attention_window) == num_hidden_layers`.

.. class:: transformers.models.longt5.configuration_longt5.LongT5Config(vocab_size=32128, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_decoder_layers=None, num_heads=8, local_radius=127, global_block_size=16, relative_attention_num_buckets=32, relative_attention_max_distance=128, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='relu', is_encoder_decoder=True, encoder_attention_type='local', use_cache=True, pad_token_id=0, eos_token_id=1, **kwargs)

The LongT5 model was proposed in `LongT5: Efficient Text-To-Text Transformer for Long Sequences <https://arxiv.org/abs/2112.07916>`__
by Mandy Guo, Joshua Ainslie, David Uthus, Santiago Ontanon, Jianmo Ni, Yun-Hsuan Sung and Yinfei Yang. It's an
encoder-decoder transformer pre-trained in a text-to-text denoising generative setting. LongT5 model is an extension of
T5 model, and it enables using one of the two different efficient attention mechanisms - (1) Local attention, or (2)
Transient-Global attention.


The abstract from the paper is the following:

*Recent work has shown that either (1) increasing the input length or (2) increasing model size can improve the
performance of Transformer-based neural models. In this paper, we present a new model, called LongT5, with which we
explore the effects of scaling both the input length and model size at the same time. Specifically, we integrated
attention ideas from long-input transformers (ETC), and adopted pre-training strategies from summarization pre-training
(PEGASUS) into the scalable T5 architecture. The result is a new attention mechanism we call {\em Transient Global}
(TGlobal), which mimics ETC's local/global attention mechanism, but without requiring additional side-inputs. We are
able to achieve state-of-the-art results on several summarization tasks and outperform the original T5 models on
question answering tasks.*

Tips:

- ``LongT5ForConditionalGeneration`` is an extension of ``T5ForConditionalGeneration`` exchanging the traditional
encoder *self-attention* layer with efficient either *local* attention or *transient-global* (*tglobal*) attention.
- Unlike the T5 model, LongT5 does not use a task prefix. Furthermore, it uses a different pre-training objective
inspired by the pre-training of ``PegasusForConditionalGeneration``.
- LongT5 model is designed to work efficiently and very well on long-range *sequence-to-sequence* tasks where the
input sequence exceeds commonly used 512 tokens. It is capable of handling input sequences of a length up to 16,384 tokens.
- For *Local Attention*, the sparse sliding-window local attention operation allows a given token to attend only `r`
tokens to the left and right of it (with `r=127` by default). *Local Attention* does not introduce any new parameters
to the model. The complexity of the mechanism is linear in input sequence length `l`: `O(l*r)`.
- *Transient Global Attention* is an extension of the *Local Attention*. It, furthermore, allows each input token to
interact with all other tokens in the layer. This is achieved via splitting an input sequence into blocks of a fixed
length `k` (with a default `k=16`). Then, a global token for such a block is obtained via summing and normalizing the embeddings of every token
in the block. Thanks to this, the attention allows each token to attend to both nearby tokens like in Local attention, and
also every global token like in the case of standard global attention (*transient* represents the fact the global tokens
are constructed dynamically within each attention operation).  As a consequence, *TGlobal* attention introduces
a few new parameters -- global relative position biases and a layer normalization for global token's embedding.
The complexity of this mechanism is `O(l(r + l/k))`.
- An example showing how to evaluate a fine-tuned LongT5 model on the `pubmed dataset <https://huggingface.co/datasets/scientific_papers>`__ is below.

::

>>> import evaluate
>>> from datasets import load_dataset
>>> from transformers import AutoTokenizer, LongT5ForConditionalGeneration

>>> dataset = load_dataset("scientific_papers", "pubmed", split="validation")
>>> model = (
...     LongT5ForConditionalGeneration.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")
...     .to("cuda")
...     .half()
... )
>>> tokenizer = AutoTokenizer.from_pretrained("Stancld/longt5-tglobal-large-16384-pubmed-3k_steps")


>>> def generate_answers(batch):
...     inputs_dict = tokenizer(
...         batch`"article"], max_length=16384, padding="max_length", truncation=True, return_tensors="pt"
...     )
...     input_ids = inputs_dict.input_ids.to("cuda")
...     attention_mask = inputs_dict.attention_mask.to("cuda")
...     output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=512, num_beams=2)
...     batch["predicted_abstract"] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
...     return batch


>>> result = dataset.map(generate_answer, batched=True, batch_size=2)
>>> rouge = evaluate.load("rouge")
>>> rouge.compute(predictions=result["predicted_abstract"], references=result["abstract"])

This model was contributed by [stancld <https://huggingface.co/stancld>`__.
The original code can be found `here <https://github.com/google-research/longt5>`__.

Arguments:
    vocab_size (`int`, *optional*, defaults to 32128):
        Vocabulary size of the LongT5 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``LongT5Model``.
    d_model (`int`, *optional*, defaults to 512):
        Size of the encoder layers and the pooler layer.
    d_kv (`int`, *optional*, defaults to 64):
        Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
        num_heads`.
    d_ff (`int`, *optional*, defaults to 2048):
        Size of the intermediate feed forward layer in each `LongT5Block`.
    num_layers (`int`, *optional*, defaults to 6):
        Number of hidden layers in the Transformer encoder.
    num_decoder_layers (`int`, *optional*):
        Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
    num_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the Transformer encoder.
    local_radius (`int`, *optional*, defaults to 127)
        Number of tokens to the left/right for each token to locally self-attend in a local attention mechanism.
    global_block_size (`int`, *optional*, defaults to 16)
        Lenght of blocks an input sequence is divided into for a global token representation. Used only for
        `encoder_attention_type = "transient-global"`.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    dropout_rate (`float`, *optional*, defaults to 0.1):
        The ratio for all dropout layers.
    layer_norm_eps (`float`, *optional*, defaults to 1e-6):
        The epsilon used by the layer normalization layers.
    initializer_factor (`float`, *optional*, defaults to 1):
        A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
        testing).
    feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. LongT5v1.1 uses the
        `"gated-gelu"` feed forward projection. Original LongT5 implementation uses `"gated-gelu"`.
    encoder_attention_type (`string`, *optional*, defaults to `"local"`):
        Type of encoder attention to be used. Should be one of `"local"` or `"transient-global"`, which are
        supported by LongT5 implementation.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.luke.configuration_luke.LukeConfig(vocab_size=50267, entity_vocab_size=500000, hidden_size=768, entity_emb_size=256, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, use_entity_aware_attention=True, classifier_dropout=None, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The LUKE model was proposed in `LUKE: Deep Contextualized Entity Representations with Entity-aware Self-attention <https://arxiv.org/abs/2010.01057>`__ by Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda and Yuji Matsumoto.
It is based on RoBERTa and adds entity embeddings as well as an entity-aware self-attention mechanism, which helps
improve performance on various downstream tasks involving reasoning about entities such as named entity recognition,
extractive and cloze-style question answering, entity typing, and relation classification.

The abstract from the paper is the following:

*Entity representations are useful in natural language tasks involving entities. In this paper, we propose new
pretrained contextualized representations of words and entities based on the bidirectional transformer. The proposed
model treats words and entities in a given text as independent tokens, and outputs contextualized representations of
them. Our model is trained using a new pretraining task based on the masked language model of BERT. The task involves
predicting randomly masked words and entities in a large entity-annotated corpus retrieved from Wikipedia. We also
propose an entity-aware self-attention mechanism that is an extension of the self-attention mechanism of the
transformer, and considers the types of tokens (words or entities) when computing attention scores. The proposed model
achieves impressive empirical performance on a wide range of entity-related tasks. In particular, it obtains
state-of-the-art results on five well-known datasets: Open Entity (entity typing), TACRED (relation classification),
CoNLL-2003 (named entity recognition), ReCoRD (cloze-style question answering), and SQuAD 1.1 (extractive question
answering).*

Tips:

- This implementation is the same as ``RobertaModel`` with the addition of entity embeddings as well
  as an entity-aware self-attention mechanism, which improves performance on tasks involving reasoning about entities.
- LUKE treats entities as input tokens; therefore, it takes `entity_ids`, `entity_attention_mask`,
  `entity_token_type_ids` and `entity_position_ids` as extra input. You can obtain those using
  ``LukeTokenizer``.
- ``LukeTokenizer`` takes `entities` and `entity_spans` (character-based start and end
  positions of the entities in the input text) as extra input. `entities` typically consist of `MASK] entities or
  Wikipedia entities. The brief description when inputting these entities are as follows:

  - *Inputting [MASK] entities to compute entity representations*: The [MASK] entity is used to mask entities to be
    predicted during pretraining. When LUKE receives the [MASK] entity, it tries to predict the original entity by
    gathering the information about the entity from the input text. Therefore, the [MASK] entity can be used to address
    downstream tasks requiring the information of entities in text such as entity typing, relation classification, and
    named entity recognition.
  - *Inputting Wikipedia entities to compute knowledge-enhanced token representations*: LUKE learns rich information
    (or knowledge) about Wikipedia entities during pretraining and stores the information in its entity embedding. By
    using Wikipedia entities as input tokens, LUKE outputs token representations enriched by the information stored in
    the embeddings of these entities. This is particularly effective for tasks requiring real-world knowledge, such as
    question answering.

- There are three head models for the former use case:

  - ``LukeForEntityClassification``, for tasks to classify a single entity in an input text such as
    entity typing, e.g. the [Open Entity dataset <https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html>`__.
    This model places a linear head on top of the output entity representation.
  - ``LukeForEntityPairClassification``, for tasks to classify the relationship between two entities
    such as relation classification, e.g. the `TACRED dataset <https://nlp.stanford.edu/projects/tacred/>`__. This
    model places a linear head on top of the concatenated output representation of the pair of given entities.
  - ``LukeForEntitySpanClassification``, for tasks to classify the sequence of entity spans, such as
    named entity recognition (NER). This model places a linear head on top of the output entity representations. You
    can address NER using this model by inputting all possible entity spans in the text to the model.

  ``LukeTokenizer`` has a `task` argument, which enables you to easily create an input to these
  head models by specifying `task="entity_classification"`, `task="entity_pair_classification"`, or
  `task="entity_span_classification"`. Please refer to the example code of each head models.

  A demo notebook on how to fine-tune ``LukeForEntityPairClassification`` for relation
  classification can be found `here <https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LUKE>`__.

  There are also 3 notebooks available, which showcase how you can reproduce the results as reported in the paper with
  the HuggingFace implementation of LUKE. They can be found `here <https://github.com/studio-ousia/luke/tree/master/notebooks>`__.

.. class:: transformers.models.m2m_100.configuration_m2m_100.M2M100Config(vocab_size=128112, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.05, decoder_layerdrop=0.05, use_cache=True, is_encoder_decoder=True, activation_function='relu', d_model=1024, dropout=0.1, attention_dropout=0.1, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=2, scale_embedding=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The M2M100 model was proposed in `Beyond English-Centric Multilingual Machine Translation <https://arxiv.org/abs/2010.11125>`__ by Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky,
Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy
Liptchinsky, Sergey Edunov, Edouard Grave, Michael Auli, Armand Joulin.

The abstract from the paper is the following:

*Existing work in translation demonstrated the potential of massively multilingual machine translation by training a
single model able to translate between any pair of languages. However, much of this work is English-Centric by training
only on data which was translated from or to English. While this is supported by large sources of training data, it
does not reflect translation needs worldwide. In this work, we create a true Many-to-Many multilingual translation
model that can translate directly between any pair of 100 languages. We build and open source a training dataset that
covers thousands of language directions with supervised data, created through large-scale mining. Then, we explore how
to effectively increase model capacity through a combination of dense scaling and language-specific sparse parameters
to create high quality models. Our focus on non-English-Centric models brings gains of more than 10 BLEU when directly
translating between non-English directions while performing competitively to the best single systems of WMT. We
open-source our scripts so that others may reproduce the data, evaluation, and final M2M-100 model.*

This model was contributed by `valhalla <https://huggingface.co/valhalla>`__.


#Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the M2M100 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``M2M100Model`` or
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.mamba.configuration_mamba.MambaConfig(vocab_size=50280, hidden_size=768, state_size=16, num_hidden_layers=32, layer_norm_epsilon=1e-05, pad_token_id=0, bos_token_id=0, eos_token_id=0, expand=2, conv_kernel=4, use_bias=False, use_conv_bias=True, hidden_act='silu', initializer_range=0.1, residual_in_fp32=True, time_step_rank='auto', time_step_scale=1.0, time_step_min=0.001, time_step_max=0.1, time_step_init_scheme='random', time_step_floor=0.0001, rescale_prenorm_residual=False, use_cache=True, **kwargs)

The Mamba model was proposed in `Mamba: Linear-Time Sequence Modeling with Selective State Spaces <https://arxiv.org/abs/2312.00752>`__ by Albert Gu and Tri Dao.

This model is a new paradigm architecture based on `state-space-models`. You can read more about the intuition behind these `here <https://srush.github.io/annotated-s4/>`__.

The abstract from the paper is the following:

*Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5× higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.*

Tips:

- Mamba is a new `state space model` architecture that rivals the classic Transformers. It is based on the line of progress on structured state space models, with an efficient hardware-aware design and implementation in the spirit of `FlashAttention <https://github.com/Dao-AILab/flash-attention>`__.
- Mamba stacks `mixer` layers, which are the equivalent of `Attention` layers. The core logic of `mamba` is held in the `MambaMixer` class.
- Two implementations cohabit: one is optimized and uses fast cuda kernels, while the other one is naive but can run on any device!
- The current implementation leverages the original cuda kernels: the equivalent of flash attention for Mamba are hosted in the ``mamba-ssm``(https://github.com/state-spaces/mamba) and the ``causal_conv1d``(https://github.com/Dao-AILab/causal-conv1d) repositories. Make sure to install them if your hardware supports them!
- Contributions to make the naive path faster are welcome 🤗

This model was contributed by `ArthurZ <https://huggingface.co/ArthurZ>`__.
The original code can be found `here <https://github.com/state-spaces/mamba>`__.

# Usage

#Args:
    vocab_size (`int`, *optional*, defaults to 50280):
        Vocabulary size of the MAMBA model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MambaModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the embeddings and hidden states.
    state_size (`int`, *optional*, defaults to 16): shape of the state space latents.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the model.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    pad_token_id (`int`, *optional*, defaults to 0):
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 0):
        The id of the beginning of sentence token in the vocabulary.
    eos_token_id (`int`, *optional*, defaults to 0):
        The id of the end of sentence token in the vocabulary.
    expand (`int`, *optional*, defaults to 2): Expanding factor used to determine the intermediate size.
    conv_kernel (`int`, *optional*, defaults to 4): Size of the convolution kernel.
    use_bias (`bool`, *optional*, defaults to `False`):
        Whether or not to use bias in ["in_proj", "out_proj"] of the mixer block
    use_conv_bias (`bool`, *optional*, defaults to `True`):
        Whether or not to use bias in the convolution layer of the mixer block.
    hidden_act (`str`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the decoder.
    initializer_range (`float`, *optional*, defaults to 0.1):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    residual_in_fp32 (`bool`, *optional*, defaults to `True`):
        Whether or not residuals should be in `float32`. If set to `False` residuals will keep the same `dtype` as the rest of the model
    time_step_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
        Rank of the the discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
    time_step_scale (`float`, *optional*, defaults to 1.0):
        Scale used used to scale `dt_proj.bias`.
    time_step_min (`float`, *optional*, defaults to 0.001):
        Minimum `time_step` used to bound `dt_proj.bias`.
    time_step_max (`float`, *optional*, defaults to 0.1):
        Maximum `time_step` used to bound `dt_proj.bias`.
    time_step_init_scheme (`float`, *optional*, defaults to `"random"`):
        Init scheme used for `dt_proj.weight`. Should be one of `["random","uniform"]`
    time_step_floor (`float`, *optional*, defaults to 0.0001):
        Minimum clamping value of the `dt_proj.bias` layer initialization.
    rescale_prenorm_residual (`bool`, *optional*, defaults to `False`):
        Whether or not to rescale `out_proj` weights when initializing.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the cache should be used.

.. class:: transformers.models.marian.configuration_marian.MarianConfig(vocab_size=58101, decoder_vocab_size=None, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=58100, scale_embedding=False, pad_token_id=58100, eos_token_id=0, forced_eos_token_id=0, share_encoder_decoder_embeddings=True, **kwargs)

A framework for translation models, using the same models as BART. Translations should be similar, but not identical to output in the test set linked to in each model card.
This model was contributed by `sshleifer <https://huggingface.co/sshleifer>`__.


Args:
    vocab_size (`int`, *optional*, defaults to 58101):
        Vocabulary size of the Marian model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MarianModel`` or ``TFMarianModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)
    forced_eos_token_id (`int`, *optional*, defaults to 0):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.

.. class:: transformers.models.markuplm.configuration_markuplm.MarkupLMConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, bos_token_id=0, eos_token_id=2, max_xpath_tag_unit_embeddings=256, max_xpath_subs_unit_embeddings=1024, tag_pad_id=216, subs_pad_id=1001, xpath_unit_hidden_size=32, max_depth=50, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The MarkupLM model was proposed in `MarkupLM: Pre-training of Text and Markup Language for Visually-rich Document
Understanding <https://arxiv.org/abs/2110.08518>`__ by Junlong Li, Yiheng Xu, Lei Cui, Furu Wei. MarkupLM is BERT, but
applied to HTML pages instead of raw text documents. The model incorporates additional embedding layers to improve
performance, similar to `LayoutLM <layoutlm>`__.

The model can be used for tasks like question answering on web pages or information extraction from web pages. It obtains
state-of-the-art results on 2 important benchmarks:
- `WebSRC <https://x-lance.github.io/WebSRC/>`__, a dataset for Web-Based Structural Reading Comprehension (a bit like SQuAD but for web pages)
- `SWDE <https://www.researchgate.net/publication/221299838_From_one_tree_to_a_forest_a_unified_solution_for_structured_web_data_extraction>`__, a dataset
for information extraction from web pages (basically named-entity recogntion on web pages)

The abstract from the paper is the following:

*Multimodal pre-training with text, layout, and image has made significant progress for Visually-rich Document
Understanding (VrDU), especially the fixed-layout documents such as scanned document images. While, there are still a
large number of digital documents where the layout information is not fixed and needs to be interactively and
dynamically rendered for visualization, making existing layout-based pre-training approaches not easy to apply. In this
paper, we propose MarkupLM for document understanding tasks with markup languages as the backbone such as
HTML/XML-based documents, where text and markup information is jointly pre-trained. Experiment results show that the
pre-trained MarkupLM significantly outperforms the existing strong baseline models on several document understanding
tasks. The pre-trained model and code will be publicly available.*

Tips:
- In addition to `input_ids`, ``~MarkupLMModel.forward`` expects 2 additional inputs, namely `xpath_tags_seq` and `xpath_subs_seq`.
These are the XPATH tags and subscripts respectively for each token in the input sequence.
- One can use ``MarkupLMProcessor`` to prepare all data for the model. Refer to the `usage guide <#usage-markuplmprocessor>`__ for more info.
- Demo notebooks can be found `here <https://github.com/NielsRogge/Transformers-Tutorials/tree/master/MarkupLM>`__.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/markuplm_architecture.jpg"
alt="drawing" width="600"/> 

<small> MarkupLM architecture. Taken from the <a href="https://arxiv.org/abs/2110.08518">original paper.</a> </small>

This model was contributed by `nielsr <https://huggingface.co/nielsr>`__. The original code can be found `here <https://github.com/microsoft/unilm/tree/master/markuplm>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the MarkupLM model. Defines the different tokens that can be represented by the
        *inputs_ids* passed to the forward method of ``MarkupLMModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed into ``MarkupLMModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    max_tree_id_unit_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the tree id unit embedding might ever use. Typically set this to something large
        just in case (e.g., 1024).
    max_xpath_tag_unit_embeddings (`int`, *optional*, defaults to 256):
        The maximum value that the xpath tag unit embedding might ever use. Typically set this to something large
        just in case (e.g., 256).
    max_xpath_subs_unit_embeddings (`int`, *optional*, defaults to 1024):
        The maximum value that the xpath subscript unit embedding might ever use. Typically set this to something
        large just in case (e.g., 1024).
    tag_pad_id (`int`, *optional*, defaults to 216):
        The id of the padding token in the xpath tags.
    subs_pad_id (`int`, *optional*, defaults to 1001):
        The id of the padding token in the xpath subscripts.
    xpath_tag_unit_hidden_size (`int`, *optional*, defaults to 32):
        The hidden size of each tree id unit. One complete tree index will have
        (50*xpath_tag_unit_hidden_size)-dim.
    max_depth (`int`, *optional*, defaults to 50):
        The maximum depth in xpath.

.. class:: transformers.models.mbart.configuration_mbart.MBartConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False, pad_token_id=1, bos_token_id=0, eos_token_id=2, forced_eos_token_id=2, **kwargs)

of MBart

The MBart model was presented in `Multilingual Denoising Pre-training for Neural Machine Translation <https://arxiv.org/abs/2001.08210>`__ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov Marjan
Ghazvininejad, Mike Lewis, Luke Zettlemoyer.

According to the abstract, MBART is a sequence-to-sequence denoising auto-encoder pretrained on large-scale monolingual
corpora in many languages using the BART objective. mBART is one of the first methods for pretraining a complete
sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only
on the encoder, decoder, or reconstructing parts of the text.

This model was contributed by `valhalla <https://huggingface.co/valhalla>`__. The Authors' code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/mbart>`__

#Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the MBART model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MBartModel`` or ``TFMBartModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)
    forced_eos_token_id (`int`, *optional*, defaults to 2):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.

.. class:: transformers.models.mega.configuration_mega.MegaConfig(vocab_size=30522, hidden_size=128, num_hidden_layers=4, intermediate_size=256, ema_projection_size=16, bidirectional=True, shared_representation_size=64, use_chunking=False, chunk_size=-1, truncation=None, normalize_before_mega=True, normalization_type='scalenorm', norm_affine=True, activation='silu', attention_activation='softmax', dropout_prob=0.1, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, use_feature_dropout=False, use_normalized_ffn=True, nffn_hidden_size=256, normalize_before_ffn=True, nffn_activation_dropout_prob=0.1, max_positions=2048, add_token_type_embeddings=False, type_vocab_size=2, initializer_range=0.02, ema_delta_alpha_range=0.2, ema_beta_range=0.02, ema_gamma_omega_range=1.0, pad_token_id=1, bos_token_id=0, eos_token_id=2, relative_positional_bias='rotary', classifier_dropout=None, use_cache=True, add_lm_hidden_dense_layer=True, **kwargs)

The MEGA model was proposed in `Mega: Moving Average Equipped Gated Attention <https://arxiv.org/abs/2209.10655>`__ by Xuezhe Ma, Chunting Zhou, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, and Luke Zettlemoyer.
MEGA proposes a new approach to self-attention with each encoder layer having a multi-headed exponential moving average in addition to a single head of standard dot-product attention, giving the attention mechanism 
stronger positional biases. This allows MEGA to perform competitively to Transformers on standard benchmarks including LRA 
while also having significantly fewer parameters. MEGA's compute efficiency allows it to scale to very long sequences, making it an 
attractive option for long-document NLP tasks.

The abstract from the paper is the following:

 *The design choices in the Transformer attention mechanism, including weak inductive bias and quadratic computational complexity, have limited its application for modeling long sequences. In this paper, we introduce Mega, a simple, theoretically grounded, single-head gated attention mechanism equipped with (exponential) moving average to incorporate inductive bias of position-aware local dependencies into the position-agnostic attention mechanism. We further propose a variant of Mega that offers linear time and space complexity yet yields only minimal quality loss, by efficiently splitting the whole sequence into multiple chunks with fixed length. Extensive experiments on a wide range of sequence modeling benchmarks, including the Long Range Arena, neural machine translation, auto-regressive language modeling, and image and speech classification, show that Mega achieves significant improvements over other sequence models, including variants of Transformers and recent state space models. *

Tips:

- MEGA can perform quite well with relatively few parameters. See Appendix D in the MEGA paper for examples of architectural specs which perform well in various settings. If using MEGA as a decoder, be sure to set `bidirectional=False` to avoid errors with default bidirectional. 
- Mega-chunk is a variant of mega that reduces time and spaces complexity from quadratic to linear. Utilize chunking with MegaConfig.use_chunking and control chunk size with MegaConfig.chunk_size 

This model was contributed by `mnaylor <https://huggingface.co/mnaylor>`__.
The original code can be found `here <https://github.com/facebookresearch/mega>`__.

Implementation Notes:

- The original implementation of MEGA had an inconsistent expectation of attention masks for padding and causal self-attention between the softmax attention and Laplace/squared ReLU method. This implementation addresses that inconsistency.
- The original implementation did not include token type embeddings; this implementation adds support for these, with the option controlled by MegaConfig.add_token_type_embeddings


Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the Mega model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MegaModel``.
    hidden_size (`int`, *optional*, defaults to 128):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of hidden layers in the Mega encoder.
    intermediate_size (`int`, *optional*, defaults to 256):
        Dimensionality of the hidden size (self-attention value projection) within the Mega encoder
    ema_projection_size (`int`, *optional*, defaults to 16):
        Dimensionality of the MegaMultiDimensionDampedEma
    bidirectional (`bool`, *optional*, defaults to `True`):
        Whether the MegaMultiDimensionDampedEma used in Mega's self-attention should work bidirectionally (`True`)
        or unidirectionally (`False`). Bidirectional EMA is incompatible with causal decoding, so this should be
        False if you intend to use the model as a decoder.
    shared_representation_size (`int`, *optional*, defaults to 64):
        Dimensionality of the linear projection for shared representation of self-attention queries and keys
    use_chunking (`bool`, *optional*, defaults to `False`):
        Whether to chunk inputs for linear self-attention complexity (described as Mega-chunk in the paper)
    chunk_size (`int`, *optional*, defaults to -1):
        If `use_chunking` is set to `True`, determines the size of the chunks to apply to the input sequence. If
        chunking is used, input sequences must be padded to a multiple of `chunk_size`
    truncation (`int`, *optional*):
        If specified, the sequence length for which to truncate MegaMultiDimensionDampedEma
    normalize_before_mega (`bool`, *optional*, defaults to `True`):
        Whether to normalize before (`True`) or after (`False`) passing through Mega encoder blocks
    normalization_type (`str`, *optional*, defaults to `"scalenorm"`):
        Type of normalization to use in Mega encoder blocks. Choose one of `"scalenorm"`, `"layernorm"`,
        `"rmsnorm"`, `"batchnorm"`, or `"syncbatchnorm"` (GPU required for syncbatchnorm)
    norm_affine (`bool`, *optional*, defaults to `True`):
        If `True`, applies a parameterized affine transformation to inputs during normalization
    activation (`str`, *optional*, defaults to `"silu"`):
        Activation function to apply within Mega encoder blocks. Choose one of `"silu"`, `"relu"`, `"linear"`,
        `"gelu"`, or `"gelu_accurate"`
    attention_activation (`str`, *optional*, defaults to `"softmax"`):
        Activation function to apply for single-headed self-attention (a la Transformer). Choose one of
        `"softmax"`, `"laplace"`, or `"relu2"`
    dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for EMA self-attention
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    use_feature_dropout (`bool`, *optional*, defaults to `False`):
        Whether to use feature-based (`True`) or standard dropout (`False`)
    use_normalized_ffn (`bool`, *optional*, defaults to `True`):
        Whether to use the normalized feed-forward sub-layer in Mega blocks (`True`) or pass Mega encoder output
        as-is (`False`)
    nffn_hidden_size (`int`, *optional*, defaults to 256):
        If using the normalized feed-forward network (NFFN) layer within Mega (`use_normalized_ffn = True`), this
        is the hidden size of the NFFN
    normalize_before_ffn (`bool`, *optional*, defaults to `True`):
        Whether to normalize before (`True`) or after (`False`) the feed-forward portion of NFFN
    nffn_activation_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the NFFN component.
    max_positions (`int`, *optional*, defaults to 2048):
        The maximum sequence length to use for positional representations. For `"simple"` relative positional bias,
        this is a hard limit on input length; `"rotary"` relative positional bias will extrapolate to longer
        sequences
    add_token_type_embeddings (`bool`, *optional*, defaults to `True`):
        Whether to account for token types in embeddings. Left as optional to maintain compatibility with original
        implementation while adding support for token types.
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``MegaModel``. Only used if
        `add_token_type_embeddings = True`
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    ema_delta_alpha_range (`float`, *optional*, defaults to 0.2):
        The standard deviation for initializing the delta (damping factor) and alpha (decay factor) parameters in
        MegaMultiDimensionDampedEma.
    ema_beta_range (`float`, *optional*, defaults to 0.02):
        The standard deviation for initializing the beta parameter (expansion matrix) in
        MegaMultiDimensionDampedEma.
    ema_gamma_omega_range (`float`, *optional*, defaults to 1.0):
        The standard deviation for initializing the gamma (projection matrix) and omega (residual weight)
        parameters in MultiDimensionEMA.
    relative_positional_bias (`str`, *optional*, defaults to `"rotary"`):
        Type of relative positional encoding. Choose one of `"rotary"` or `"simple"`. If `"simple"` is selected,
        `max_positions` is used as a limit on input size, while `"rotary"` extrapolates beyond `max_positions`.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.
    add_lm_hidden_dense_layer (`bool`, *optional*, defaults to `True`):
        Whether to include a hidden layer for projection between encoder outputs and LM heads (`True`) or pass
        hidden states directly to LM head (`False`). Remains optional for compatibility with original
        implementation

.. class:: transformers.models.megatron_bert.configuration_megatron_bert.MegatronBertConfig(vocab_size=29056, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type='absolute', use_cache=True, **kwargs)

The MegatronBERT model was proposed in `Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
Parallelism <https://arxiv.org/abs/1909.08053>`__ by Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley,
Jared Casper and Bryan Catanzaro.

The abstract from the paper is the following:

*Recent work in language modeling demonstrates that training large transformer models advances the state of the art in
Natural Language Processing applications. However, very large models can be quite difficult to train due to memory
constraints. In this work, we present our techniques for training very large transformer models and implement a simple,
efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our
approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model
parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We
illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain
15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline
that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance
the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9
billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in
BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we
achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA
accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy
of 89.4%).*

Tips:

We have provided pretrained `BERT-345M <https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m>`__ checkpoints
for use to evaluate or finetuning downstream tasks.

To access these checkpoints, first `sign up <https://ngc.nvidia.com/signup>`__ for and setup the NVIDIA GPU Cloud (NGC)
Registry CLI. Further documentation for downloading models can be found in the `NGC documentation <https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1>`__.

Alternatively, you can directly download the checkpoints using:

BERT-345M-uncased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip
-O megatron_bert_345m_v0_1_uncased.zip
```

BERT-345M-cased:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O
megatron_bert_345m_v0_1_cased.zip
```

Once you have obtained the checkpoints from NVIDIA GPU Cloud (NGC), you have to convert them to a format that will
easily be loaded by Hugging Face Transformers and our port of the BERT code.

The following commands allow you to do the conversion. We assume that the folder `models/megatron_bert` contains
`megatron_bert_345m_v0_1_{cased, uncased}.zip` and that the commands are run from inside that folder:

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_uncased.zip
```

```bash
python3 $PATH_TO_TRANSFORMERS/models/megatron_bert/convert_megatron_bert_checkpoint.py megatron_bert_345m_v0_1_cased.zip
```

This model was contributed by `jdemouth <https://huggingface.co/jdemouth>`__. The original code can be found `here <https://github.com/NVIDIA/Megatron-LM>`__. That repository contains a multi-GPU and multi-node implementation of the
Megatron Language models. In particular, it contains a hybrid model parallel approach using "tensor parallel" and
"pipeline parallel" techniques.

Args:
    vocab_size (`int`, *optional*, defaults to 29056):
        Vocabulary size of the MEGATRON_BERT model. Defines the number of different tokens that can be represented
        by the `inputs_ids` passed when calling ``MegatronBertModel``.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``MegatronBertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.

.. class:: transformers.models.mixtral.configuration_mixtral.MixtralConfig(vocab_size=32000, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8, hidden_act='silu', max_position_embeddings=131072, initializer_range=0.02, rms_norm_eps=1e-05, use_cache=True, pad_token_id=None, bos_token_id=1, eos_token_id=2, tie_word_embeddings=False, rope_theta=1000000.0, sliding_window=None, attention_dropout=0.0, num_experts_per_tok=2, num_local_experts=8, output_router_logits=False, router_aux_loss_coef=0.001, **kwargs)

Mixtral-8x7B is Mistral AI's second Large Language Model (LLM). 

The Mixtral model was proposed by the `Mistral AI <https://mistral.ai/>`__ team.

It was introduced in the `Mixtral of Experts blogpost <https://mistral.ai/news/mixtral-of-experts/>`__ with the following introduction:

*Today, the team is proud to release Mixtral 8x7B, a high-quality sparse mixture of experts models (SMoE) with open weights. Licensed under Apache 2.0. Mixtral outperforms Llama 2 70B on most benchmarks with 6x faster inference. It is the strongest open-weight model with a permissive license and the best model overall regarding cost/performance trade-offs. In particular, it matches or outperforms GPT3.5 on most standard benchmarks.*

Tips:


- The model needs to be converted using the `conversion script <https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/convert_mixtral_weights_to_hf.py>`__.
- If the model is quantized to 4bits, a single A100 is enough to fit the entire 45B model.

This model was contributed by `Younes Belkada <https://huggingface.co/ybelkada>`__ and `Arthur Zucker <https://huggingface.co/ArthurZ>`__ .
The original code can be found `here <https://github.com/mistralai/mistral-src>`__.


#Args:
    vocab_size (`int`, *optional*, defaults to 32000):
        Vocabulary size of the Mixtral model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MixtralModel``
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 14336):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer encoder.
    num_key_value_heads (`int`, *optional*, defaults to 8):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to `8`.
    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
        The maximum sequence length that this model might ever be used with. Mixtral's sliding window attention
        allows sequence of up to 4096*32 tokens.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    rms_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the rms normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    pad_token_id (`int`, *optional*):
        The id of the padding token.
    bos_token_id (`int`, *optional*, defaults to 1):
        The id of the "beginning-of-sequence" token.
    eos_token_id (`int`, *optional*, defaults to 2):
        The id of the "end-of-sequence" token.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether the model's input and output word embeddings should be tied.
    rope_theta (`float`, *optional*, defaults to 1000000.0):
        The base period of the RoPE embeddings.
    sliding_window (`int`, *optional*):
        Sliding window attention window size. If not specified, will default to `4096`.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    num_experts_per_tok (`int`, *optional*, defaults to 2):
        The number of experts to root per-token, can be also interpreted as the `top-p` routing
        parameter
    num_local_experts (`int`, *optional*, defaults to 8):
        Number of experts per Sparse MLP layer.
    output_router_logits (`bool`, *optional*, defaults to `False`):
        Whether or not the router logits should be returned by the model. Enabeling this will also
        allow the model to output the auxiliary loss. See `here <>`__ for more details
    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
        The aux loss factor for the total loss.

::

>>> from transformers import MixtralModel, MixtralConfig

>>> # Initializing a Mixtral 7B style configuration
>>> configuration = MixtralConfig()

>>> # Initializing a model from the Mixtral 7B style configuration
>>> model = MixtralModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

.. class:: transformers.models.mobilebert.configuration_mobilebert.MobileBertConfig(vocab_size=30522, hidden_size=512, num_hidden_layers=24, num_attention_heads=4, intermediate_size=512, hidden_act='relu', hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, embedding_size=128, trigram_input=True, use_bottleneck=True, intra_bottleneck_size=128, use_bottleneck_attention=False, key_query_shared_bottleneck=True, num_feedforward_networks=4, normalization_type='no_norm', classifier_activation=True, classifier_dropout=None, **kwargs)

The MobileBERT model was proposed in `MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices <https://arxiv.org/abs/2004.02984>`__ by Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, and Denny
Zhou. It's a bidirectional transformer based on the BERT model, which is compressed and accelerated using several
approaches.

The abstract from the paper is the following:

*Natural Language Processing (NLP) has recently achieved great success by using huge pre-trained models with hundreds
of millions of parameters. However, these models suffer from heavy model sizes and high latency such that they cannot
be deployed to resource-limited mobile devices. In this paper, we propose MobileBERT for compressing and accelerating
the popular BERT model. Like the original BERT, MobileBERT is task-agnostic, that is, it can be generically applied to
various downstream NLP tasks via simple fine-tuning. Basically, MobileBERT is a thin version of BERT_LARGE, while
equipped with bottleneck structures and a carefully designed balance between self-attentions and feed-forward networks.
To train MobileBERT, we first train a specially designed teacher model, an inverted-bottleneck incorporated BERT_LARGE
model. Then, we conduct knowledge transfer from this teacher to MobileBERT. Empirical studies show that MobileBERT is
4.3x smaller and 5.5x faster than BERT_BASE while achieving competitive results on well-known benchmarks. On the
natural language inference tasks of GLUE, MobileBERT achieves a GLUEscore o 77.7 (0.6 lower than BERT_BASE), and 62 ms
latency on a Pixel 4 phone. On the SQuAD v1.1/v2.0 question answering task, MobileBERT achieves a dev F1 score of
90.0/79.2 (1.5/2.1 higher than BERT_BASE).*

Tips:

- MobileBERT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather
  than the left.
- MobileBERT is similar to BERT and therefore relies on the masked language modeling (MLM) objective. It is therefore
  efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation. Models trained
  with a causal language modeling (CLM) objective are better in that regard.

This model was contributed by `vshampor <https://huggingface.co/vshampor>`__. The original code can be found `here <https://github.com/google-research/mobilebert>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the MobileBERT model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``MobileBertModel`` or ``TFMobileBertModel``.
    hidden_size (`int`, *optional*, defaults to 512):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 4):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 512):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"relu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``MobileBertModel`` or
        ``TFMobileBertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.

    pad_token_id (`int`, *optional*, defaults to 0):
        The ID of the token in the word embedding to use as padding.
    embedding_size (`int`, *optional*, defaults to 128):
        The dimension of the word embedding vectors.
    trigram_input (`bool`, *optional*, defaults to `True`):
        Use a convolution of trigram as input.
    use_bottleneck (`bool`, *optional*, defaults to `True`):
        Whether to use bottleneck in BERT.
    intra_bottleneck_size (`int`, *optional*, defaults to 128):
        Size of bottleneck layer output.
    use_bottleneck_attention (`bool`, *optional*, defaults to `False`):
        Whether to use attention inputs from the bottleneck transformation.
    key_query_shared_bottleneck (`bool`, *optional*, defaults to `True`):
        Whether to use the same linear transformation for query&key in the bottleneck.
    num_feedforward_networks (`int`, *optional*, defaults to 4):
        Number of FFNs in a block.
    normalization_type (`str`, *optional*, defaults to `"no_norm"`):
        The normalization type in MobileBERT.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.mpnet.configuration_mpnet.MPNetConfig(vocab_size=30527, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, initializer_range=0.02, layer_norm_eps=1e-12, relative_attention_num_buckets=32, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The MPNet model was proposed in `MPNet: Masked and Permuted Pre-training for Language Understanding <https://arxiv.org/abs/2004.09297>`__ by Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, Tie-Yan Liu.

MPNet adopts a novel pre-training method, named masked and permuted language modeling, to inherit the advantages of
masked language modeling and permuted language modeling for natural language understanding.

The abstract from the paper is the following:

*BERT adopts masked language modeling (MLM) for pre-training and is one of the most successful pre-training models.
Since BERT neglects dependency among predicted tokens, XLNet introduces permuted language modeling (PLM) for
pre-training to address this problem. However, XLNet does not leverage the full position information of a sentence and
thus suffers from position discrepancy between pre-training and fine-tuning. In this paper, we propose MPNet, a novel
pre-training method that inherits the advantages of BERT and XLNet and avoids their limitations. MPNet leverages the
dependency among predicted tokens through permuted language modeling (vs. MLM in BERT), and takes auxiliary position
information as input to make the model see a full sentence and thus reducing the position discrepancy (vs. PLM in
XLNet). We pre-train MPNet on a large-scale dataset (over 160GB text corpora) and fine-tune on a variety of
down-streaming tasks (GLUE, SQuAD, etc). Experimental results show that MPNet outperforms MLM and PLM by a large
margin, and achieves better results on these tasks compared with previous state-of-the-art pre-trained methods (e.g.,
BERT, XLNet, RoBERTa) under the same model setting.*

Tips:

- MPNet doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. just
  separate your segments with the separation token `tokenizer.sep_token` (or ``sep]`).

The original code can be found [here <https://github.com/microsoft/MPNet>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30527):
        Vocabulary size of the MPNet model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MPNetModel`` or ``TFMPNetModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.

.. class:: transformers.models.mpt.configuration_mpt.MptConfig(d_model: int = 2048, n_heads: int = 16, n_layers: int = 24, expansion_ratio: int = 4, max_seq_len: int = 2048, vocab_size: int = 50368, resid_pdrop: float = 0.0, layer_norm_epsilon: float = 1e-05, emb_pdrop: float = 0.0, learned_pos_emb: bool = True, attn_config: transformers.models.mpt.configuration_mpt.MptAttentionConfig = None, init_device: str = 'cpu', logit_scale: Union[float, str, NoneType] = None, no_bias: bool = True, verbose: int = 0, embedding_fraction: float = 1.0, norm_type: str = 'low_precision_layernorm', use_cache: bool = False, initializer_range=0.02, **kwargs)

The MPT model was proposed by the `MosaicML <https://www.mosaicml.com/>`__ team and released with multiple sizes and finetuned variants. The MPT models is a series of open source and commercially usable LLMs pre-trained on 1T tokens. 

MPT models are GPT-style decoder-only transformers with several improvements: performance-optimized layer implementations, architecture changes that provide greater training stability, and the elimination of context length limits by replacing positional embeddings with ALiBi. 

- MPT base: MPT base pre-trained models on next token prediction 
- MPT instruct: MPT base models fine-tuned on instruction based tasks
- MPT storywriter: MPT base models fine-tuned for 2500 steps on 65k-token excerpts of fiction books contained in the books3 corpus, this enables the model to handle very long sequences

The original code is available at the  ``llm-foundry``(https://github.com/mosaicml/llm-foundry/tree/main) repository.

Read more about it `in the release blogpost <https://www.mosaicml.com/blog/mpt-7b>`__

Tips:

- Learn more about some techniques behind training of the model `in this section of llm-foundry repository <https://github.com/mosaicml/llm-foundry/blob/main/TUTORIAL.md#faqs>`__
- If you want to use the advanced version of the model (triton kernels, direct flash attention integration), you can still use the original model implementation by adding `trust_remote_code=True` when calling `from_pretrained`.

- `Fine-tuning Notebook <https://colab.research.google.com/drive/1HCpQkLL7UXW8xJUJJ29X7QAeNJKO0frZ?usp=sharing>`__ on how to fine-tune MPT-7B on a free Google Colab instance to turn the model into a Chatbot.


Args:
    d_model (`int`, *optional*, defaults to 2048):
        Dimensionality of the embeddings and hidden states.
    n_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    n_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer encoder.
    expansion_ratio (`int`, *optional*, defaults to 4):
        The ratio of the up/down scale in the MLP.
    max_seq_len (`int`, *optional*, defaults to 2048):
        The maximum sequence length of the model.
    vocab_size (`int`, *optional*, defaults to 50368):
        Vocabulary size of the Mpt model. Defines the maximum number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``MptModel``. Check `this
        discussion <https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a>`__ on how the
        `vocab_size` has been defined.
    resid_pdrop (`float`, *optional*, defaults to 0.0):
        The dropout probability applied to the attention output before combining with residual.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    emb_pdrop (`float`, *optional*, defaults to 0.0):
        The dropout probability for the embedding layer.
    learned_pos_emb (`bool`, *optional*, defaults to `True`):
        Whether to use learned positional embeddings.
    attn_config (`dict`, *optional*):
        A dictionary used to configure the model's attention module.
    init_device (`str`, *optional*, defaults to `"cpu"`):
        The device to use for parameter initialization. Defined for backward compatibility
    logit_scale (`float`, *optional*):
        If not None, scale the logits by this value.
    no_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in all linear layers.
    verbose (`int`, *optional*, defaults to 0):
        The verbosity level to use for logging. Used in the previous versions of MPT models for logging. This
        argument is deprecated.
    embedding_fraction (`float`, *optional*, defaults to 1.0):
        The fraction to scale the gradients of the embedding layer by.
    norm_type (`str`, *optional*, defaults to `"low_precision_layernorm"`):
        Type of layer norm to use. All MPT models uses the same layer norm implementation. Defined for backward
        compatibility.
    use_cache (`bool`, *optional*, defaults to `False`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

.. class:: transformers.models.mra.configuration_mra.MraConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=1, initializer_range=0.02, layer_norm_eps=1e-05, position_embedding_type='absolute', block_per_row=4, approx_mode='full', initial_prior_first_n_blocks=0, initial_prior_diagonal_n_blocks=0, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The MRA model was proposed in `Multi Resolution Analysis (MRA) for Approximate Self-Attention <https://arxiv.org/abs/2207.10284>`__ by Zhanpeng Zeng, Sourav Pal, Jeffery Kline, Glenn M Fung, and Vikas Singh.

The abstract from the paper is the following:

*Transformers have emerged as a preferred model for many tasks in natural langugage processing and vision. Recent efforts on training and deploying Transformers more efficiently have identified many strategies to approximate the self-attention matrix, a key module in a Transformer architecture. Effective ideas include various prespecified sparsity patterns, low-rank basis expansions and combinations thereof. In this paper, we revisit classical Multiresolution Analysis (MRA) concepts such as Wavelets, whose potential value in this setting remains underexplored thus far. We show that simple approximations based on empirical feedback and design choices informed by modern hardware and implementation challenges, eventually yield a MRA-based approach for self-attention with an excellent performance profile across most criteria of interest. We undertake an extensive set of experiments and demonstrate that this multi-resolution scheme outperforms most efficient self-attention proposals and is favorable for both short and long sequences. Code is available at https://github.com/mlpen/mra-attention.*

This model was contributed by `novice03 <https://huggingface.co/novice03>`__.
The original code can be found `here <https://github.com/mlpen/mra-attention>`__.


Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the Mra model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MraModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 1):
        The vocabulary size of the `token_type_ids` passed when calling ``MraModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`.
    block_per_row (`int`, *optional*, defaults to 4):
        Used to set the budget for the high resolution scale.
    approx_mode (`str`, *optional*, defaults to `"full"`):
        Controls whether both low and high resolution approximations are used. Set to `"full"` for both low and
        high resolution and `"sparse"` for only low resolution.
    initial_prior_first_n_blocks (`int`, *optional*, defaults to 0):
        The initial number of blocks for which high resolution is used.
    initial_prior_diagonal_n_blocks (`int`, *optional*, defaults to 0):
        The number of diagonal blocks for which high resolution is used.

.. class:: transformers.models.mvp.configuration_mvp.MvpConfig(vocab_size=50267, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=False, use_cache=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, is_encoder_decoder=True, decoder_start_token_id=2, forced_eos_token_id=2, use_prompt=False, prompt_length=100, prompt_mid_dim=800, **kwargs)

The MVP model was proposed in `MVP: Multi-task Supervised Pre-training for Natural Language Generation <https://arxiv.org/abs/2206.12131>`__ by Tianyi Tang, Junyi Li, Wayne Xin Zhao and Ji-Rong Wen.


According to the abstract,

- MVP follows a standard Transformer encoder-decoder architecture.
- MVP is supervised pre-trained using labeled datasets.
- MVP also has task-specific soft prompts to stimulate the model's capacity in performing a certain task.
- MVP is specially designed for natural language generation and can be adapted to a wide range of generation tasks, including but not limited to summarization, data-to-text generation, open-ended dialogue system, story generation, question answering, question generation, task-oriented dialogue system, commonsense generation, paraphrase generation, text style transfer, and text simplification. Our model can also be adapted to natural language understanding tasks such as sequence classification and (extractive) question answering.

Tips:
- We have released a series of models `here <https://huggingface.co/models?filter=mvp>`__, including MVP, MVP with task-specific prompts, and multi-task pre-trained variants.
- If you want to use a model without prompts (standard Transformer), you can load it through `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp')`.
- If you want to use a model with task-specific prompts, such as summarization, you can load it through `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp-summarization')`.
- Our model supports lightweight prompt tuning following `Prefix-tuning <https://arxiv.org/abs/2101.00190>`__ with method `set_lightweight_tuning()`.

This model was contributed by `Tianyi Tang <https://huggingface.co/StevenTang>`__. The detailed information and instructions can be found `here <https://github.com/RUCAIBox/MVP>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50267):
        Vocabulary size of the MVP model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``MvpModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    forced_eos_token_id (`int`, *optional*, defaults to 2):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.
    use_prompt (`bool`, *optional*, defaults to `False`):
        Whether or not to use prompt.
    prompt_length (`int`, *optional*, defaults to 100):
        The length of prompt.
    prompt_mid_dim (`int`, *optional*, defaults to 800):
        Dimensionality of the "intermediate" layer in prompt.

.. class:: transformers.models.nezha.configuration_nezha.NezhaConfig(vocab_size=21128, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, max_relative_position=64, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, classifier_dropout=0.1, pad_token_id=0, bos_token_id=2, eos_token_id=3, use_cache=True, **kwargs)

The Nezha model was proposed in `NEZHA: Neural Contextualized Representation for Chinese Language Understanding <https://arxiv.org/abs/1909.00204>`__ by Junqiu Wei et al.

The abstract from the paper is the following:

*The pre-trained language models have achieved great successes in various natural language understanding (NLU) tasks
due to its capacity to capture the deep contextualized information in text by pre-training on large-scale corpora.
In this technical report, we present our practice of pre-training language models named NEZHA (NEural contextualiZed
representation for CHinese lAnguage understanding) on Chinese corpora and finetuning for the Chinese NLU tasks. 
The current version of NEZHA is based on BERT with a collection of proven improvements, which include Functional 
Relative Positional Encoding as an effective positional encoding scheme, Whole Word Masking strategy,
Mixed Precision Training and the LAMB Optimizer in training the models. The experimental results show that NEZHA
achieves the state-of-the-art performances when finetuned on several representative Chinese tasks, including
named entity recognition (People's Daily NER), sentence matching (LCQMC), Chinese sentiment classification (ChnSenti)
and natural language inference (XNLI).*

This model was contributed by `sijunhe <https://huggingface.co/sijunhe>`__. The original code can be found `here <https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-PyTorch>`__.

Args:
    vocab_size (`int`, optional, defaults to 21128):
        Vocabulary size of the NEZHA model. Defines the different tokens that can be represented by the
        *inputs_ids* passed to the forward method of ``NezhaModel``.
    hidden_size (`int`, optional, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, optional, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, optional, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, optional, defaults to 3072):
        The dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, optional, defaults to "gelu"):
        The non-linear activation function (function or string) in the encoder and pooler.
    hidden_dropout_prob (`float`, optional, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, optional, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, optional, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, optional, defaults to 2):
        The vocabulary size of the *token_type_ids* passed into ``NezhaModel``.
    initializer_range (`float`, optional, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, optional, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    classifier_dropout (`float`, optional, defaults to 0.1):
        The dropout ratio for attached classifiers.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.

.. class:: transformers.models.nllb_moe.configuration_nllb_moe.NllbMoeConfig(vocab_size=128112, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.05, decoder_layerdrop=0.05, use_cache=True, is_encoder_decoder=True, activation_function='relu', d_model=1024, dropout=0.1, attention_dropout=0.1, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=2, scale_embedding=True, router_bias=False, router_dtype='float32', router_ignore_padding_tokens=False, num_experts=128, expert_capacity=64, encoder_sparse_step=4, decoder_sparse_step=4, router_z_loss_coef=0.001, router_aux_loss_coef=0.001, second_expert_policy='all', normalize_router_prob_before_dropping=False, batch_prioritized_routing=False, moe_eval_capacity_token_fraction=1.0, moe_token_dropout=0.2, pad_token_id=1, bos_token_id=0, eos_token_id=2, output_router_logits=False, **kwargs)

The NLLB model was presented in `No Language Left Behind: Scaling Human-Centered Machine Translation <https://arxiv.org/abs/2207.04672>`__ by Marta R. Costa-jussà, James Cross, Onur Çelebi,
Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula,
Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews,
Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers,
Safiyyah Saleem, Holger Schwenk, and Jeff Wang.

The abstract of the paper is the following:

*Driven by the goal of eradicating language barriers on a global scale, machine translation has solidified itself as a key focus of artificial intelligence research today.
However, such efforts have coalesced around a small subset of languages, leaving behind the vast majority of mostly low-resource languages. What does it take to break the
200 language barrier while ensuring safe, high quality results, all while keeping ethical considerations in mind? In No Language Left Behind, we took on this challenge by
first contextualizing the need for low-resource language translation support through exploratory interviews with native speakers. Then, we created datasets and models aimed
at narrowing the performance gap between low and high-resource languages. More specifically, we developed a conditional compute model based on Sparsely Gated Mixture of
Experts that is trained on data obtained with novel and effective data mining techniques tailored for low-resource languages. We propose multiple architectural and training
improvements to counteract overfitting while training on thousands of tasks. Critically, we evaluated the performance of over 40,000 different translation directions using
a human-translated benchmark, Flores-200, and combined human evaluation with a novel toxicity benchmark covering all languages in Flores-200 to assess translation safety.
Our model achieves an improvement of 44% BLEU relative to the previous state-of-the-art, laying important groundwork towards realizing a universal translation system.*

Tips:

- M2M100ForConditionalGeneration is the base model for both NLLB and NLLB MoE
- The NLLB-MoE is very similar to the NLLB model, but it's feed forward layer is based on the implementation of SwitchTransformers.
- The tokenizer is the same as the NLLB models.

This model was contributed by `Arthur Zucker <https://huggingface.co/ArtZucker>`__.
The original code can be found `here <https://github.com/facebookresearch/fairseq>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the NllbMoe model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``NllbMoeModel`` or
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in encoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    second_expert_policy ( `str`, *optional*, default to `"all"`):
        The policy used for the sampling the probability of being sampled to a second expert for each token.
    normalize_router_prob_before_dropping (`bool`, *optional*, defaults to `True`):
        Whether or not to normalize the router probabilities before applying a mask based on the experts capacity
        (capacity dropping).
    batch_prioritized_routing (`bool`, *optional*, defaults to `True`):
        Whether or not to orders the tokens by their router probabilities before capacity dropping. This means that
        the tokens that have the highest probabilities will be routed before other tokens that might be further in
        the sequence.
    moe_eval_capacity_token_fraction (`float`, *optional*, defaults to 1.0):
        Fraction of tokens as capacity during validation, if set to negative, uses the same as training. Should be
        in range: (0.0, 1.0].
    num_experts (`int`, *optional*, defaults to 128):
        Number of experts for each NllbMoeSparseMlp layer.
    expert_capacity (`int`, *optional*, defaults to 64):
        Number of tokens that can be stored in each expert.
    encoder_sparse_step (`int`, *optional*, defaults to 4):
        Frequency of the sparse layers in the encoder. 4 means that one out of 4 layers will be sparse.
    decoder_sparse_step (`int`, *optional*, defaults to 4):
        Frequency of the sparse layers in the decoder. 4 means that one out of 4 layers will be sparse.
    router_dtype (`str`, *optional*, default to `"float32"`):
        The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
        *selective precision* discussion in `the paper <https://arxiv.org/abs/2101.03961>`__.
    router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
        Whether to ignore padding tokens when routing. if `False`, the padding tokens are not routed to any
        experts.
    router_bias (`bool`, *optional*, defaults to `False`):
        Whether or not the classifier of the router should have a bias.
    moe_token_dropout (`float`, *optional*, defualt ot 0.2):
        Masking rate for MoE expert output masking (EOM), which is implemented via a Dropout2d on the expert
        outputs.
    output_router_logits (`bool`, *optional*, defaults to `False`):
        Whether or not to return the router logits. Only set to `True` to get the auxiliary loss when training.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.nystromformer.configuration_nystromformer.NystromformerConfig(vocab_size=30000, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu_new', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=510, type_vocab_size=2, segment_means_seq_len=64, num_landmarks=64, conv_kernel_size=65, inv_coeff_init_option=False, initializer_range=0.02, layer_norm_eps=1e-05, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The Nyströmformer model was proposed in `*Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention* <https://arxiv.org/abs/2102.03902>`__ by Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn
Fung, Yin Li, and Vikas Singh.

The abstract from the paper is the following:

*Transformers have emerged as a powerful tool for a broad range of natural language processing tasks. A key component
that drives the impressive performance of Transformers is the self-attention mechanism that encodes the influence or
dependence of other tokens on each specific token. While beneficial, the quadratic complexity of self-attention on the
input sequence length has limited its application to longer sequences -- a topic being actively studied in the
community. To address this limitation, we propose Nyströmformer -- a model that exhibits favorable scalability as a
function of sequence length. Our idea is based on adapting the Nyström method to approximate standard self-attention
with O(n) complexity. The scalability of Nyströmformer enables application to longer sequences with thousands of
tokens. We perform evaluations on multiple downstream tasks on the GLUE benchmark and IMDB reviews with standard
sequence length, and find that our Nyströmformer performs comparably, or in a few cases, even slightly better, than
standard self-attention. On longer sequence tasks in the Long Range Arena (LRA) benchmark, Nyströmformer performs
favorably relative to other efficient self-attention methods. Our code is available at this https URL.*

This model was contributed by `novice03 <https://huggingface.co/novice03>`__. The original code can be found `here <https://github.com/mlpen/Nystromformer>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30000):
        Vocabulary size of the Nystromformer model. Defines the number of different tokens that can be represented
        by the `inputs_ids` passed when calling ``NystromformerModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``NystromformerModel``.
    segment_means_seq_len (`int`, *optional*, defaults to 64):
        Sequence length used in segment-means.
    num_landmarks (`int`, *optional*, defaults to 64):
        The number of landmark (or Nystrom) points to use in Nystrom approximation of the softmax self-attention
        matrix.
    conv_kernel_size (`int`, *optional*, defaults to 65):
        The kernel size of depthwise convolution used in Nystrom approximation.
    inv_coeff_init_option (`bool`, *optional*, defaults to `False`):
        Whether or not to use exact coefficient computation for the initial values for the iterative method of
        calculating the Moore-Penrose inverse of a matrix.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.

.. class:: transformers.models.openai.configuration_openai.OpenAIGPTConfig(vocab_size=40478, n_positions=512, n_embd=768, n_layer=12, n_head=12, afn='gelu', resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-05, initializer_range=0.02, summary_type='cls_index', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, **kwargs)

OpenAI GPT model was proposed in `Improving Language Understanding by Generative Pre-Training <https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf>`__
by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever. It's a causal (unidirectional) transformer
pre-trained using language modeling on a large corpus will long range dependencies, the Toronto Book Corpus.

The abstract from the paper is the following:

*Natural language understanding comprises a wide range of diverse tasks such as textual entailment, question answering,
semantic similarity assessment, and document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to
perform adequately. We demonstrate that large gains on these tasks can be realized by generative pretraining of a
language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. In
contrast to previous approaches, we make use of task-aware input transformations during fine-tuning to achieve
effective transfer while requiring minimal changes to the model architecture. We demonstrate the effectiveness of our
approach on a wide range of benchmarks for natural language understanding. Our general task-agnostic model outperforms
discriminatively trained models that use architectures specifically crafted for each task, significantly improving upon
the state of the art in 9 out of the 12 tasks studied.*

Tips:

- GPT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- GPT was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows GPT-2 to generate syntactically coherent text as it can be
  observed in the *run_generation.py* example script.

`Write With Transformer <https://transformer.huggingface.co/doc/gpt>`__ is a webapp created and hosted by Hugging Face
showcasing the generative capabilities of several models. GPT is one of them.

This model was contributed by `thomwolf <https://huggingface.co/thomwolf>`__. The original code can be found `here <https://github.com/openai/finetune-transformer-lm>`__.

Note:

If you want to reproduce the original tokenization process of the *OpenAI GPT* paper, you will need to install `ftfy`
and `SpaCy`:

```bash
pip install spacy ftfy==4.4.3
python -m spacy download en
```

If you don't install `ftfy` and `SpaCy`, the ``OpenAIGPTTokenizer`` will default to tokenize
using BERT's `BasicTokenizer` followed by Byte-Pair Encoding (which should be fine for most usage, don't worry).

Args:
    vocab_size (`int`, *optional*, defaults to 40478):
        Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``OpenAIGPTModel`` or ``TFOpenAIGPTModel``.
    n_positions (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    n_embd (`int`, *optional*, defaults to 768):
        Dimensionality of the embeddings and hidden states.
    n_layer (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    afn (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    resid_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    embd_pdrop (`int`, *optional*, defaults to 0.1):
        The dropout ratio for the embeddings.
    attn_pdrop (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    summary_type (`str`, *optional*, defaults to `"cls_index"`):
        Argument used when doing sequence summary, used in the models ``OpenAIGPTDoubleHeadsModel`` and
        ``OpenAIGPTDoubleHeadsModel``.

        Has to be one of the following options:

            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models ``OpenAIGPTDoubleHeadsModel`` and
        ``OpenAIGPTDoubleHeadsModel``.

        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary, used in the models ``OpenAIGPTDoubleHeadsModel`` and
        ``OpenAIGPTDoubleHeadsModel``.

        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary, used in the models ``OpenAIGPTDoubleHeadsModel`` and
        ``OpenAIGPTDoubleHeadsModel``.

        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Argument used when doing sequence summary, used in the models ``OpenAIGPTDoubleHeadsModel`` and
        ``OpenAIGPTDoubleHeadsModel``.

        The dropout ratio to be used after the projection and activation.

.. class:: transformers.models.opt.configuration_opt.OPTConfig(vocab_size=50272, hidden_size=768, num_hidden_layers=12, ffn_dim=3072, max_position_embeddings=2048, do_layer_norm_before=True, _remove_final_layer_norm=False, word_embed_proj_dim=None, dropout=0.1, attention_dropout=0.0, num_attention_heads=12, activation_function='relu', layerdrop=0.0, init_std=0.02, use_cache=True, pad_token_id=1, bos_token_id=2, eos_token_id=2, enable_bias=True, layer_norm_elementwise_affine=True, **kwargs)

The OPT model was proposed in `Open Pre-trained Transformer Language Models <https://arxiv.org/pdf/2205.01068>`__ by Meta AI.
OPT is a series of open-sourced large causal language models which perform similar in performance to GPT3.

The abstract from the paper is the following:

*Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models.*

Tips:
- OPT has the same architecture as ``BartDecoder``.
- Contrary to GPT2, OPT adds the EOS token `</s>` to the beginning of every prompt.

This model was contributed by `Arthur Zucker <https://huggingface.co/ArthurZ>`__, `Younes Belkada <https://huggingface.co/ybelkada>`__, and `Patrick Von Platen <https://huggingface.co/patrickvonplaten>`__.
The original code can be found `here <https://github.com/facebookresearch/metaseq>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50272):
        Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``OPTModel``
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    ffn_dim (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    do_layer_norm_before (`bool`, *optional*, defaults to `True`):
        Whether to perform layer normalization before the attention block.
    word_embed_proj_dim (`int`, *optional*):
        `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
        `hidden_size`.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__ for more
        details.
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    enable_bias (`bool`, *optional*, defaults to `True`):
        Whether or not if the linear layers in the attention blocks should use the bias term.
    layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
        Whether or not if the layer norms should have learnable parameters.

.. class:: transformers.models.pegasus.configuration_pegasus.PegasusConfig(vocab_size=50265, max_position_embeddings=1024, encoder_layers=12, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=12, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=0, scale_embedding=False, pad_token_id=0, eos_token_id=1, forced_eos_token_id=1, **kwargs)

The Pegasus model was proposed in `PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization <https://arxiv.org/pdf/1912.08777.pdf>`__ by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu on Dec 18, 2019.

According to the abstract,

- Pegasus' pretraining task is intentionally similar to summarization: important sentences are removed/masked from an
  input document and are generated together as one output sequence from the remaining sentences, similar to an
  extractive summary.
- Pegasus achieves SOTA summarization performance on all 12 downstream tasks, as measured by ROUGE and human eval.

This model was contributed by `sshleifer <https://huggingface.co/sshleifer>`__. The Authors' code can be found `here <https://github.com/google-research/pegasus>`__.

Tips:

- Sequence-to-sequence model with the same encoder-decoder model architecture as BART. Pegasus is pre-trained jointly on two self-supervised objective functions: Masked Language Modeling (MLM) and a novel summarization specific pretraining objective, called Gap Sentence Generation (GSG).

  * MLM: encoder input tokens are randomly replaced by a mask tokens and have to be predicted by the encoder (like in BERT)
  * GSG: whole encoder input sentences are replaced by a second mask token and fed to the decoder, but which has a causal mask to hide the future words like a regular auto-regressive transformer decoder.

Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the PEGASUS model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``PegasusModel`` or ``TFPegasusModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `False`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)
    forced_eos_token_id (`int`, *optional*, defaults to 1):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.

.. class:: transformers.models.pegasus_x.configuration_pegasus_x.PegasusXConfig(vocab_size=96103, max_position_embeddings=16384, encoder_layers=16, encoder_ffn_dim=4096, encoder_attention_heads=16, decoder_layers=16, decoder_ffn_dim=4096, decoder_attention_heads=16, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=1024, dropout=0.1, attention_dropout=0.0, activation_dropout=0.0, init_std=0.02, decoder_start_token_id=0, scale_embedding=True, pad_token_id=0, eos_token_id=1, forced_eos_token_id=1, num_global_tokens=32, block_size=512, stagger_local_blocks=True, **kwargs)

The PEGASUS-X model was proposed in `Investigating Efficiently Extending Transformers for Long Input Summarization <https://arxiv.org/abs/2208.04347>`__  by Jason Phang, Yao Zhao and Peter J. Liu.

PEGASUS-X (PEGASUS eXtended) extends the PEGASUS models for long input summarization through additional long input pretraining and using staggered block-local attention with global tokens in the encoder.

The abstract from the paper is the following:

*While large pretrained Transformer models have proven highly capable at tackling natural language tasks, handling long sequence inputs continues to be a significant challenge. One such task is long input summarization, where inputs are longer than the maximum input context of most pretrained models. Through an extensive set of experiments, we investigate what model architectural changes and pretraining paradigms can most efficiently adapt a pretrained Transformer for long input summarization. We find that a staggered, block-local Transformer with global encoder tokens strikes a good balance of performance and efficiency, and that an additional pretraining phase on long sequences meaningfully improves downstream summarization performance. Based on our findings, we introduce PEGASUS-X, an extension of the PEGASUS model with additional long input pretraining to handle inputs of up to 16K tokens. PEGASUS-X achieves strong performance on long input summarization tasks comparable with much larger models while adding few additional parameters and not requiring model parallelism to train.*

Tips:

* PEGASUS-X uses the same tokenizer as PEGASUS.

This model was contributed by `zphang <<https://huggingface.co/zphang>`__. The original code can be found `here <https://github.com/google-research/pegasus>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 96103):
        Vocabulary size of the PEGASUS-X model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``PegasusXModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimension of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 16):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 16):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimension of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimension of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    max_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)
    forced_eos_token_id (`int`, *optional*, defaults to 1):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.
    num_global_tokens (`int`, *optional*, defaults to 128):
        Number of global tokens to use for the encoder
    block_size (`int`, *optional*, defaults to 512):
        Block size for encoder local attention. Sequence length should be an exact multiple of block size.
        block_size must be a multiple of 2 if stagger_local_block is True
    stagger_local_block (`bool`, *optional*, defaults to `True`):
        Whether to stagger every other local attention by half a block

.. class:: transformers.models.persimmon.configuration_persimmon.PersimmonConfig(vocab_size=262144, hidden_size=4096, intermediate_size=16384, num_hidden_layers=36, num_attention_heads=64, hidden_act='relu2', max_position_embeddings=16384, initializer_range=0.02, layer_norm_eps=1e-05, use_cache=True, tie_word_embeddings=False, rope_theta=25000.0, rope_scaling=None, qk_layernorm=True, hidden_dropout=0.0, attention_dropout=0.0, partial_rotary_factor=0.5, pad_token_id=None, bos_token_id=1, eos_token_id=2, **kwargs)

The Persimmon model was created by `ADEPT <https://www.adept.ai/blog/persimmon-8b>`__, and authored by Erich Elsen, Augustus Odena, Maxwell Nye, Sağnak Taşırlar, Tri Dao, Curtis Hawthorne, Deepak Moparthi, Arushi Somani.

The authors introduced Persimmon-8B, a decoder model based on the classic transformers architecture, with query and key normalization. Persimmon-8B is a fully permissively-licensed model with approximately 8 billion parameters, released under the Apache license.  Some of the key attributes of Persimmon-8B are long context size (16K), performance, and capabilities for multimodal extensions.

The authors showcase their approach to model evaluation, focusing on practical text generation, mirroring how users interact with language models. The work also includes a comparative analysis, pitting Persimmon-8B against other prominent models (MPT 7B Instruct and Llama 2 Base 7B 1-Shot), across various evaluation tasks. The results demonstrate Persimmon-8B's competitive performance, even with limited training data.

In terms of model details, the work outlines the architecture and training methodology of Persimmon-8B, providing insights into its design choices, sequence length, and dataset composition. The authors present a fast inference code that outperforms traditional implementations through operator fusion and CUDA graph utilization while maintaining code coherence. They express their anticipation of how the community will leverage this contribution to drive innovation, hinting at further upcoming releases as part of an ongoing series of developments.


<Tip warning={true}>

The `Persimmon` models were trained using `bfloat16`, but the original inference uses `float16` The checkpoints uploaded on the hub use `torch_dtype = 'float16'` which will be
used by the `AutoModel` API to cast the checkpoints from `torch.float32` to `torch.float16`. 

The `dtype` of the online weights is mostly irrelevant, unless you are using `torch_dtype="auto"` when initializing a model using `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`. The reason is that the model will first be downloaded ( using the `dtype` of the checkpoints online) then it will be cast to the default `dtype` of `torch` (becomes `torch.float32`). Users should specify the `torch_dtype` they want, and if they don't it will be `torch.float32`.

Finetuning the model in `float16` is not recommended and known to produce `nan`, as such the model should be fine-tuned in `bfloat16`.

</Tip>


Tips:

- To convert the model, you need to clone the original repository using `git clone https://github.com/persimmon-ai-labs/adept-inference`, then get the checkpoints:

```bash
git clone https://github.com/persimmon-ai-labs/adept-inference
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_base_model_release.tar
tar -xvf 8b_base_model_release.tar
python src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py  --input_dir /path/to/downloaded/persimmon/weights/ --output_dir /output/path \
    --pt_model_path /path/to/8b_chat_model_release/iter_0001251/mp_rank_00/model_optim_rng.pt
    --ada_lib_path /path/to/adept-inference
```

For the chat model:
```bash
wget https://axtkn4xl5cip.objectstorage.us-phoenix-1.oci.customer-oci.com/n/axtkn4xl5cip/b/adept-public-data/o/8b_chat_model_release.tar
tar -xvf 8b_base_model_release.tar
```

Thereafter, models can be loaded via:

```py
from transformers import PersimmonForCausalLM, PersimmonTokenizer

model = PersimmonForCausalLM.from_pretrained("/output/path")
tokenizer = PersimmonTokenizer.from_pretrained("/output/path")
```

This model was contributed by `ArthurZ <https://huggingface.co/ArthurZ>`__.
The original code can be found `here <https://github.com/persimmon-ai-labs/adept-inference>`__.

- Perismmon uses a `sentencepiece` based tokenizer, with a `Unigram` model. It supports bytefallback, which is only available in `tokenizers==0.14.0` for the fast tokenizer.
The `LlamaTokenizer` is used as it is a standard wrapper around sentencepiece. The `chat` template will be updated with the templating functions in a follow up PR!

- The authors suggest to use the following prompt format for the chat mode: `f"human: {prompt}\n\nadept:"`


Args:
    vocab_size (`int`, *optional*, defaults to 262144):
        Vocabulary size of the Persimmon model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``PersimmonModel``
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 16384):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 36):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 64):
        Number of attention heads for each attention layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"relu2"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that this model might ever be used with.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        The epsilon used by the rms normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    tie_word_embeddings(`bool`, *optional*, defaults to `False`):
        Whether to tie weight embeddings
    rope_theta (`float`, *optional*, defaults to 25000.0):
        The base period of the RoPE embeddings.
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
        strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
        `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalPersimmon/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This
        is an experimental feature, subject to breaking API changes in future versions.
    qk_layernorm (`bool`, *optional*, default to `True`):
        Whether or not to normalize the Queries and Keys after projecting the hidden states
    hidden_dropout (`float`, *optional*, default to 0.0):
        The dropout ratio after applying the MLP to the hidden states.
    attention_dropout (`float`, *optional*, default to 0.0):
        The dropout ratio after computing the attention scores.
    partial_rotary_factor (`float`, *optional*, default to 0.5):
        Percentage of the query and keys which will have rotary embedding.

.. class:: transformers.models.phi.configuration_phi.PhiConfig(vocab_size=51200, hidden_size=2048, intermediate_size=8192, num_hidden_layers=24, num_attention_heads=32, num_key_value_heads=None, resid_pdrop=0.0, embd_pdrop=0.0, attention_dropout=0.0, hidden_act='gelu_new', max_position_embeddings=2048, initializer_range=0.02, layer_norm_eps=1e-05, use_cache=True, tie_word_embeddings=False, rope_theta=10000.0, rope_scaling=None, partial_rotary_factor=0.5, qk_layernorm=False, bos_token_id=1, eos_token_id=2, **kwargs)

The Phi-1 model was proposed in `Textbooks Are All You Need <https://arxiv.org/abs/2306.11644>`__ by Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee and Yuanzhi Li.

The Phi-1.5 model was proposed in `Textbooks Are All You Need II: phi-1.5 technical report <https://arxiv.org/abs/2309.05463>`__ by Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar and Yin Tat Lee.

#Args:
    vocab_size (`int`, *optional*, defaults to 51200):
        Vocabulary size of the Phi model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``PhiModel``.
    hidden_size (`int`, *optional*, defaults to 2048):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 8192):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer decoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer decoder.
    num_key_value_heads (`int`, *optional*):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to
        `num_attention_heads`.
    resid_pdrop (`float`, *optional*, defaults to 0.0):
        Dropout probability for mlp outputs.
    embd_pdrop (`int`, *optional*, defaults to 0.0):
        The dropout ratio for the embeddings.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio after computing the attention scores.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Phi-1 and Phi-1.5 supports up to 2048
        tokens.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the rms normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`. Whether to tie weight embeddings or not.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to tie weight embeddings
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
        strategies: linear and dynamic. Their scaling factor must be an float greater than 1. The expected format
        is `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalPersimmon/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This
        is an experimental feature, subject to breaking API changes in future versions.
    partial_rotary_factor (`float`, *optional*, defaults to 0.5):
        Percentage of the query and keys which will have rotary embedding.
    qk_layernorm (`bool`, *optional*, defaults to `False`):
        Whether or not to normalize the Queries and Keys after projecting the hidden states.
    bos_token_id (`int`, *optional*, defaults to 1):
        Denotes beginning of sequences token id.
    eos_token_id (`int`, *optional*, defaults to 2):
        Denotes end of sequences token id.

.. class:: transformers.models.plbart.configuration_plbart.PLBartConfig(vocab_size=50005, max_position_embeddings=1024, encoder_layers=6, encoder_ffn_dim=3072, encoder_attention_heads=12, decoder_layers=6, decoder_ffn_dim=3072, decoder_attention_heads=12, encoder_layerdrop=0.0, decoder_layerdrop=0.0, use_cache=True, is_encoder_decoder=True, activation_function='gelu', d_model=768, dropout=0.1, attention_dropout=0.1, activation_dropout=0.0, init_std=0.02, classifier_dropout=0.0, scale_embedding=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, forced_eos_token_id=2, **kwargs)

of PLBart

The PLBART model was proposed in `Unified Pre-training for Program Understanding and Generation <https://arxiv.org/abs/2103.06333>`__ by Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang.
This is a BART-like model which can be used to perform code-summarization, code-generation, and code-translation tasks. The pre-trained model `plbart-base` has been trained using multilingual denoising task
on Java, Python and English.

According to the abstract

*Code summarization and generation empower conversion between programming language (PL) and natural language (NL),
while code translation avails the migration of legacy code from one PL to another. This paper introduces PLBART, 
a sequence-to-sequence model capable of performing a broad spectrum of program and language understanding and generation tasks.
PLBART is pre-trained on an extensive collection of Java and Python functions and associated NL text via denoising autoencoding.
Experiments on code summarization in the English language, code generation, and code translation in seven programming languages
show that PLBART outperforms or rivals state-of-the-art models. Moreover, experiments on discriminative tasks, e.g., program
repair, clone detection, and vulnerable code detection, demonstrate PLBART's effectiveness in program understanding.
Furthermore, analysis reveals that PLBART learns program syntax, style (e.g., identifier naming convention), logical flow
(e.g., if block inside an else block is equivalent to else if block) that are crucial to program semantics and thus excels
even with limited annotations.*

This model was contributed by `gchhablani <https://huggingface.co/gchhablani>`__. The Authors' code can be found `here <https://github.com/wasiahmad/PLBART>`__.

#Args:
    vocab_size (`int`, *optional*, defaults to 50005):
        Vocabulary size of the PLBART model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``PLBartModel``.
    d_model (`int`, *optional*, defaults to 768):
        Dimensionality of the layers and the pooler layer.
    encoder_layers (`int`, *optional*, defaults to 6):
        Number of encoder layers.
    decoder_layers (`int`, *optional*, defaults to 6):
        Number of decoder layers.
    encoder_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer decoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    encoder_ffn_dim (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    max_position_embeddings (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    encoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    decoder_layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the decoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    scale_embedding (`bool`, *optional*, defaults to `True`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models)
    forced_eos_token_id (`int`, *optional*, defaults to 2):
        The id of the token to force as the last generated token when `max_length` is reached. Usually set to
        `eos_token_id`.

.. class:: transformers.models.prophetnet.configuration_prophetnet.ProphetNetConfig(activation_dropout: Optional[float] = 0.1, activation_function: Union[str, Callable, NoneType] = 'gelu', vocab_size: Optional[int] = 30522, hidden_size: Optional[int] = 1024, encoder_ffn_dim: Optional[int] = 4096, num_encoder_layers: Optional[int] = 12, num_encoder_attention_heads: Optional[int] = 16, decoder_ffn_dim: Optional[int] = 4096, num_decoder_layers: Optional[int] = 12, num_decoder_attention_heads: Optional[int] = 16, attention_dropout: Optional[float] = 0.1, dropout: Optional[float] = 0.1, max_position_embeddings: Optional[int] = 512, init_std: Optional[float] = 0.02, is_encoder_decoder: Optional[bool] = True, add_cross_attention: Optional[bool] = True, decoder_start_token_id: Optional[int] = 0, ngram: Optional[int] = 2, num_buckets: Optional[int] = 32, relative_max_distance: Optional[int] = 128, disable_ngram_loss: Optional[bool] = False, eps: Optional[float] = 0.0, use_cache: Optional[bool] = True, pad_token_id: Optional[int] = 0, bos_token_id: Optional[int] = 1, eos_token_id: Optional[int] = 2, **kwargs)

The ProphetNet model was proposed in `ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training, <https://arxiv.org/abs/2001.04063>`__ by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.

ProphetNet is an encoder-decoder model and can predict n-future tokens for "ngram" language modeling instead of just
the next token.

The abstract from the paper is the following:

*In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel
self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of
the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by
n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time
step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent
overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale
dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for
abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new
state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus.*

Tips:

- ProphetNet is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- The model architecture is based on the original Transformer, but replaces the “standard” self-attention mechanism in the decoder by a a main self-attention mechanism and a self and n-stream (predict) self-attention mechanism.

The Authors' code can be found `here <https://github.com/microsoft/ProphetNet>`__.

Args:
    activation_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for activations inside the fully connected layer.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``ProphetNetModel``.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    num_encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    num_encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the `intermediate` (often named feed-forward) layer in decoder.
    num_decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    num_decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    add_cross_attention (`bool`, *optional*, defaults to `True`):
        Whether cross-attention layers should be added to the model.
    is_encoder_decoder (`bool`, *optional*, defaults to `True`):
        Whether this is an encoder/decoder model.
    pad_token_id (`int`, *optional*, defaults to 1)
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 0)
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 2)
        End of stream token id.
    ngram (`int`, *optional*, defaults to 2)
        Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
        token.
    num_buckets (`int`, *optional*, defaults to 32)
        The number of buckets to use for each attention layer. This is for relative position calculation. See the
        `T5 paper <see https://arxiv.org/abs/1910.10683>`__ for more details.
    relative_max_distance (`int`, *optional*, defaults to 128)
        Relative distances greater than this number will be put into the last same bucket. This is for relative
        position calculation. See the `T5 paper <see https://arxiv.org/abs/1910.10683>`__ for more details.
    disable_ngram_loss (`bool`, *optional*, defaults to `False`):
        Whether be trained predicting only the next first token.
    eps (`float`, *optional*, defaults to 0.0):
        Controls the `epsilon` parameter value for label smoothing in the loss calculation. If set to 0, no label
        smoothing is performed.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.qwen2.configuration_qwen2.Qwen2Config(vocab_size=151936, hidden_size=4096, intermediate_size=22016, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32, hidden_act='silu', max_position_embeddings=32768, initializer_range=0.02, rms_norm_eps=1e-06, use_cache=True, tie_word_embeddings=False, rope_theta=10000.0, use_sliding_window=False, sliding_window=4096, max_window_layers=28, attention_dropout=0.0, **kwargs)

Qwen2 is the new model series of large language models from the Qwen team. Previously, we released the Qwen series, including Qwen-72B, Qwen-1.8B, Qwen-VL, Qwen-Audio, etc.

#Args:
    vocab_size (`int`, *optional*, defaults to 151936):
        Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``Qwen2Model``
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 22016):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer encoder.
    num_key_value_heads (`int`, *optional*, defaults to 32):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to `32`.
    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to 32768):
        The maximum sequence length that this model might ever be used with.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    rms_norm_eps (`float`, *optional*, defaults to 1e-06):
        The epsilon used by the rms normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether the model's input and output word embeddings should be tied.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    use_sliding_window (`bool`, *optional*, defaults to `False`):
        Whether to use sliding window attention.
    sliding_window (`int`, *optional*, defaults to 4096):
        Sliding window attention (SWA) window size. If not specified, will default to `4096`.
    max_window_layers (`int`, *optional*, defaults to 28):
        The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.

::

>>> from transformers import Qwen2Model, Qwen2Config

>>> # Initializing a Qwen2 style configuration
>>> configuration = Qwen2Config()

>>> # Initializing a model from the Qwen2-7B style configuration
>>> model = Qwen2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

.. class:: transformers.models.reformer.configuration_reformer.ReformerConfig(attention_head_size=64, attn_layers=['local', 'lsh', 'local', 'lsh', 'local', 'lsh'], axial_norm_std=1.0, axial_pos_embds=True, axial_pos_shape=[64, 64], axial_pos_embds_dim=[64, 192], chunk_size_lm_head=0, eos_token_id=2, feed_forward_size=512, hash_seed=None, hidden_act='relu', hidden_dropout_prob=0.05, hidden_size=256, initializer_range=0.02, is_decoder=False, layer_norm_eps=1e-12, local_num_chunks_before=1, local_num_chunks_after=0, local_attention_probs_dropout_prob=0.05, local_attn_chunk_length=64, lsh_attn_chunk_length=64, lsh_attention_probs_dropout_prob=0.0, lsh_num_chunks_before=1, lsh_num_chunks_after=0, max_position_embeddings=4096, num_attention_heads=12, num_buckets=None, num_hashes=1, pad_token_id=0, vocab_size=320, tie_word_embeddings=False, use_cache=True, classifier_dropout=None, **kwargs)

The Reformer model was proposed in the paper `Reformer: The Efficient Transformer <https://arxiv.org/abs/2001.04451.pdf>`__ by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.

The abstract from the paper is the following:

*Large Transformer models routinely achieve state-of-the-art results on a number of tasks but training these models can
be prohibitively costly, especially on long sequences. We introduce two techniques to improve the efficiency of
Transformers. For one, we replace dot-product attention by one that uses locality-sensitive hashing, changing its
complexity from O(L^2) to O(Llog(L)), where L is the length of the sequence. Furthermore, we use reversible residual
layers instead of the standard residuals, which allows storing activations only once in the training process instead of
N times, where N is the number of layers. The resulting model, the Reformer, performs on par with Transformer models
while being much more memory-efficient and much faster on long sequences.*

This model was contributed by `patrickvonplaten <https://huggingface.co/patrickvonplaten>`__. The Authors' code can be
found `here <https://github.com/google/trax/tree/master/trax/models/reformer>`__.

Tips:

- Reformer does **not** work with *torch.nn.DataParallel* due to a bug in PyTorch, see `issue #36035 <https://github.com/pytorch/pytorch/issues/36035>`__.
- Use Axial position encoding (see below for more details). It’s a mechanism to avoid having a huge positional encoding matrix (when the sequence length is very big) by factorizing it into smaller matrices.
- Replace traditional attention by LSH (local-sensitive hashing) attention (see below for more details). It’s a technique to avoid computing the full product query-key in the attention layers.
- Avoid storing the intermediate results of each layer by using reversible transformer layers to obtain them during the backward pass (subtracting the residuals from the input of the next layer gives them back) or recomputing them for results inside a given layer (less efficient than storing them but saves memory).
- Compute the feedforward operations by chunks and not on the whole batch.

Args:
    attention_head_size (`int`, *optional*, defaults to 64):
        Dimensionality of the projected key, query and value vectors
    attn_layers (`List`str]`, *optional*, defaults to `["local", "lsh", "local", "lsh", "local", "lsh"]`):
        List of attention layer types in ascending order. It can be chosen between a LSHSelfAttention layer
        (`"lsh"`) and a LocalSelfAttention layer (`"local"`).

        For more information on LSHSelfAttention layer, see [LSH Self Attention <reformer#lsh-self-attention>`__. For
        more information on LocalSelfAttention layer, see `Local Self Attention <reformer#local-self-attention>`__.
    axial_pos_embds (`bool`, *optional*, defaults to `True`):
        Whether or not to use axial position embeddings. For more information on how axial position embeddings
        work, see `Axial Position Encodings <reformer#axial-positional-encodings>`__.
    axial_norm_std (`float`, *optional*, defaults to 1.0):
        The standard deviation of the normal_initializer for initializing the weight matrices of the axial
        positional encodings.
    axial_pos_shape (`List`int]`, *optional*, defaults to `[64, 64]`):
        The position dims of the axial position encodings. During training, the product of the position dims has to
        be equal to the sequence length.

        For more information on how axial position embeddings work, see [Axial Position
        Encodings <reformer#axial-positional-encodings>`__.
    axial_pos_embds_dim (`List`int]`, *optional*, defaults to `[64, 192]`):
        The embedding dims of the axial position encodings. The sum of the embedding dims has to be equal to the
        hidden size.

        For more information on how axial position embeddings work, see [Axial Position
        Encodings <reformer#axial-positional-encodings>`__.
    chunk_size_lm_head (`int`, *optional*, defaults to 0):
        The chunk size of the final language model feed forward head layer. A chunk size of 0 means that the feed
        forward layer is not chunked. A chunk size of n means that the feed forward layer processes n <
        sequence_length embeddings at a time.

        For more information on feed forward chunking, see `How does Feed Forward Chunking
        work? <../glossary#feed-forward-chunking>`__.
    eos_token_id (`int`, *optional*, defaults to 2):
        The token id for the end-of-sentence token.
    feed_forward_size (`int`, *optional*, defaults to 512):
        Dimensionality of the feed_forward layer in the residual attention block.
    hash_seed (`int`, *optional*):
        Seed that can be used to make local sensitive hashing in `LSHSelfAttention` deterministic. This should only
        be set for testing purposed. For evaluation and training purposes `hash_seed` should be left as `None` to
        ensure fully random rotations in local sensitive hashing scheme.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"relu"`):
        The non-linear activation function (function or string) in the feed forward layer in the residual attention
        block. If string, `"gelu"`, `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.05):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    hidden_size (`int`, *optional*, defaults to 256):
        Dimensionality of the output hidden states of the residual attention blocks.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether or not to use a causal mask in addition to the `attention_mask` passed to ``ReformerModel``. When
        using the Reformer for causal language modeling, this argument should be set to `True`.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    local_chunk_length (`int`, *optional*, defaults to 64):
        Length of chunk which attends to itself in `LocalSelfAttention`. Chunking reduces memory complexity from
        sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk
        length (chunked self attention).
    local_num_chunks_before (`int`, *optional*, defaults to 1):
        Number of previous neighbouring chunks to attend to in `LocalSelfAttention` layer to itself.
    local_num_chunks_after (`int`, *optional*, defaults to 0):
        Number of following neighbouring chunks to attend to in `LocalSelfAttention` layer in addition to itself.
    local_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities in `LocalSelfAttention`.
    lsh_attn_chunk_length (`int`, *optional*, defaults to 64):
        Length of chunk which attends to itself in `LSHSelfAttention`. Chunking reduces memory complexity from
        sequence length x sequence length (self attention) to chunk length x chunk length x sequence length / chunk
        length (chunked self attention).
    lsh_num_chunks_before (`int`, *optional*, defaults to 1):
        Number of previous neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
    lsh_num_chunks_after (`int`, *optional*, defaults to 0):
        Number of following neighbouring chunks to attend to in `LSHSelfAttention` layer to itself.
    lsh_attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities in `LSHSelfAttention`.
    max_position_embeddings (`int`, *optional*, defaults to 4096):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    num_buckets (`int` or `List[int]`, *optional*):
        Number of buckets, the key query vectors can be "hashed into" using the locality sensitive hashing scheme.
        Each query key vector is hashed into a hash in `1, ..., num_buckets`. The number of buckets can also be
        factorized into a list for improved memory complexity. In this case, each query key vector is hashed into a
        hash in `1-1, 1-2, ..., num_buckets[0]-1, ..., num_buckets[0]-num_buckets[1]` if `num_buckets` is
        factorized into two factors. The number of buckets (or the product the factors) should approximately equal
        sequence length / lsh_chunk_length. If `num_buckets` not set, a good value is calculated on the fly.
    num_hashes (`int`, *optional*, defaults to 1):
        Number of hashing rounds (e.g., number of random rotations) in Local Sensitive Hashing scheme. The higher
        `num_hashes`, the more accurate the `LSHSelfAttention` becomes, but also the more memory and time intensive
        the hashing becomes.
    pad_token_id (`int`, *optional*, defaults to 0):
        The token id for the padding token.
    vocab_size (`int`, *optional*, defaults to 320):\
        Vocabulary size of the Reformer model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``ReformerModel``.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to tie input and output embeddings.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.rembert.configuration_rembert.RemBertConfig(vocab_size=250300, hidden_size=1152, num_hidden_layers=32, num_attention_heads=18, input_embedding_size=256, output_embedding_size=1664, intermediate_size=4608, hidden_act='gelu', hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, use_cache=True, pad_token_id=0, bos_token_id=312, eos_token_id=313, **kwargs)

The RemBERT model was proposed in `Rethinking Embedding Coupling in Pre-trained Language Models <https://arxiv.org/abs/2010.12821>`__ by Hyung Won Chung, Thibault Févry, Henry Tsai, Melvin Johnson, Sebastian Ruder.

The abstract from the paper is the following:

*We re-evaluate the standard practice of sharing weights between input and output embeddings in state-of-the-art
pre-trained language models. We show that decoupled embeddings provide increased modeling flexibility, allowing us to
significantly improve the efficiency of parameter allocation in the input embedding of multilingual models. By
reallocating the input embedding parameters in the Transformer layers, we achieve dramatically better performance on
standard natural language understanding tasks with the same number of parameters during fine-tuning. We also show that
allocating additional capacity to the output embedding provides benefits to the model that persist through the
fine-tuning stage even though the output embedding is discarded after pre-training. Our analysis shows that larger
output embeddings prevent the model's last layers from overspecializing to the pre-training task and encourage
Transformer representations to be more general and more transferable to other tasks and languages. Harnessing these
findings, we are able to train models that achieve strong performance on the XTREME benchmark without increasing the
number of parameters at the fine-tuning stage.*

Tips:

For fine-tuning, RemBERT can be thought of as a bigger version of mBERT with an ALBERT-like factorization of the
embedding layer. The embeddings are not tied in pre-training, in contrast with BERT, which enables smaller input
embeddings (preserved during fine-tuning) and bigger output embeddings (discarded at fine-tuning). The tokenizer is
also similar to the Albert one rather than the BERT one.

Args:
    vocab_size (`int`, *optional*, defaults to 250300):
        Vocabulary size of the RemBERT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``RemBertModel`` or ``TFRemBertModel``. Vocabulary size of the model.
        Defines the different tokens that can be represented by the *inputs_ids* passed to the forward method of
        ``RemBertModel``.
    hidden_size (`int`, *optional*, defaults to 1152):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 18):
        Number of attention heads for each attention layer in the Transformer encoder.
    input_embedding_size (`int`, *optional*, defaults to 256):
        Dimensionality of the input embeddings.
    output_embedding_size (`int`, *optional*, defaults to 1664):
        Dimensionality of the output embeddings.
    intermediate_size (`int`, *optional*, defaults to 4608):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0):
        The dropout ratio for the attention probabilities.
    classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the classifier layer when fine-tuning.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``RemBertModel`` or ``TFRemBertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.

.. class:: transformers.models.roberta.configuration_roberta.RobertaConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The RoBERTa model was proposed in `RoBERTa: A Robustly Optimized BERT Pretraining Approach <https://arxiv.org/abs/1907.11692>`__ by Yinhan Liu, `Myle Ott <https://huggingface.co/myleott>`__, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer
Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. It is based on Google's BERT model released in 2018.

It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with
much larger mini-batches and learning rates.

The abstract from the paper is the following:

*Language model pretraining has led to significant performance gains but careful comparison between different
approaches is challenging. Training is computationally expensive, often done on private datasets of different sizes,
and, as we will show, hyperparameter choices have significant impact on the final results. We present a replication
study of BERT pretraining (Devlin et al., 2019) that carefully measures the impact of many key hyperparameters and
training data size. We find that BERT was significantly undertrained, and can match or exceed the performance of every
model published after it. Our best model achieves state-of-the-art results on GLUE, RACE and SQuAD. These results
highlight the importance of previously overlooked design choices, and raise questions about the source of recently
reported improvements. We release our models and code.*

Tips:

- This implementation is the same as ``BertModel`` with a tiny embeddings tweak as well as a setup
  for Roberta pretrained models.
- RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a
  different pretraining scheme.
- RoBERTa doesn't have `token_type_ids`, you don't need to indicate which token belongs to which segment. Just
  separate your segments with the separation token `tokenizer.sep_token` (or `</s>`)
- Same as BERT with better pretraining tricks:

    * dynamic masking: tokens are masked differently at each epoch, whereas BERT does it once and for all
    * together to reach 512 tokens (so the sentences are in an order than may span several documents)
    * train with larger batches
    * use BPE with bytes as a subunit and not characters (because of unicode characters)
- `CamemBERT <camembert>`__ is a wrapper around RoBERTa. Refer to this page for usage examples.

This model was contributed by `julien-c <https://huggingface.co/julien-c>`__. The original code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/roberta>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the RoBERTa model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``RobertaModel`` or ``TFRobertaModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``RobertaModel`` or ``TFRobertaModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.roberta_prelayernorm.configuration_roberta_prelayernorm.RobertaPreLayerNormConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The RoBERTa-PreLayerNorm model was proposed in `fairseq: A Fast, Extensible Toolkit for Sequence Modeling <https://arxiv.org/abs/1904.01038>`__ by Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli.
It is identical to using the `--encoder-normalize-before` flag in `fairseq <https://fairseq.readthedocs.io/>`__.

The abstract from the paper is the following:

*fairseq is an open-source sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling, and other text generation tasks. The toolkit is based on PyTorch and supports distributed training across multiple GPUs and machines. We also support fast mixed-precision training and inference on modern GPUs.*

Tips:

- The implementation is the same as `Roberta <roberta>`__ except instead of using _Add and Norm_ it does _Norm and Add_. _Add_ and _Norm_ refers to the Addition and LayerNormalization as described in `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`__.
- This is identical to using the `--encoder-normalize-before` flag in `fairseq <https://fairseq.readthedocs.io/>`__.

This model was contributed by `andreasmaden <https://huggingface.co/andreasmaden>`__.
The original code can be found `here <https://github.com/princeton-nlp/DinkyTrain>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the RoBERTa-PreLayerNorm model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``RobertaPreLayerNormModel`` or ``TFRobertaPreLayerNormModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``RobertaPreLayerNormModel`` or ``TFRobertaPreLayerNormModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.roc_bert.configuration_roc_bert.RoCBertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, use_cache=True, pad_token_id=0, position_embedding_type='absolute', classifier_dropout=None, enable_pronunciation=True, enable_shape=True, pronunciation_embed_dim=768, pronunciation_vocab_size=910, shape_embed_dim=512, shape_vocab_size=24858, concat_input=True, **kwargs)

The RoCBert model was proposed in `RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining <https://aclanthology.org/2022.acl-long.65.pdf>`__  by HuiSu, WeiweiShi, XiaoyuShen, XiaoZhou, TuoJi, JiaruiFang, JieZhou.
It's a pretrained Chinese language model that is robust under various forms of adversarial attacks.

The abstract from the paper is the following:

*Large-scale pretrained language models have achieved SOTA results on NLP tasks. However, they have been shown
vulnerable to adversarial attacks especially for logographic languages like Chinese. In this work, we propose
ROCBERT: a pretrained Chinese Bert that is robust to various forms of adversarial attacks like word perturbation,
synonyms, typos, etc. It is pretrained with the contrastive learning objective which maximizes the label consistency
under different synthesized adversarial examples. The model takes as input multimodal information including the
semantic, phonetic and visual features. We show all these features are important to the model robustness since the
attack can be performed in all the three forms. Across 5 Chinese NLU tasks, ROCBERT outperforms strong baselines under
three blackbox adversarial algorithms without sacrificing the performance on clean testset. It also performs the best
in the toxic content detection task under human-made attacks.*

This model was contributed by `weiweishi <https://huggingface.co/weiweishi>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the RoCBert model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``RoCBertModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``RoCBertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.
    enable_pronunciation (`bool`, *optional*, defaults to `True`):
        Whether or not the model use pronunciation embed when training.
    enable_shape (`bool`, *optional*, defaults to `True`):
        Whether or not the model use shape embed when training.
    pronunciation_embed_dim (`int`, *optional*, defaults to 768):
        Dimension of the pronunciation_embed.
    pronunciation_vocab_size (`int`, *optional*, defaults to 910):
        Pronunciation Vocabulary size of the RoCBert model. Defines the number of different tokens that can be
        represented by the `input_pronunciation_ids` passed when calling ``RoCBertModel``.
    shape_embed_dim (`int`, *optional*, defaults to 512):
        Dimension of the shape_embed.
    shape_vocab_size (`int`, *optional*, defaults to 24858):
        Shape Vocabulary size of the RoCBert model. Defines the number of different tokens that can be represented
        by the `input_shape_ids` passed when calling ``RoCBertModel``.
    concat_input (`bool`, *optional*, defaults to `True`):
        Defines the way of merging the shape_embed, pronunciation_embed and word_embed, if the value is true,
        output_embed = torch.cat((word_embed, shape_embed, pronunciation_embed), -1), else output_embed =
        (word_embed + shape_embed + pronunciation_embed) / 3

.. class:: transformers.models.roformer.configuration_roformer.RoFormerConfig(vocab_size=50000, embedding_size=None, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=1536, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, rotary_value=False, use_cache=True, **kwargs)

The RoFormer model was proposed in `RoFormer: Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/pdf/2104.09864v1.pdf>`__ by Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu.

The abstract from the paper is the following:

*Position encoding in transformer architecture provides supervision for dependency modeling between elements at
different positions in the sequence. We investigate various methods to encode positional information in
transformer-based language models and propose a novel implementation named Rotary Position Embedding(RoPE). The
proposed RoPE encodes absolute positional information with rotation matrix and naturally incorporates explicit relative
position dependency in self-attention formulation. Notably, RoPE comes with valuable properties such as flexibility of
being expand to any sequence lengths, decaying inter-token dependency with increasing relative distances, and
capability of equipping the linear self-attention with relative position encoding. As a result, the enhanced
transformer with rotary position embedding, or RoFormer, achieves superior performance in tasks with long texts. We
release the theoretical analysis along with some preliminary experiment results on Chinese data. The undergoing
experiment for English benchmark will soon be updated.*

Tips:

- RoFormer is a BERT-like autoencoding model with rotary position embeddings. Rotary position embeddings have shown
  improved performance on classification tasks with long texts.


This model was contributed by `junnyu <https://huggingface.co/junnyu>`__. The original code can be found `here <https://github.com/ZhuiyiTechnology/roformer>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50000):
        Vocabulary size of the RoFormer model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``RoFormerModel`` or ``TFRoFormerModel``.
    embedding_size (`int`, *optional*, defaults to None):
        Dimensionality of the encoder layers and the pooler layer. Defaults to the `hidden_size` if not provided.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 1536):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 1536).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``RoFormerModel`` or ``TFRoFormerModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    rotary_value (`bool`, *optional*, defaults to `False`):
        Whether or not apply rotary position embeddings on value layer.

.. class:: transformers.models.rwkv.configuration_rwkv.RwkvConfig(vocab_size=50277, context_length=1024, hidden_size=4096, num_hidden_layers=32, attention_hidden_size=None, intermediate_size=None, layer_norm_epsilon=1e-05, bos_token_id=0, eos_token_id=0, rescale_every=6, tie_word_embeddings=False, use_cache=True, **kwargs)

The RWKV model was proposed in `this repo <https://github.com/BlinkDL/RWKV-LM>`__

It suggests a tweak in the traditional Transformer attention to make it linear. This way, the model can be used as recurrent network: passing inputs for timestamp 0 and timestamp 1 together is the same as passing inputs at timestamp 0, then inputs at timestamp 1 along with the state of timestamp 0 (see example below).

This can be more efficient than a regular Transformer and can deal with sentence of any length (even if the model uses a fixed context length for training).

This model was contributed by `sgugger <https://huggingface.co/sgugger>`__.
The original code can be found `here <https://github.com/BlinkDL/RWKV-LM>`__.

Example of use as an RNN:

```py
import torch
from transformers import AutoTokenizer, RwkvConfig, RwkvModel

model = RwkvModel.from_pretrained("sgugger/rwkv-430M-pile")
tokenizer = AutoTokenizer.from_pretrained("sgugger/rwkv-430M-pile")

inputs = tokenizer("This is an example.", return_tensors="pt")
# Feed everything to the model
outputs = model(inputs["input_ids"])
output_whole = outputs.last_hidden_state

outputs = model(inputs["input_ids"][:, :2])
output_one = outputs.last_hidden_state

# Using the state computed on the first inputs, we will get the same output
outputs = model(inputs["input_ids"][:, 2:], state=outputs.state)
output_two = outputs.last_hidden_state

torch.allclose(torch.cat([output_one, output_two], dim=1), output_whole, atol=1e-5)
```

Args:
    vocab_size (`int`, *optional*, defaults to 50277):
        Vocabulary size of the RWKV model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``RwkvModel``.
    context_length (`int`, *optional*, defaults to 1024):
        The maximum sequence length that this model can be be used with in a single forward (using it in RNN mode
        lets use any sequence length).
    hidden_size (`int`, *optional*, defaults to 4096):
        Dimensionality of the embeddings and hidden states.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the model.
    attention_hidden_size (`int`, *optional*):
        Dimensionality of the attention hidden states. Will default to `hidden_size` if unset.
    intermediate_size (`int`, *optional*):
        Dimensionality of the inner feed-forward layers. Will default to 4 times `hidden_size` if unset.
    layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
        The epsilon to use in the layer normalization layers.
    bos_token_id (`int`, *optional*, defaults to 0):
        The id of the beginning of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer
        as GPTNeoX.
    eos_token_id (`int`, *optional*, defaults to 0):
        The id of the end of sentence token in the vocabulary. Defaults to 0 as RWKV uses the same tokenizer as
        GPTNeoX.
    rescale_every (`int`, *optional*, defaults to 6):
        At inference, the hidden states (and weights of the correponding output layers) are divided by 2 every
        `rescale_every` layer. If set to 0 or a negative number, no rescale is done.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether or not to tie the word embeddings with the input token embeddings.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last state.

.. class:: transformers.models.splinter.configuration_splinter.SplinterConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, use_cache=True, pad_token_id=0, question_token_id=104, **kwargs)

The Splinter model was proposed in `Few-Shot Question Answering by Pretraining Span Selection <https://arxiv.org/abs/2101.00438>`__ by Ori Ram, Yuval Kirstain, Jonathan Berant, Amir Globerson, Omer Levy. Splinter
is an encoder-only transformer (similar to BERT) pretrained using the recurring span selection task on a large corpus
comprising Wikipedia and the Toronto Book Corpus.

The abstract from the paper is the following:

In several question answering benchmarks, pretrained models have reached human parity through fine-tuning on an order
of 100,000 annotated questions and answers. We explore the more realistic few-shot setting, where only a few hundred
training examples are available, and observe that standard models perform poorly, highlighting the discrepancy between
current pretraining objectives and question answering. We propose a new pretraining scheme tailored for question
answering: recurring span selection. Given a passage with multiple sets of recurring spans, we mask in each set all
recurring spans but one, and ask the model to select the correct span in the passage for each masked span. Masked spans
are replaced with a special token, viewed as a question representation, that is later used during fine-tuning to select
the answer span. The resulting model obtains surprisingly good results on multiple benchmarks (e.g., 72.7 F1 on SQuAD
with only 128 training examples), while maintaining competitive performance in the high-resource setting.

Tips:

- Splinter was trained to predict answers spans conditioned on a special `QUESTION] token. These tokens contextualize
  to question representations which are used to predict the answers. This layer is called QASS, and is the default
  behaviour in the ``SplinterForQuestionAnswering`` class. Therefore:
- Use ``SplinterTokenizer`` (rather than ``BertTokenizer``), as it already
  contains this special token. Also, its default behavior is to use this token when two sequences are given (for
  example, in the *run_qa.py* script).
- If you plan on using Splinter outside *run_qa.py*, please keep in mind the question token - it might be important for
  the success of your model, especially in a few-shot setting.
- Please note there are two different checkpoints for each size of Splinter. Both are basically the same, except that
  one also has the pretrained weights of the QASS layer (*tau/splinter-base-qass* and *tau/splinter-large-qass*) and one
  doesn't (*tau/splinter-base* and *tau/splinter-large*). This is done to support randomly initializing this layer at
  fine-tuning, as it is shown to yield better results for some cases in the paper.

This model was contributed by [yuvalkirstain <https://huggingface.co/yuvalkirstain>`__ and `oriram <https://huggingface.co/oriram>`__. The original code can be found `here <https://github.com/oriram/splinter>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the Splinter model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``SplinterModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``SplinterModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    question_token_id (`int`, *optional*, defaults to 104):
        The id of the `[QUESTION]` token.

.. class:: transformers.models.squeezebert.configuration_squeezebert.SqueezeBertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, embedding_size=768, q_groups=4, k_groups=4, v_groups=4, post_attention_groups=1, intermediate_groups=4, output_groups=4, **kwargs)

The SqueezeBERT model was proposed in `SqueezeBERT: What can computer vision teach NLP about efficient neural networks? <https://arxiv.org/abs/2006.11316>`__ by Forrest N. Iandola, Albert E. Shaw, Ravi Krishna, Kurt W. Keutzer. It's a
bidirectional transformer similar to the BERT model. The key difference between the BERT architecture and the
SqueezeBERT architecture is that SqueezeBERT uses `grouped convolutions <https://blog.yani.io/filter-group-tutorial>`__
instead of fully-connected layers for the Q, K, V and FFN layers.

The abstract from the paper is the following:

*Humans read and write hundreds of billions of messages every day. Further, due to the availability of large datasets,
large computing systems, and better neural network models, natural language processing (NLP) technology has made
significant strides in understanding, proofreading, and organizing these messages. Thus, there is a significant
opportunity to deploy NLP in myriad applications to help web users, social networks, and businesses. In particular, we
consider smartphones and other mobile devices as crucial platforms for deploying NLP models at scale. However, today's
highly-accurate NLP neural network models such as BERT and RoBERTa are extremely computationally expensive, with
BERT-base taking 1.7 seconds to classify a text snippet on a Pixel 3 smartphone. In this work, we observe that methods
such as grouped convolutions have yielded significant speedups for computer vision networks, but many of these
techniques have not been adopted by NLP neural network designers. We demonstrate how to replace several operations in
self-attention layers with grouped convolutions, and we use this technique in a novel network architecture called
SqueezeBERT, which runs 4.3x faster than BERT-base on the Pixel 3 while achieving competitive accuracy on the GLUE test
set. The SqueezeBERT code will be released.*

Tips:

- SqueezeBERT is a model with absolute position embeddings so it's usually advised to pad the inputs on the right
  rather than the left.
- SqueezeBERT is similar to BERT and therefore relies on the masked language modeling (MLM) objective. It is therefore
  efficient at predicting masked tokens and at NLU in general, but is not optimal for text generation. Models trained
  with a causal language modeling (CLM) objective are better in that regard.
- For best results when finetuning on sequence classification tasks, it is recommended to start with the
  *squeezebert/squeezebert-mnli-headless* checkpoint.

This model was contributed by `forresti <https://huggingface.co/forresti>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the SqueezeBERT model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``SqueezeBertModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``BertModel`` or ``TFBertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):

    pad_token_id (`int`, *optional*, defaults to 0):
        The ID of the token in the word embedding to use as padding.
    embedding_size (`int`, *optional*, defaults to 768):
        The dimension of the word embedding vectors.

    q_groups (`int`, *optional*, defaults to 4):
        The number of groups in Q layer.
    k_groups (`int`, *optional*, defaults to 4):
        The number of groups in K layer.
    v_groups (`int`, *optional*, defaults to 4):
        The number of groups in V layer.
    post_attention_groups (`int`, *optional*, defaults to 1):
        The number of groups in the first feed forward network layer.
    intermediate_groups (`int`, *optional*, defaults to 4):
        The number of groups in the second feed forward network layer.
    output_groups (`int`, *optional*, defaults to 4):
        The number of groups in the third feed forward network layer.

.. class:: transformers.models.stablelm.configuration_stablelm.StableLmConfig(vocab_size=50304, intermediate_size=6912, hidden_size=2560, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32, hidden_act='silu', max_position_embeddings=4096, initializer_range=0.02, layer_norm_eps=1e-05, use_cache=True, tie_word_embeddings=False, rope_theta=10000, rope_scaling=None, use_qkv_bias=False, hidden_dropout=0.0, attention_dropout=0.0, partial_rotary_factor=0.25, bos_token_id=0, eos_token_id=0, **kwargs)

`StableLM 3B 4E1T` was proposed in ``StableLM 3B 4E1T`: Technical Report <https://stability.wandb.io/stability-llm/stable-lm/reports/StableLM-3B-4E1T--VmlldzoyMjU4?accessToken=u3zujipenkx5g7rtcj9qojjgxpconyjktjkli2po09nffrffdhhchq045vp0wyfo>`__ by Stability AI and is the first model in a series of multi-epoch pre-trained language models.

#Args:
    vocab_size (`int`, *optional*, defaults to 50304):
        Vocabulary size of the StableLM model. Defines the number of different tokens that
        can be represented by the `inputs_ids` passed when calling ``StableLmModel``.
    intermediate_size (`int`, *optional*, defaults to 6912):
        Dimension of the MLP representations.
    hidden_size (`int`, *optional*, defaults to 2560):
        Number of hidden layers in the Transformer decoder.
    num_hidden_layers (`int`, *optional*, defaults to 32):
        Number of hidden layers in the Transformer decoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer encoder.
    num_key_value_heads (`int`, *optional*, defaults to 32):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to
        `num_attention_heads`.
    hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        The non-linear activation function (function or string).
    max_position_embeddings (`int`, *optional*, defaults to 4096):
        The maximum sequence length that this model might ever be used with.
        Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing
         all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-05):
        The epsilon used by the normalization layers.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions
        (not used by all models). Only relevant if `config.is_decoder=True`.
    tie_word_embeddings (`bool`, *optional*, defaults to `False`):
        Whether the model's input and output word embeddings should be tied.
    rope_theta (`float`, *optional*, defaults to `10000.0`):
        The base period of the RoPE embeddings.
    rope_scaling (`Dict`, *optional*):
        Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
        strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
        `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
        `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
        these scaling strategies behave:
        https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This
        is an experimental feature, subject to breaking API changes in future versions.
    use_qkv_bias (`bool`, *optional*, defaults to `False`):
        Whether or not the model should use bias for qkv layers.
    hidden_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio after applying the MLP to the hidden states.
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    partial_rotary_factor (`float`, *optional*, defaults to 0.25):
        Percentage of the query and keys which will have rotary embedding.
    bos_token_id (int, *optional*, defaults to 0):
        The id of the `BOS` token in the vocabulary.
    eos_token_id (int, *optional*, defaults to 0):
        The id of the `EOS` token in the vocabulary.

.. class:: transformers.models.starcoder2.configuration_starcoder2.Starcoder2Config(vocab_size=49152, hidden_size=3072, intermediate_size=12288, num_hidden_layers=30, num_attention_heads=24, num_key_value_heads=2, hidden_act='gelu_pytorch_tanh', max_position_embeddings=4096, initializer_range=0.018042, norm_epsilon=1e-05, use_cache=True, bos_token_id=50256, eos_token_id=50256, rope_theta=10000.0, sliding_window=None, attention_dropout=0.0, residual_dropout=0.0, embedding_dropout=0.0, use_bias=True, **kwargs)

StarCoder2 is a family of open LLMs for code and comes in 3 different sizes with 3B, 7B and 15B parameters. The flagship StarCoder2-15B model is trained on over 4 trillion tokens and 600+ programming languages from The Stack v2. All models use Grouped Query Attention, a context window of 16,384 tokens with a sliding window attention of 4,096 tokens, and were trained using the Fill-in-the-Middle objective. The models have been released with the paper `StarCoder 2 and The Stack v2: The Next Generation <https://arxiv.org/abs/2402.19173>`__ by Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, Tianyang Liu, Max Tian, Denis Kocetkov, Arthur Zucker, Younes Belkada, Zijian Wang, Qian Liu, Dmitry Abulkhanov, Indraneil Paul, Zhuang Li, Wen-Ding Li, Megan Risdal, Jia Li, Jian Zhu, Terry Yue Zhuo, Evgenii Zheltonozhskii, Nii Osae Osae Dade, Wenhao Yu, Lucas Krauß, Naman Jain, Yixuan Su, Xuanli He, Manan Dey, Edoardo Abati, Yekun Chai, Niklas Muennighoff, Xiangru Tang, Muhtasham Oblokulov, Christopher Akiki, Marc Marone, Chenghao Mou, Mayank Mishra, Alex Gu, Binyuan Hui, Tri Dao, Armel Zebaze, Olivier Dehaene, Nicolas Patry, Canwen Xu, Julian McAuley, Han Hu, Torsten Scholak, Sebastien Paquet, Jennifer Robinson, Carolyn Jane Anderson, Nicolas Chapados, Mostofa Patwary, Nima Tajbakhsh, Yacine Jernite, Carlos Muñoz Ferrandis, Lingming Zhang, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries.

The abstract of the paper is the following:

> The BigCode project, an open-scientific collaboration focused on the responsible development of Large Language Models for Code (Code LLMs), introduces StarCoder2. In partnership with Software Heritage (SWH), we build The Stack v2 on top of the digital commons of their source code archive. Alongside the SWH repositories spanning 619 programming languages, we carefully select other high-quality data sources, such as GitHub pull requests, Kaggle notebooks, and code documentation. This results in a training set that is 4x larger than the first StarCoder dataset. We train StarCoder2 models with 3B, 7B, and 15B parameters on 3.3 to 4.3 trillion tokens and thoroughly evaluate them on a comprehensive set of Code LLM benchmarks. We find that our small model, StarCoder2-3B, outperforms other Code LLMs of similar size on most benchmarks, and also outperforms StarCoderBase-15B. Our large model, StarCoder2- 15B, significantly outperforms other models of comparable size. In addition, it matches or outperforms CodeLlama-34B, a model more than twice its size. Although DeepSeekCoder- 33B is the best-performing model at code completion for high-resource languages, we find that StarCoder2-15B outperforms it on math and code reasoning benchmarks, as well as several low-resource languages. We make the model weights available under an OpenRAIL license and ensure full transparency regarding the training data by releasing the SoftWare Heritage persistent IDentifiers (SWHIDs) of the source code data.
Args:
    vocab_size (`int`, *optional*, defaults to 49152):
        Vocabulary size of the Starcoder2 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``Starcoder2Model``
    hidden_size (`int`, *optional*, defaults to 3072):
        Dimension of the hidden representations.
    intermediate_size (`int`, *optional*, defaults to 12288):
        Dimension of the MLP representations.
    num_hidden_layers (`int`, *optional*, defaults to 30):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 24):
        Number of attention heads for each attention layer in the Transformer encoder.
    num_key_value_heads (`int`, *optional*, defaults to 2):
        This is the number of key_value heads that should be used to implement Grouped Query Attention. If
        `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
        `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
        converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
        by meanpooling all the original heads within that group. For more details checkout `this
        paper <https://arxiv.org/pdf/2305.13245.pdf>`__. If it is not specified, will default to `8`.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
        The non-linear activation function (function or string) in the decoder.
    max_position_embeddings (`int`, *optional*, defaults to 4096):
        The maximum sequence length that this model might ever be used with. Starcoder2's sliding window attention
        allows sequence of up to 4096*32 tokens.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    norm_epsilon (`float`, *optional*, defaults to 1e-05):
        Epsilon value for the layer norm
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    bos_token_id (`int`, *optional*, defaults to 50256):
        The id of the "beginning-of-sequence" token.
    eos_token_id (`int`, *optional*, defaults to 50256):
        The id of the "end-of-sequence" token.
    rope_theta (`float`, *optional*, defaults to 10000.0):
        The base period of the RoPE embeddings.
    sliding_window (`int`, *optional*):
        Sliding window attention window size. If not specified, will default to `None` (no sliding window).
    attention_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for the attention probabilities.
    residual_dropout (`float`, *optional*, defaults to 0.0):
        Residual connection dropout value.
    embedding_dropout (`float`, *optional*, defaults to 0.0):
        Embedding dropout.
    use_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias term on linear layers of the model.


::

>>> from transformers import Starcoder2Model, Starcoder2Config

>>> # Initializing a Starcoder2 7B style configuration
>>> configuration = Starcoder2Config()

>>> # Initializing a model from the Starcoder2 7B style configuration
>>> model = Starcoder2Model(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config

.. class:: transformers.models.switch_transformers.configuration_switch_transformers.SwitchTransformersConfig(vocab_size=32128, d_model=768, d_kv=64, d_ff=2048, expert_capacity=64, num_layers=12, num_sparse_encoder_layers=3, num_decoder_layers=12, num_sparse_decoder_layers=3, num_heads=12, num_experts=8, router_bias=False, router_jitter_noise=0.01, router_dtype='float32', router_ignore_padding_tokens=False, relative_attention_num_buckets=32, relative_attention_max_distance=128, dropout_rate=0.1, layer_norm_epsilon=1e-06, router_z_loss_coef=0.001, router_aux_loss_coef=0.001, initializer_factor=1.0, dense_act_fn='relu', is_encoder_decoder=True, add_router_probs=False, use_cache=True, pad_token_id=0, eos_token_id=1, **kwargs)

The SwitchTransformers model was proposed in `Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity <https://arxiv.org/abs/2101.03961>`__ by William Fedus, Barret Zoph, Noam Shazeer.

The Switch Transformer model uses a sparse T5 encoder-decoder architecture, where the MLP are replaced by a Mixture of Experts (MoE). A routing mechanism (top 1 in this case) associates each token to one of the expert, where each expert is a dense MLP. While switch transformers have a lot more weights than their equivalent dense models, the sparsity allows better scaling and better finetuning performance at scale.
During a forward pass, only a fraction of the weights are used. The routing mechanism allows the model to select relevant weights on the fly which increases the model capacity without increasing the number of operations.


The abstract from the paper is the following:

*In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) defies this and instead selects different parameters for each incoming example. The result is a sparsely-activated model -- with outrageous numbers of parameters -- but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs and training instability -- we address these with the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats. We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus" and achieve a 4x speedup over the T5-XXL model.*

Tips:

- SwitchTransformers uses the ``T5Tokenizer``, which can be loaded directly from each model's repository.
- The released weights are pretrained on English `Masked Language Modeling <https://moon-ci-docs.huggingface.co/docs/transformers/pr_19323/en/glossary#general-terms>`__ task, and should be finetuned.

This model was contributed by `Younes Belkada <https://huggingface.co/ybelkada>`__ and `Arthur Zucker <https://huggingface.co/ArtZucker>`__ .
The original code can be found `here <https://github.com/google/flaxformer/tree/main/flaxformer/architectures/moe>`__.

Arguments:
    vocab_size (`int`, *optional*, defaults to 32128):
        Vocabulary size of the SwitchTransformers model. Defines the number of different tokens that can be
        represented by the `inputs_ids` passed when calling ``SwitchTransformersModel``.
    d_model (`int`, *optional*, defaults to 768):
        Size of the encoder layers and the pooler layer.
    d_kv (`int`, *optional*, defaults to 64):
        Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
        num_heads`.
    d_ff (`int`, *optional*, defaults to 2048):
        Size of the intermediate feed forward layer in each `SwitchTransformersBlock`.
    expert_capacity (`int`, *optional*, defaults to 64):
        Number of tokens that can be stored in each expert. If set to 1, the model will behave like a regular
        Transformer.
    num_layers (`int`, *optional*, defaults to 12):
        Number of dense hidden layers in the Transformer encoder layer.
    num_sparse_encoder_layers (`int`, *optional*, defaults to 3):
        Number of sparse (MoE) dense hidden layers in the Transformer encoder layer.
    num_decoder_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
    num_sparse_decoder_layers (`int`, *optional*, defaults to 3):
        Number of sparse (MoE) dense hidden layers in the Transformer decoder layer.
    num_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    num_experts (`int`, *optional*, defaults to 8):
        Number of experts for each SwitchTransformer layer.
    router_bias (`bool`, *optional*, defaults to `False`):
        Whether to add a bias to the router.
    router_jitter_noise (`float`, *optional*, defaults to 0.01):
        Amount of noise to add to the router.
    router_dtype (`str`, *optional*, default to `"float32"`):
        The `dtype` used for the routers. It is preferable to keep the `dtype` to `"float32"` as specified in the
        *selective precision* discussion in `the paper <https://arxiv.org/abs/2101.03961>`__.
    router_ignore_padding_tokens (`bool`, *optional*, defaults to `False`):
        Whether to ignore padding tokens when routing.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    dropout_rate (`float`, *optional*, defaults to 0.1):
        The ratio for all dropout layers.
    layer_norm_eps (`float`, *optional*, defaults to 1e-6):
        The epsilon used by the layer normalization layers.
    router_z_loss_coef (`float`, *optional*, defaults to 0.001):
        The z loss factor for the total loss.
    router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
        The aux loss factor for the total loss.
    initializer_factor (`float`, *optional*, defaults to 1.0):
        A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
        testing).
    dense_act_fn (`string`, *optional*, defaults to `"relu"`):
        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. SwitchTransformersv1.1
        uses the `"gated-gelu"` feed forward projection. Original SwitchTransformers uses `"relu"`.
    add_router_probs (`bool`, *optional*, defaults to `False`):
        Whether to output router probabilities to compute router auxiliary loss.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.t5.configuration_t5.T5Config(vocab_size=32128, d_model=512, d_kv=64, d_ff=2048, num_layers=6, num_decoder_layers=None, num_heads=8, relative_attention_num_buckets=32, relative_attention_max_distance=128, dropout_rate=0.1, layer_norm_epsilon=1e-06, initializer_factor=1.0, feed_forward_proj='relu', is_encoder_decoder=True, use_cache=True, pad_token_id=0, eos_token_id=1, classifier_dropout=0.0, **kwargs)

The T5 model was presented in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer <https://arxiv.org/pdf/1910.10683.pdf>`__ by `Colin Raffel <https://huggingface.co/craffel>`__, Noam Shazeer, `Adam Roberts <https://huggingface.co/adarob>`__, Katherine Lee, Sharan Narang,
Michael Matena, Yanqi Zhou, Wei Li, `Peter J. Liu <https://huggingface.co/peterjliu>`__.

The abstract from the paper is the following:

*Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream
task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning
has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of
transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a
text-to-text format. Our systematic study compares pretraining objectives, architectures, unlabeled datasets, transfer
approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration
with scale and our new "Colossal Clean Crawled Corpus", we achieve state-of-the-art results on many benchmarks covering
summarization, question answering, text classification, and more. To facilitate future work on transfer learning for
NLP, we release our dataset, pre-trained models, and code.*

Tips:

- T5 is an encoder-decoder model pre-trained on a multi-task mixture of unsupervised and supervised tasks and for which
each task is converted into a text-to-text format. T5 works well on a variety of tasks out-of-the-box by prepending a
different prefix to the input corresponding to each task, e.g., for translation: *translate English to German: ...*,
for summarization: *summarize: ...*.
- The pretraining includes both supervised and self-supervised training. Supervised training is conducted on downstream tasks provided by the GLUE and SuperGLUE benchmarks (converting them into text-to-text tasks as explained above).
- Self-supervised training uses corrupted tokens, by randomly removing 15% of the tokens and replacing them with individual sentinel tokens (if several consecutive tokens are marked for removal, the whole group is replaced with a single sentinel token). The input of the encoder is the corrupted sentence, the input of the decoder is the original sentence and the target is then the dropped out tokens delimited by their sentinel tokens.

- T5 uses relative scalar embeddings. Encoder input padding can be done on the left and on the right.

- See the `training <#training>`__, `inference <#inference>`__ and `scripts <#scripts>`__ sections below for all details regarding usage.

T5 comes in different sizes:

- `t5-small <https://huggingface.co/t5-small>`__

- `t5-base <https://huggingface.co/t5-base>`__

- `t5-large <https://huggingface.co/t5-large>`__

- `t5-3b <https://huggingface.co/t5-3b>`__

- `t5-11b <https://huggingface.co/t5-11b>`__.

Based on the original T5 model, Google has released some follow-up works:

- **T5v1.1**: T5v1.1 is an improved version of T5 with some architectural tweaks, and is pre-trained on C4 only without
  mixing in the supervised tasks. Refer to the documentation of T5v1.1 which can be found `here <t5v1.1>`__.

- **mT5**: mT5 is a multilingual T5 model. It is pre-trained on the mC4 corpus, which includes 101 languages. Refer to
  the documentation of mT5 which can be found `here <mt5>`__.

- **byT5**: byT5 is a T5 model pre-trained on byte sequences rather than SentencePiece subword token sequences. Refer
  to the documentation of byT5 which can be found `here <byt5>`__.

- **UL2**: UL2 is a T5 like model pretrained on various denoising objectives

- **Flan-T5**: Flan is a pretraining methods that is based on prompting. The Flan-T5 are T5 models trained on the Flan collection of 
    datasets which include: `taskmaster2`, `djaym7/wiki_dialog`, `deepmind/code_contests`, `lambada`, `gsm8k`, `aqua_rat`, `esnli`, `quasc` and `qed`.

- **FLan-UL2** : the UL2 model finetuned using the "Flan" prompt tuning and dataset collection.

- **UMT5**: UmT5 is a multilingual T5 model trained on an improved and refreshed mC4 multilingual corpus,  29 trillion characters across 107 language, using a new sampling method, UniMax. Refer to
 the documentation of mT5 which can be found `here <umt5>`__.

All checkpoints can be found on the `hub <https://huggingface.co/models?search=t5>`__.

This model was contributed by `thomwolf <https://huggingface.co/thomwolf>`__. The original code can be found `here <https://github.com/google-research/text-to-text-transfer-transformer>`__.

<a id='training'></a>

Arguments:
    vocab_size (`int`, *optional*, defaults to 32128):
        Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``T5Model`` or ``TFT5Model``.
    d_model (`int`, *optional*, defaults to 512):
        Size of the encoder layers and the pooler layer.
    d_kv (`int`, *optional*, defaults to 64):
        Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
        be defined as `num_heads * d_kv`.
    d_ff (`int`, *optional*, defaults to 2048):
        Size of the intermediate feed forward layer in each `T5Block`.
    num_layers (`int`, *optional*, defaults to 6):
        Number of hidden layers in the Transformer encoder.
    num_decoder_layers (`int`, *optional*):
        Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
    num_heads (`int`, *optional*, defaults to 8):
        Number of attention heads for each attention layer in the Transformer encoder.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    dropout_rate (`float`, *optional*, defaults to 0.1):
        The ratio for all dropout layers.
    classifier_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for classifier.
    layer_norm_eps (`float`, *optional*, defaults to 1e-6):
        The epsilon used by the layer normalization layers.
    initializer_factor (`float`, *optional*, defaults to 1):
        A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
        testing).
    feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
        Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
        `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.visual_bert.configuration_visual_bert.VisualBertConfig(vocab_size=30522, hidden_size=768, visual_embedding_dim=512, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, bypass_transformer=False, special_visual_initialize=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The VisualBERT model was proposed in `VisualBERT: A Simple and Performant Baseline for Vision and Language <https://arxiv.org/pdf/1908.03557>`__ by Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang.
VisualBERT is a neural network trained on a variety of (image, text) pairs.

The abstract from the paper is the following:

*We propose VisualBERT, a simple and flexible framework for modeling a broad range of vision-and-language tasks.
VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an
associated input image with self-attention. We further propose two visually-grounded language model objectives for
pre-training VisualBERT on image caption data. Experiments on four vision-and-language tasks including VQA, VCR, NLVR2,
and Flickr30K show that VisualBERT outperforms or rivals with state-of-the-art models while being significantly
simpler. Further analysis demonstrates that VisualBERT can ground elements of language to image regions without any
explicit supervision and is even sensitive to syntactic relationships, tracking, for example, associations between
verbs and image regions corresponding to their arguments.*

Tips:

1. Most of the checkpoints provided work with the ``VisualBertForPreTraining`` configuration. Other
   checkpoints provided are the fine-tuned checkpoints for down-stream tasks - VQA ('visualbert-vqa'), VCR
   ('visualbert-vcr'), NLVR2 ('visualbert-nlvr2'). Hence, if you are not working on these downstream tasks, it is
   recommended that you use the pretrained checkpoints.

2. For the VCR task, the authors use a fine-tuned detector for generating visual embeddings, for all the checkpoints.
   We do not provide the detector and its weights as a part of the package, but it will be available in the research
   projects, and the states can be loaded directly into the detector provided.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the VisualBERT model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``VisualBertModel``. Vocabulary size of the model. Defines the
        different tokens that can be represented by the `inputs_ids` passed to the forward method of
        ``VisualBertModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    visual_embedding_dim (`int`, *optional*, defaults to 512):
        Dimensionality of the visual embeddings to be passed to the model.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``VisualBertModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    bypass_transformer (`bool`, *optional*, defaults to `False`):
        Whether or not the model should bypass the transformer for the visual embeddings. If set to `True`, the
        model directly concatenates the visual embeddings from ``VisualBertEmbeddings`` with text output from
        transformers, and then pass it to a self-attention layer.
    special_visual_initialize (`bool`, *optional*, defaults to `True`):
        Whether or not the visual token type and position type embedding weights should be initialized the same as
        the textual token type and positive type embeddings. When set to `True`, the weights of the textual token
        type and position type embeddings are copied to the respective visual embedding layers.

.. class:: transformers.models.xglm.configuration_xglm.XGLMConfig(vocab_size=256008, max_position_embeddings=2048, d_model=1024, ffn_dim=4096, num_layers=24, attention_heads=16, activation_function='gelu', dropout=0.1, attention_dropout=0.1, activation_dropout=0.0, layerdrop=0.0, init_std=0.02, scale_embedding=True, use_cache=True, decoder_start_token_id=2, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The XGLM model was proposed in `Few-shot Learning with Multilingual Language Models <https://arxiv.org/abs/2112.10668>`__
by Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, 
Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O'Horo, 
Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, Xian Li.

The abstract from the paper is the following:

*Large-scale autoregressive language models such as GPT-3 are few-shot learners that can perform a wide range of language 
tasks without fine-tuning. While these models are known to be able to jointly represent many different languages, 
their training data is dominated by English, potentially limiting their cross-lingual generalization. 
In this work, we train multilingual autoregressive language models on a balanced corpus covering a diverse set of languages, 
and study their few- and zero-shot learning capabilities in a wide range of tasks. Our largest model with 7.5 billion parameters 
sets new state of the art in few-shot learning in more than 20 representative languages, outperforming GPT-3 of comparable size 
in multilingual commonsense reasoning (with +7.4% absolute accuracy improvement in 0-shot settings and +9.4% in 4-shot settings) 
and natural language inference (+5.4% in each of 0-shot and 4-shot settings). On the FLORES-101 machine translation benchmark, 
our model outperforms GPT-3 on 171 out of 182 translation directions with 32 training examples, while surpassing the 
official supervised baseline in 45 directions. We present a detailed analysis of where the model succeeds and fails, 
showing in particular that it enables cross-lingual in-context learning on some tasks, while there is still room for improvement 
on surface form robustness and adaptation to tasks that do not have a natural cloze form. Finally, we evaluate our models 
in social value tasks such as hate speech detection in five languages and find it has limitations similar to comparable sized GPT-3 models.*


This model was contributed by `Suraj <https://huggingface.co/valhalla>`__. The original code can be found `here <https://github.com/pytorch/fairseq/tree/main/examples/xglm>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 256008):
        Vocabulary size of the XGLM model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``XGLMModel`` or ``FlaxXGLMModel``.
    max_position_embeddings (`int`, *optional*, defaults to 2048):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    d_model (`int`, *optional*, defaults to 1024):
        Dimension of the layers and the pooler layer.
    ffn_dim (`int`, *optional*, defaults to 4096):
        Dimension of the "intermediate" (often named feed-forward) layer in decoder.
    num_layers (`int`, *optional*, defaults to 24):
        Number of hidden layers Transformer decoder.
    attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, dencoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    layerdrop (`float`, *optional*, defaults to 0.0):
        The LayerDrop probability for the encoder. See the `LayerDrop paper <see https://arxiv.org/abs/1909.11556>`__
        for more details.
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    scale_embedding (`bool`, *optional*, defaults to `True`):
        Scale embeddings by diving by sqrt(d_model).
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.xlm.configuration_xlm.XLMConfig(vocab_size=30145, emb_dim=2048, n_layers=12, n_heads=16, dropout=0.1, attention_dropout=0.1, gelu_activation=True, sinusoidal_embeddings=False, causal=False, asm=False, n_langs=1, use_lang_emb=True, max_position_embeddings=512, embed_init_std=0.02209708691207961, layer_norm_eps=1e-12, init_std=0.02, bos_index=0, eos_index=1, pad_index=2, unk_index=3, mask_index=5, is_encoder=True, summary_type='first', summary_use_proj=True, summary_activation=None, summary_proj_to_labels=True, summary_first_dropout=0.1, start_n_top=5, end_n_top=5, mask_token_id=0, lang_id=0, pad_token_id=2, bos_token_id=0, **kwargs)

The XLM model was proposed in `Cross-lingual Language Model Pretraining <https://arxiv.org/abs/1901.07291>`__ by
Guillaume Lample, Alexis Conneau. It's a transformer pretrained using one of the following objectives:

- a causal language modeling (CLM) objective (next token prediction),
- a masked language modeling (MLM) objective (BERT-like), or
- a Translation Language Modeling (TLM) object (extension of BERT's MLM to multiple language inputs)

The abstract from the paper is the following:

*Recent studies have demonstrated the efficiency of generative pretraining for English natural language understanding.
In this work, we extend this approach to multiple languages and show the effectiveness of cross-lingual pretraining. We
propose two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual
data, and one supervised that leverages parallel data with a new cross-lingual language model objective. We obtain
state-of-the-art results on cross-lingual classification, unsupervised and supervised machine translation. On XNLI, our
approach pushes the state of the art by an absolute gain of 4.9% accuracy. On unsupervised machine translation, we
obtain 34.3 BLEU on WMT'16 German-English, improving the previous state of the art by more than 9 BLEU. On supervised
machine translation, we obtain a new state of the art of 38.5 BLEU on WMT'16 Romanian-English, outperforming the
previous best approach by more than 4 BLEU. Our code and pretrained models will be made publicly available.*

Tips:

- XLM has many different checkpoints, which were trained using different objectives: CLM, MLM or TLM. Make sure to
  select the correct objective for your task (e.g. MLM checkpoints are not suitable for generation).
- XLM has multilingual checkpoints which leverage a specific `lang` parameter. Check out the `multi-lingual <../multilingual>`__ page for more information.
- A transformer model trained on several languages. There are three different type of training for this model and the library provides checkpoints for all of them:

    * Causal language modeling (CLM) which is the traditional autoregressive training (so this model could be in the previous section as well). One of the languages is selected for each training sample, and the model input is a sentence of 256 tokens, that may span over several documents in one of those languages.
    * Masked language modeling (MLM) which is like RoBERTa. One of the languages is selected for each training sample, and the model input is a sentence of 256 tokens, that may span over several documents in one of those languages, with dynamic masking of the tokens.
    * A combination of MLM and translation language modeling (TLM). This consists of concatenating a sentence in two different languages, with random masking. To predict one of the masked tokens, the model can use both, the surrounding context in language 1 and the context given by language 2.

This model was contributed by `thomwolf <https://huggingface.co/thomwolf>`__. The original code can be found `here <https://github.com/facebookresearch/XLM/>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30145):
        Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``XLMModel`` or ``TFXLMModel``.
    emb_dim (`int`, *optional*, defaults to 2048):
        Dimensionality of the encoder layers and the pooler layer.
    n_layer (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for the attention mechanism
    gelu_activation (`bool`, *optional*, defaults to `True`):
        Whether or not to use *gelu* for the activations instead of *relu*.
    sinusoidal_embeddings (`bool`, *optional*, defaults to `False`):
        Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.
    causal (`bool`, *optional*, defaults to `False`):
        Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in
        order to only attend to the left-side context instead if a bidirectional context.
    asm (`bool`, *optional*, defaults to `False`):
        Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction
        layer.
    n_langs (`int`, *optional*, defaults to 1):
        The number of languages the model handles. Set to 1 for monolingual models.
    use_lang_emb (`bool`, *optional*, defaults to `True`)
        Whether to use language embeddings. Some models use additional language embeddings, see `the multilingual
        models page <http://huggingface.co/transformers/multilingual.html#xlm-language-embeddings>`__ for information
        on how to use them.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    embed_init_std (`float`, *optional*, defaults to 2048^-0.5):
        The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.
    init_std (`int`, *optional*, defaults to 50257):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the
        embedding matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    bos_index (`int`, *optional*, defaults to 0):
        The index of the beginning of sentence token in the vocabulary.
    eos_index (`int`, *optional*, defaults to 1):
        The index of the end of sentence token in the vocabulary.
    pad_index (`int`, *optional*, defaults to 2):
        The index of the padding token in the vocabulary.
    unk_index (`int`, *optional*, defaults to 3):
        The index of the unknown token in the vocabulary.
    mask_index (`int`, *optional*, defaults to 5):
        The index of the masking token in the vocabulary.
    is_encoder(`bool`, *optional*, defaults to `True`):
        Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.
    summary_type (`string`, *optional*, defaults to "first"):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Has to be one of the following options:

            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`bool`, *optional*, defaults to `True`):
        Used in the sequence classification and multiple choice models.

        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_first_dropout (`float`, *optional*, defaults to 0.1):
        Used in the sequence classification and multiple choice models.

        The dropout ratio to be used after the projection and activation.
    start_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    end_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    mask_token_id (`int`, *optional*, defaults to 0):
        Model agnostic parameter to identify masked tokens when generating text in an MLM context.
    lang_id (`int`, *optional*, defaults to 1):
        The ID of the language used by the model. This parameter is used when generating text in a given language.

.. class:: transformers.models.xlm_prophetnet.configuration_xlm_prophetnet.XLMProphetNetConfig(activation_dropout: Optional[float] = 0.1, activation_function: Union[str, Callable, NoneType] = 'gelu', vocab_size: Optional[int] = 30522, hidden_size: Optional[int] = 1024, encoder_ffn_dim: Optional[int] = 4096, num_encoder_layers: Optional[int] = 12, num_encoder_attention_heads: Optional[int] = 16, decoder_ffn_dim: Optional[int] = 4096, num_decoder_layers: Optional[int] = 12, num_decoder_attention_heads: Optional[int] = 16, attention_dropout: Optional[float] = 0.1, dropout: Optional[float] = 0.1, max_position_embeddings: Optional[int] = 512, init_std: Optional[float] = 0.02, is_encoder_decoder: Optional[bool] = True, add_cross_attention: Optional[bool] = True, decoder_start_token_id: Optional[int] = 0, ngram: Optional[int] = 2, num_buckets: Optional[int] = 32, relative_max_distance: Optional[int] = 128, disable_ngram_loss: Optional[bool] = False, eps: Optional[float] = 0.0, use_cache: Optional[bool] = True, pad_token_id: Optional[int] = 0, bos_token_id: Optional[int] = 1, eos_token_id: Optional[int] = 2, **kwargs)

The XLM-ProphetNet model was proposed in `ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training, <https://arxiv.org/abs/2001.04063>`__ by Yu Yan, Weizhen Qi, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei
Zhang, Ming Zhou on 13 Jan, 2020.

XLM-ProphetNet is an encoder-decoder model and can predict n-future tokens for "ngram" language modeling instead of
just the next token. Its architecture is identical to ProhpetNet, but the model was trained on the multi-lingual
"wiki100" Wikipedia dump.

The abstract from the paper is the following:

*In this paper, we present a new sequence-to-sequence pretraining model called ProphetNet, which introduces a novel
self-supervised objective named future n-gram prediction and the proposed n-stream self-attention mechanism. Instead of
the optimization of one-step ahead prediction in traditional sequence-to-sequence model, the ProphetNet is optimized by
n-step ahead prediction which predicts the next n tokens simultaneously based on previous context tokens at each time
step. The future n-gram prediction explicitly encourages the model to plan for the future tokens and prevent
overfitting on strong local correlations. We pre-train ProphetNet using a base scale dataset (16GB) and a large scale
dataset (160GB) respectively. Then we conduct experiments on CNN/DailyMail, Gigaword, and SQuAD 1.1 benchmarks for
abstractive summarization and question generation tasks. Experimental results show that ProphetNet achieves new
state-of-the-art results on all these datasets compared to the models using the same scale pretraining corpus.*

The Authors' code can be found `here <https://github.com/microsoft/ProphetNet>`__.

Tips:

- XLM-ProphetNet's model architecture and pretraining objective is same as ProphetNet, but XLM-ProphetNet was pre-trained on the cross-lingual dataset XGLUE.

Args:
    activation_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for activations inside the fully connected layer.
    activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the ProphetNET model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``XLMProphetNetModel``.
    hidden_size (`int`, *optional*, defaults to 1024):
        Dimensionality of the layers and the pooler layer.
    encoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
    num_encoder_layers (`int`, *optional*, defaults to 12):
        Number of encoder layers.
    num_encoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    decoder_ffn_dim (`int`, *optional*, defaults to 4096):
        Dimensionality of the `intermediate` (often named feed-forward) layer in decoder.
    num_decoder_layers (`int`, *optional*, defaults to 12):
        Number of decoder layers.
    num_decoder_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer decoder.
    attention_dropout (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    init_std (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    add_cross_attention (`bool`, *optional*, defaults to `True`):
        Whether cross-attention layers should be added to the model.
    is_encoder_decoder (`bool`, *optional*, defaults to `True`):
        Whether this is an encoder/decoder model.
    pad_token_id (`int`, *optional*, defaults to 1)
        Padding token id.
    bos_token_id (`int`, *optional*, defaults to 0)
        Beginning of stream token id.
    eos_token_id (`int`, *optional*, defaults to 2)
        End of stream token id.
    ngram (`int`, *optional*, defaults to 2)
        Number of future tokens to predict. Set to 1 to be same as traditional Language model to predict next first
        token.
    num_buckets (`int`, *optional*, defaults to 32)
        The number of buckets to use for each attention layer. This is for relative position calculation. See the
        `T5 paper <see https://arxiv.org/abs/1910.10683>`__ for more details.
    relative_max_distance (`int`, *optional*, defaults to 128)
        Relative distances greater than this number will be put into the last same bucket. This is for relative
        position calculation. See the `T5 paper <see https://arxiv.org/abs/1910.10683>`__ for more details.
    disable_ngram_loss (`bool`, *optional*, defaults to `False`):
        Whether be trained predicting only the next first token.
    eps (`float`, *optional*, defaults to 0.0):
        Controls the `epsilon` parameter value for label smoothing in the loss calculation. If set to 0, no label
        smoothing is performed.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models).

.. class:: transformers.models.xlm_roberta.configuration_xlm_roberta.XLMRobertaConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The XLM-RoBERTa model was proposed in `Unsupervised Cross-lingual Representation Learning at Scale <https://arxiv.org/abs/1911.02116>`__ by Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume
Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov. It is based on Facebook's
RoBERTa model released in 2019. It is a large multi-lingual language model, trained on 2.5TB of filtered CommonCrawl
data.

The abstract from the paper is the following:

*This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a
wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred
languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly
outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +13.8% average accuracy on
XNLI, +12.3% average F1 score on MLQA, and +2.1% average F1 score on NER. XLM-R performs particularly well on
low-resource languages, improving 11.8% in XNLI accuracy for Swahili and 9.2% for Urdu over the previous XLM model. We
also present a detailed empirical evaluation of the key factors that are required to achieve these gains, including the
trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource
languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing
per-language performance; XLM-Ris very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We
will make XLM-R code, data, and models publicly available.*

Tips:

- XLM-RoBERTa is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does
  not require `lang` tensors to understand which language is used, and should be able to determine the correct
  language from the input ids.
- Uses RoBERTa tricks on the XLM approach, but does not use the translation language modeling objective. It only uses masked language modeling on sentences coming from one language.
- This implementation is the same as RoBERTa. Refer to the `documentation of RoBERTa <roberta>`__ for usage examples
  as well as the information relative to the inputs and outputs.

This model was contributed by `stefan-it <https://huggingface.co/stefan-it>`__. The original code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/xlmr>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 30522):
        Vocabulary size of the XLM-RoBERTa model. Defines the number of different tokens that can be represented by
        the `inputs_ids` passed when calling ``XLMRobertaModel`` or ``TFXLMRobertaModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``XLMRobertaModel`` or
        ``TFXLMRobertaModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    is_decoder (`bool`, *optional*, defaults to `False`):
        Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.xlm_roberta_xl.configuration_xlm_roberta_xl.XLMRobertaXLConfig(vocab_size=250880, hidden_size=2560, num_hidden_layers=36, num_attention_heads=32, intermediate_size=10240, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=514, type_vocab_size=1, initializer_range=0.02, layer_norm_eps=1e-05, pad_token_id=1, bos_token_id=0, eos_token_id=2, position_embedding_type='absolute', use_cache=True, classifier_dropout=None, **kwargs)

The XLM-RoBERTa-XL model was proposed in `Larger-Scale Transformers for Multilingual Masked Language Modeling <https://arxiv.org/abs/2105.00572>`__ by Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, Alexis Conneau. 

The abstract from the paper is the following:

*Recent work has demonstrated the effectiveness of cross-lingual language model pretraining for cross-lingual understanding. In this study, we present the results of two larger multilingual masked language models, with 3.5B and 10.7B parameters. Our two new models dubbed XLM-R XL and XLM-R XXL outperform XLM-R by 1.8% and 2.4% average accuracy on XNLI. Our model also outperforms the RoBERTa-Large model on several English tasks of the GLUE benchmark by 0.3% on average while handling 99 more languages. This suggests pretrained models with larger capacity may obtain both strong performance on high-resource languages while greatly improving low-resource languages. We make our code and models publicly available.*

Tips:

- XLM-RoBERTa-XL is a multilingual model trained on 100 different languages. Unlike some XLM multilingual models, it does
  not require `lang` tensors to understand which language is used, and should be able to determine the correct
  language from the input ids.

This model was contributed by `Soonhwan-Kwon <https://github.com/Soonhwan-Kwon>`__ and `stefan-it <https://huggingface.co/stefan-it>`__. The original code can be found `here <https://github.com/pytorch/fairseq/tree/master/examples/xlmr>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 250880):
        Vocabulary size of the XLM_ROBERTA_XL model. Defines the number of different tokens that can be represented
        by the `inputs_ids` passed when calling ``XLMRobertaXLModel``.
    hidden_size (`int`, *optional*, defaults to 2560):
        Dimensionality of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 36):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 32):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 10240):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"silu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 514):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 1):
        The vocabulary size of the `token_type_ids` passed when calling ``XLMRobertaXLModel`` or
        ``TFXLMRobertaXLModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-5):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
        positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
        `Self-Attention with Relative Position Representations (Shaw et al.) <https://arxiv.org/abs/1803.02155>`__.
        For more information on `"relative_key_query"`, please refer to *Method 4* in `Improve Transformer Models
        with Better Relative Position Embeddings (Huang et al.) <https://arxiv.org/abs/2009.13658>`__.
    use_cache (`bool`, *optional*, defaults to `True`):
        Whether or not the model should return the last key/values attentions (not used by all models). Only
        relevant if `config.is_decoder=True`.
    classifier_dropout (`float`, *optional*):
        The dropout ratio for the classification head.

.. class:: transformers.models.xlnet.configuration_xlnet.XLNetConfig(vocab_size=32000, d_model=1024, n_layer=24, n_head=16, d_inner=4096, ff_activation='gelu', untie_r=True, attn_type='bi', initializer_range=0.02, layer_norm_eps=1e-12, dropout=0.1, mem_len=512, reuse_len=None, use_mems_eval=True, use_mems_train=False, bi_data=False, clamp_len=-1, same_length=False, summary_type='last', summary_use_proj=True, summary_activation='tanh', summary_last_dropout=0.1, start_n_top=5, end_n_top=5, pad_token_id=5, bos_token_id=1, eos_token_id=2, **kwargs)

The XLNet model was proposed in `XLNet: Generalized Autoregressive Pretraining for Language Understanding <https://arxiv.org/abs/1906.08237>`__ by Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov,
Quoc V. Le. XLnet is an extension of the Transformer-XL model pre-trained using an autoregressive method to learn
bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization
order.

The abstract from the paper is the following:

*With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves
better performance than pretraining approaches based on autoregressive language modeling. However, relying on
corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a
pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive
pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all
permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive
formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into
pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large
margin, including question answering, natural language inference, sentiment analysis, and document ranking.*

Tips:

- The specific attention pattern can be controlled at training and test time using the `perm_mask` input.
- Due to the difficulty of training a fully auto-regressive model over various factorization order, XLNet is pretrained
  using only a sub-set of the output tokens as target which are selected with the `target_mapping` input.
- To use XLNet for sequential decoding (i.e. not in fully bi-directional setting), use the `perm_mask` and
  `target_mapping` inputs to control the attention span and outputs (see examples in
  *examples/pytorch/text-generation/run_generation.py*)
- XLNet is one of the few models that has no sequence length limit.
- XLNet is not a traditional autoregressive model but uses a training strategy that builds on that. It permutes the tokens in the sentence, then allows the model to use the last n tokens to predict the token n+1. Since this is all done with a mask, the sentence is actually fed in the model in the right order, but instead of masking the first n tokens for n+1, XLNet uses a mask that hides the previous tokens in some given permutation of 1,…,sequence length.
- XLNet also uses the same recurrence mechanism as Transformer-XL to build long-term dependencies.

This model was contributed by `thomwolf <https://huggingface.co/thomwolf>`__. The original code can be found `here <https://github.com/zihangdai/xlnet/>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 32000):
        Vocabulary size of the XLNet model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``XLNetModel`` or ``TFXLNetModel``.
    d_model (`int`, *optional*, defaults to 1024):
        Dimensionality of the encoder layers and the pooler layer.
    n_layer (`int`, *optional*, defaults to 24):
        Number of hidden layers in the Transformer encoder.
    n_head (`int`, *optional*, defaults to 16):
        Number of attention heads for each attention layer in the Transformer encoder.
    d_inner (`int`, *optional*, defaults to 4096):
        Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
    ff_activation (`str` or `Callable`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the If string, `"gelu"`, `"relu"`, `"silu"` and
        `"gelu_new"` are supported.
    untie_r (`bool`, *optional*, defaults to `True`):
        Whether or not to untie relative position biases
    attn_type (`str`, *optional*, defaults to `"bi"`):
        The attention type used by the model. Set `"bi"` for XLNet, `"uni"` for Transformer-XL.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    dropout (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    mem_len (`int` or `None`, *optional*):
        The number of tokens to cache. The key/value pairs that have already been pre-computed in a previous
        forward pass won't be re-computed. See the
        `quickstart <https://huggingface.co/transformers/quickstart.html#using-the-past>`__ for more information.
    reuse_len (`int`, *optional*):
        The number of tokens in the current batch to be cached and reused in the future.
    bi_data (`bool`, *optional*, defaults to `False`):
        Whether or not to use bidirectional input pipeline. Usually set to `True` during pretraining and `False`
        during finetuning.
    clamp_len (`int`, *optional*, defaults to -1):
        Clamp all relative distances larger than clamp_len. Setting this attribute to -1 means no clamping.
    same_length (`bool`, *optional*, defaults to `False`):
        Whether or not to use the same attention length for each token.
    summary_type (`str`, *optional*, defaults to "last"):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Has to be one of the following options:

            - `"last"`: Take the last token hidden state (like XLNet).
            - `"first"`: Take the first token hidden state (like BERT).
            - `"mean"`: Take the mean of all tokens hidden states.
            - `"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
            - `"attn"`: Not implemented now, use multi-head attention.
    summary_use_proj (`bool`, *optional*, defaults to `True`):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Whether or not to add a projection after the vector extraction.
    summary_activation (`str`, *optional*):
        Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

        Pass `"tanh"` for a tanh activation to the output, any other value will result in no activation.
    summary_proj_to_labels (`boo`, *optional*, defaults to `True`):
        Used in the sequence classification and multiple choice models.

        Whether the projection outputs should have `config.num_labels` or `config.hidden_size` classes.
    summary_last_dropout (`float`, *optional*, defaults to 0.1):
        Used in the sequence classification and multiple choice models.

        The dropout ratio to be used after the projection and activation.
    start_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    end_n_top (`int`, *optional*, defaults to 5):
        Used in the SQuAD evaluation script.
    use_mems_eval (`bool`, *optional*, defaults to `True`):
        Whether or not the model should make use of the recurrent memory mechanism in evaluation mode.
    use_mems_train (`bool`, *optional*, defaults to `False`):
        Whether or not the model should make use of the recurrent memory mechanism in train mode.

        <Tip>

        For pretraining, it is recommended to set `use_mems_train` to `True`. For fine-tuning, it is recommended to
        set `use_mems_train` to `False` as discussed
        `here <https://github.com/zihangdai/xlnet/issues/41#issuecomment-505102587>`__. If `use_mems_train` is set to
        `True`, one has to make sure that the train batches are correctly pre-processed, *e.g.* `batch_1 = [[This
        line is], [This is the]]` and `batch_2 = [[ the first line], [ second line]]` and that all batches are of
        equal size.

        </Tip>

.. class:: transformers.models.yoso.configuration_yoso.YosoConfig(vocab_size=50265, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=4096, type_vocab_size=1, initializer_range=0.02, layer_norm_eps=1e-12, position_embedding_type='absolute', use_expectation=True, hash_code_len=9, num_hash=64, conv_window=None, use_fast_hash=True, lsh_backward=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs)

The YOSO model was proposed in `You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling <https://arxiv.org/abs/2111.09714>`__  
by Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. YOSO approximates standard softmax self-attention
via a Bernoulli sampling scheme based on Locality Sensitive Hashing (LSH). In principle, all the Bernoulli random variables can be sampled with
a single hash. 

The abstract from the paper is the following:

*Transformer-based models are widely used in natural language processing (NLP). Central to the transformer model is 
the self-attention mechanism, which captures the interactions of token pairs in the input sequences and depends quadratically 
on the sequence length. Training such models on longer sequences is expensive. In this paper, we show that a Bernoulli sampling 
attention mechanism based on Locality Sensitive Hashing (LSH), decreases the quadratic complexity of such models to linear. 
We bypass the quadratic cost by considering self-attention as a sum of individual tokens associated with Bernoulli random 
variables that can, in principle, be sampled at once by a single hash (although in practice, this number may be a small constant). 
This leads to an efficient sampling scheme to estimate self-attention which relies on specific modifications of 
LSH (to enable deployment on GPU architectures). We evaluate our algorithm on the GLUE benchmark with standard 512 sequence 
length where we see favorable performance relative to a standard pretrained Transformer. On the Long Range Arena (LRA) benchmark, 
for evaluating performance on long sequences, our method achieves results consistent with softmax self-attention but with sizable 
speed-ups and memory savings and often outperforms other efficient self-attention methods. Our code is available at this https URL*

Tips:

- The YOSO attention algorithm is implemented through custom CUDA kernels, functions written in CUDA C++ that can be executed multiple times
in parallel on a GPU.
- The kernels provide a `fast_hash` function, which approximates the random projections of the queries and keys using the Fast Hadamard Transform. Using these
hash codes, the `lsh_cumulation` function approximates self-attention via LSH-based Bernoulli sampling.
- To use the custom kernels, the user should set `config.use_expectation = False`. To ensure that the kernels are compiled successfully, 
the user must install the correct version of PyTorch and cudatoolkit. By default, `config.use_expectation = True`, which uses YOSO-E and 
does not require compiling CUDA kernels.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/yoso_architecture.jpg"
alt="drawing" width="600"/> 

<small> YOSO Attention Algorithm. Taken from the <a href="https://arxiv.org/abs/2111.09714">original paper</a>.</small>

This model was contributed by `novice03 <https://huggingface.co/novice03>`__. The original code can be found `here <https://github.com/mlpen/YOSO>`__.

Args:
    vocab_size (`int`, *optional*, defaults to 50265):
        Vocabulary size of the YOSO model. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed when calling ``YosoModel``.
    hidden_size (`int`, *optional*, defaults to 768):
        Dimension of the encoder layers and the pooler layer.
    num_hidden_layers (`int`, *optional*, defaults to 12):
        Number of hidden layers in the Transformer encoder.
    num_attention_heads (`int`, *optional*, defaults to 12):
        Number of attention heads for each attention layer in the Transformer encoder.
    intermediate_size (`int`, *optional*, defaults to 3072):
        Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` are supported.
    hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
        The dropout ratio for the attention probabilities.
    max_position_embeddings (`int`, *optional*, defaults to 512):
        The maximum sequence length that this model might ever be used with. Typically set this to something large
        just in case (e.g., 512 or 1024 or 2048).
    type_vocab_size (`int`, *optional*, defaults to 2):
        The vocabulary size of the `token_type_ids` passed when calling ``YosoModel``.
    initializer_range (`float`, *optional*, defaults to 0.02):
        The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
    layer_norm_eps (`float`, *optional*, defaults to 1e-12):
        The epsilon used by the layer normalization layers.
    position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
        Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`.
    use_expectation (`bool`, *optional*, defaults to `True`):
        Whether or not to use YOSO Expectation. Overrides any effect of num_hash.
    hash_code_len (`int`, *optional*, defaults to 9):
        The length of hashes generated by the hash functions.
    num_hash (`int`, *optional*, defaults to 64):
        Number of hash functions used in ``YosoSelfAttention``.
    conv_window (`int`, *optional*):
        Kernel size of depth-wise convolution.
    use_fast_hash (`bool`, *optional*, defaults to `False`):
        Whether or not to use custom cuda kernels which perform fast random projection via hadamard transform.
    lsh_backward (`bool`, *optional*, defaults to `True`):
        Whether or not to perform backpropagation using Locality Sensitive Hashing.

