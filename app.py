from transformers import AutoTokenizer, TFAutoModel, AutoConfig
from keras import utils
import numpy as np
from keras.layers import Dense, Embedding
from tqdm.notebook import tqdm
import tensorflow as tf
from glob import glob

tokenizer = AutoTokenizer.from_pretrained('./model')
config = AutoConfig.from_pretrained('./model', output_hidden_states=True)
config.hidden_dropout_prob = 0
config.attention_probs_dropout_prob = 0
backbone = TFAutoModel.from_pretrained('./model', config=config)

class MeanPool(tf.keras.layers.Layer):
    def call(self, inputs, mask=None):
        broadcast_mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        embedding_sum = tf.reduce_sum(inputs * broadcast_mask, axis=1)
        mask_sum = tf.reduce_sum(broadcast_mask, axis=1)
        mask_sum = tf.math.maximum(mask_sum, tf.constant([1e-9]))
        return embedding_sum / mask_sum

class WeightsSumOne(tf.keras.constraints.Constraint):
    def __call__(self, w):
        return tf.nn.softmax(w, axis=0)              

def build_model():
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_masks = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_masks")
    
    normal_initializer = tf.keras.initializers.GlorotUniform()
    zeros_initializer = tf.keras.initializers.Zeros()
    ones_initializer = tf.keras.initializers.Ones()
    
    for encoder_block in backbone.deberta.encoder.layer[-1:]:
        for layer in encoder_block.submodules:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel.assign(normal_initializer(shape=layer.kernel.shape,
                                                       dtype=layer.kernel.dtype))
                if layer.bias is not None:
                    layer.bias.assign(zeros_initializer(shape=layer.bias.shape,
                                                        dtype=layer.bias.dtype))
            
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                layer.beta.assign(zeros_initializer(shape=layer.beta.shape,
                                                    dtype=layer.beta.dtype))
                layer.gamma.assign(ones_initializer(shape=layer.gamma.shape,
                                                    dtype=layer.gamma.dtype))

    x = backbone.deberta(input_ids, attention_mask=attention_masks)
    x = x.hidden_states 
    x = tf.stack([MeanPool()(hidden_s, mask=attention_masks) for hidden_s in x[-4:]], axis=2) 
    x = tf.keras.layers.Dense(1, use_bias=False, kernel_constraint=WeightsSumOne())(x)
    x = tf.squeeze(x, axis=-1)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=x)

    return model      

def predict_sentiment(text):
    tokenized = tokenizer(' '.join(text.split('  ')[0:int(len(text.split('  '))/2)]), 
                          ' '.join(text.split('  ')[int(len(text.split('  '))/2)::]),
                          truncation=True,
                          max_length=128,
                          padding='max_length',
                          return_attention_mask=True,
                          return_tensors="np"
                          )
    ids = []
    masks = []
    tokens = []
    ids.append(tokenized['input_ids'][0])
    masks.append(tokenized['attention_mask'][0])
    tokens.append(tokenized['token_type_ids'][0])
    ids = np.array(ids, dtype="int32")
    masks = np.array(masks, dtype="int32")
    tokens = np.array(tokens, dtype="int32")
    model = build_model()
    pr = []
    for w in glob('./*.h5'):
        model.load_weights(w)
        p = model.predict((ids,masks), batch_size=8)
        pr.append(p)
    pr = np.mean(pr)
    return pr

def process(in_text):
    result = predict_sentiment(in_text)
    if result > 0.5:
       print('Positivity: ',round(result,1))
    else:
        print('Negativity: 'round(1-result,1))
