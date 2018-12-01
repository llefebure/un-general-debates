"""Implements the unsupervised Variational Deep Semantic Hashing Model (VDSH)
[1]_ for document hashing.
    
.. [1] "Variational Deep Semantic Hashing for Text Documents", 
   https://arxiv.org/pdf/1708.03436.pdf
"""
import tensorflow as tf
import keras.layers as layers
import numpy as np
import os
from keras.models import Model
from keras import backend as K
from gensim.matutils import corpus2dense
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from src import HOME_DIR
from src.utils.corpus import load_corpus
from src.utils.tokenization import WordTokenizer

def _sampling(args, epsilon_std=1.):
    """VAE sampling"""
    mu, sigma = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=epsilon_std)
    return mu + sigma * epsilon

def _join_model_path(model_file):
    return os.path.join(HOME_DIR, 'models', model_file)

class VDSH:

    full = None # The end-to-end Keras model.
    encoder = None # The encoder portion of the model.

    def build_model(self, input_dim, intermediate_dim=30, latent_dim=10):
        inputs = layers.Input(
            shape=(input_dim,), dtype='float32', name='bow_input')
        t1 = layers.Dense(
            intermediate_dim, activation='relu', name='t1')(inputs)
        t2 = layers.Dense(intermediate_dim, activation='relu', name='t2')(t1)
        mu = layers.Dense(latent_dim, name='mu')(t2)
        sigma = layers.Dense(
            latent_dim, activation='exponential', name='sigma')(t2)
        s = layers.Lambda(_sampling, name='s')([mu, sigma])
        c = layers.Dense(input_dim, activation='exponential', name='c')(s)
        P = layers.Activation('softmax', name='P')(c)
        full = Model(inputs, P, name='vdsh')
        encoder = Model(inputs, mu, name='vdsh')
        
        def _loss(bow, p):
            """Computes custom loss between true bow representations and
            softmax predictions."""

            # Flatten the input tensors.
            bow_flat = K.reshape(bow, shape=(-1,))
            p_flat = K.reshape(p, shape=(-1,))

            # Gather nonzero indices.
            indices = K.squeeze(tf.where(K.greater(bow_flat, 0.)), axis=1)
            bow_flat = K.gather(bow_flat, indices)
            p_flat = K.gather(p_flat, indices)

            reconstr_loss = -K.sum(K.log(K.maximum(bow_flat * p_flat, 1e-10)))
            reconstr_loss /= K.cast(K.shape(bow)[0], dtype='float32')
            kl_loss = K.sum(
                K.square(sigma) + K.square(mu) - 2 * K.log(sigma) - 1, axis=-1)
            kl_loss = 0.5 * K.mean(kl_loss)
            return reconstr_loss + kl_loss

        full.compile(optimizer='adam', loss=_loss)
        self.full = full
        self.encoder = encoder

    def load_weights(self, model_file):
        """Load weights from a pretrained model."""
        if not self.full:
            raise TypeError('You need to build a model using the method '
                            '`build_model` before trying to load an already '
                            'trained one.')
        self.full.load_weights(_join_model_path(model_file))

    def train(self, X, epochs=1, batch_size=128, model_file=None):
        if not self.full:
            self.build_model(input_dim=X.shape[1])
        self.full.fit(X, X, epochs=epochs, batch_size=batch_size)
        if model_file:
            self.full.save_weights(_join_model_path(model_file))

    def encoder_predict(self, X):
        return self.encoder.predict(X)

if __name__ == '__main__':
    from src.utils.corpus import load_corpus, generate_tfidf
    corpus = load_corpus()
    X, dictionary = generate_tfidf(corpus)
    vdsh = VDSH()
    vdsh.build_model(X.shape[1])
    vdsh.train(X, epochs=5, model_file='vdsh.hdf5')
