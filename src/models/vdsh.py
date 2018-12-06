"""Implements the unsupervised Variational Deep Semantic Hashing Model (VDSH)
[1]_ for document hashing.
    
.. [1] "Variational Deep Semantic Hashing for Text Documents", 
   https://arxiv.org/pdf/1708.03436.pdf
"""
import tensorflow as tf
import json
import keras.layers as layers
import logging
import numpy as np
import os
from keras.models import Model
from keras import backend as K
from gensim.corpora import Dictionary
from src import HOME_DIR
from src.utils.corpus import load_corpus
from src.utils.tokenization import WordTokenizer

logger = logging.getLogger(__name__)

def _sampling(args, epsilon_std=1.):
    """VAE sampling"""
    mu, sigma = args
    epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=epsilon_std)
    return mu + sigma * epsilon

def _join_model_path(model_file):
    return os.path.join(HOME_DIR, 'models', model_file)

def reconstruction_loss(bow, p):
    """Computes reconstruction loss between true bow representations and
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
    return reconstr_loss

class VDSH:

    def __init__(self):
        self.encoder = None # The encoder portion of the model.
        self.encoder_decoder = None # The end-to-end Keras model.

    def build_model(self, input_dim, intermediate_dim=500, latent_dim=32):
        """Builds the model architecture and assigns the `encoder` and
        `encoder_decoder` instance variables."""
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
        encoder_decoder = Model(inputs, P, name='vdsh_encoder_decoder')
        encoder = Model(inputs, mu, name='vdsh_encoder')

        # Compute KL loss
        kl_loss = 0.5 * K.mean(K.sum(
            K.square(sigma) + K.square(mu) - 2 * K.log(sigma) - 1, axis=-1))

        def _loss(bow, p):
            return reconstruction_loss(bow, p) + kl_loss

        encoder_decoder.compile(optimizer='adam', loss=_loss,
                                metrics=[reconstruction_loss])
        self.encoder_decoder = encoder_decoder
        self.encoder = encoder

    def load_weights(self, model_file):
        """Load weights from a pretrained model."""
        if not self.encoder_decoder:
            raise TypeError('You need to build a model using the method '
                            '`build_model` before trying to load an already '
                            'trained one.')
        self.encoder_decoder.load_weights(_join_model_path(model_file))

    def train(self, X, epochs=1, batch_size=128, model_file=None,
              history_file=None):
        """Fits the model parameters

        Params
        ------
        X : scipy.sparse.csr.csr_matrix
            TFIDF matrix
        epochs : int
        batch_size : int
        model_file : str
            Name of file to save model weights to.
        history_file : str
            Name of file to save training history to.
        """
        if not self.encoder_decoder:
            self.build_model(input_dim=X.shape[1])

        if os.path.exists(_join_model_path(model_file)):
            logger.info('Loading existing weights from %s.' % model_file)
            self.load_weights(model_file)

        self.encoder_decoder.fit(X, X, batch_size=batch_size, epochs=epochs)
        if model_file:
            self.encoder_decoder.save_weights(_join_model_path(model_file))
        if history_file:
            json.dump(
                self.encoder_decoder.history.history,
                open(_join_model_path(history_file), 'w'))

    def encoder_predict(self, X):
        return self.encoder.predict(X)

if __name__ == '__main__':
    from src.utils.corpus import load_corpus, generate_tfidf
    corpus = load_corpus().head(20000)
    dictionary = Dictionary(corpus.bag_of_words)
    dictionary.filter_extremes(no_below=100)
    X = generate_tfidf(corpus, dictionary)
    vdsh = VDSH()
    vdsh.build_model(X.shape[1])
    vdsh.train(X, epochs=1, model_file='vdsh.hdf5', history_file='vdsh.history')
