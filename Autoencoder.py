from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

class Autoencoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.build_autoencoder()
        
    def build_autoencoder(self):
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        
        # Decoder
        decoded = Dense(self.input_dim, activation='sigmoid')(encoded)
        
        # Autoencoder Model
        self.autoencoder = Model(inputs=input_layer, outputs=decoded)
        
        # Encoder Model
        self.encoder = Model(inputs=input_layer, outputs=encoded)
        
        # Decoder Model
        encoded_input = Input(shape=(self.encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
        
        self.autoencoder.compile(optimizer='adam', loss='mse')
    
    def train(self, X, epochs=50, batch_size=256, validation_split=0.2):
        """Train the autoencoder on the input data."""
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split)
    
    def encode(self, X):
        """Return the compressed (encoded) version of the data."""
        return self.encoder.predict(X)
    
    def decode(self, X_encoded):
        """Return the reconstructed version of the compressed data."""
        return self.decoder.predict(X_encoded)