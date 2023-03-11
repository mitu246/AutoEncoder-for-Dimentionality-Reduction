from encoder_nn import *

callbacks_list=[tf.keras.callbacks.EarlyStopping(
monitor='loss',patience=7),
                tf.keras.callbacks.ReduceLROnPlateau(monitor = "loss", factor = 0.1, patience = 6)
               ]

optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.001)

# Model Fit:
auto_enc,enc = auto_encoder(X.shape[1],[64, 8],'sigmoid')
auto_enc.compile(optimizer=optimizeru,loss='cosine_similarity',metrics=['cosine_similarity'])
auto_enc.fit(X,X,epochs=40,batch_size=64,callbacks=callbacks_list)
encoded_features=enc.predict(X)