from keras import models 
from Preproces import *
img_height, img_width = 32,32
conv_base = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='max', input_shape = (img_width, img_height, 3))
for layer in conv_base.layers:
    print(layer, layer.trainable)
    
    
    
model = models.Sequential()
model.add(conv_base)
model.add(keras.layers.Dense(43, activation='softmax'))
model.summary()

learning_rate = 5e-5
epochs = 10
batch_size = 32
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=learning_rate, clipnorm = 1.), metrics = ['acc'])
history = model.fit(aug.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=X_train.shape[0] // batch_size,epochs=epochs, validation_data=(X_val, y_val),verbose=1)
model.save('sign_classification_vgg16.h5')
## Evaluating the model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig("model_history.png")
plt.show()

