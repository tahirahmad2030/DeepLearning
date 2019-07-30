import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


callbacks = myCallback()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
print(model.evaluate(x_test, y_test))

classifications = model.predict(x_test)

ten_class_prob = (classifications[0])
print(ten_class_prob)
print("Label Predicted as : ", list(ten_class_prob).index(max(ten_class_prob)))
print("Actual Label : ", y_test[0])