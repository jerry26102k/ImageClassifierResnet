from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D
from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.models import Model
import tensorflow as tf



def identity_block(X, f, filters, training=True, initializer=random_uniform):
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(filters=F1, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2, training=True, initializer=glorot_uniform):
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)

    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid',
                        kernel_initializer=initializer(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def res_net_50(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)

    X = Conv2D(filters=64, kernel_size=3, strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(X)
    BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=(64,64,256), s=1)
    X = identity_block(X, 3, filters=[64, 64, 256])
    X = identity_block(X, 3, filters=[64, 64, 256])

    X = convolutional_block(X, f=3, filters=(128, 128, 512), s=2)
    X = identity_block(X, 3, filters=[128, 128, 512])
    X = identity_block(X, 3, filters=[128, 128, 512])
    X = identity_block(X, 3, filters=[128, 128, 512])

    X = convolutional_block(X, f=3, filters=(256, 256, 1024), s=2)
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])
    X = identity_block(X, 3, filters=[256, 256, 1024])

    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048])
    X = identity_block(X, f=3, filters=[512, 512, 2048])

    X = AveragePooling2D(pool_size=(2, 2))(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)
    model = Model(inputs=X_input, outputs=X)

    return model


my_model = res_net_50(input_shape=(32, 32, 3), classes=10)
print(my_model.summary())
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = tf.keras.datasets.cifar10.load_data()
X_train = x_train_orig / 255.
X_test = x_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = tf.keras.utils.to_categorical(y_train_orig, 10)
Y_test = tf.keras.utils.to_categorical(y_test_orig, 10)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

my_model.fit(X_train, Y_train, epochs=20, batch_size=32)

preds = my_model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


