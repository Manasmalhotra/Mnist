import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

dataset,mnist_info=tfds.load(name='mnist',with_info=True,as_supervised=True)
mnist_train,mnist_test=dataset['train'],dataset['test']
validation=0.1*mnist_info.splits['train'].num_examples
validation=tf.cast(validation,tf.int64)
num_test=mnist_info.splits['test'].num_examples
num_test=tf.cast(num_test,tf.int64)

def scale(image,label):
    image=tf.cast(image,tf.float32)
    image/=255.
    return image,label

scaled_train_and_validate=mnist_train.map(scale)
test_data=mnist_test.map(scale)

buffer=10000
shuffled_train_and_validation=scaled_train_and_validate.shuffle(buffer)
validation_data=shuffled_train_and_validation.take(validation)
train_data=shuffled_train_and_validation.skip(validation)

batches=100
train_data=train_data.batch(batches)
validation_data=validation_data.batch(validation)
test_data=test_data.batch(num_test)

validation_inputs, validation_targets=next(iter(validation_data))
input_size=784
out_size=10
hidden_layer=50
model=tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28,28,1))
                           ,tf.keras.layers.Dense(hidden_layer,activation='relu'),
                           tf.keras.layers.Dense(hidden_layer,activation='relu'),
                           tf.keras.layers.Dense(out_size,activation='softmax')
                           ])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
epockk=5
model.fit(train_data,epochs=epockk,validation_data=(validation_inputs, validation_targets),verbose=2,validation_steps=validation)
test_loss,test_accuracy=model.evaluate(test_data)
print(test_loss,test_accuracy)
