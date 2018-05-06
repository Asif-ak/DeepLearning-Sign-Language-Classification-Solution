from keras import layers,models
import numpy as np
from keras.utils import to_categorical

Image_path='dataset\sign-language-digits-dataset\Sign-language-digits-dataset\X.npy'
Images=np.load(Image_path)
label_path='dataset\sign-language-digits-dataset\Sign-language-digits-dataset\Y.npy'
labels=np.load(label_path)

network=models.Sequential(name='Sign Language Model')
network.add(layers.Dense(512,activation='relu',input_shape=(64*64,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

train_images=Images[0:2000]
test_images=Images[2001:]

train_labels=labels[0:2000]
test_labels=labels[2001:]

train_images=train_images.reshape((2000,64*64))
train_images=train_images.astype('float32') / 255

test_images=test_images.reshape((61,64*64))
test_images=test_images.astype('float32') / 255

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

network.fit(train_images,train_labels,batch_size=50,epochs=5)

loss,accuracy=network.evaluate(test_images,test_labels)
print(loss)
print(accuracy)




