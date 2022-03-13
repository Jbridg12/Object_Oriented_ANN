import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters_ex3 import generateExample3

# Create a feed forward network
model=Sequential()

# Add convolutional layers, flatten, and fully connected layer
model.add(layers.Conv2D(2,3,input_shape=(8,8,1),activation='sigmoid'))
#model.add(layers.MaxPool2D((2,2), strides=(1,1), padding='valid'))
model.add(layers.MaxPool2D((2,2), padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))

# Call weight/data generating function
l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, output = generateExample3()

#Set weights to desired values
print(f'W1: {l1k1}')

#setting weights and bias of first layer.
l1k1=l1k1.reshape(3,3,1,1)
l1k2=l1k2.reshape(3,3,1,1)

# w1 = l1k1
print(f'B1: {np.array([l1b1[0]])}')
# print(l1k1)
w1=np.concatenate((l1k1,l1k2),axis=3)
model.layers[0].set_weights([w1,np.array([l1b1[0],l1b2[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)
# model.layers[1].set_weights()

#setting weights and bias of second layer.
# l2c1=l2c1.reshape(3,3,1,1)
# l2c2=l2c2.reshape(3,3,1,1)
#
# w1=np.concatenate((l2c1,l2c2),axis=2)
# model.layers[1].set_weights([w1,l2b])

#setting weights and bias of fully connected layer.
print(f'Dense: {l3}')
print(f'Dense bias: {l3b}')
model.layers[3].set_weights([np.transpose(l3),l3b])

# new_input = []
# for i in input[:5]:
#     new_input.append(i[:5])
#
# input = np.array(new_input)

#Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
img=np.expand_dims(input,axis=(0,3))
print('input:')
print(input)
print()

print('output:')
print(output)
print()

# Inserted by Josh
model_output = model.layers[0].output
m = keras.Model(inputs=model.input, outputs=model_output)
print(m.predict(img))

#print needed values.
np.set_printoptions(precision=5)
print('model output before:')
print(model.predict(img))
sgd = optimizers.SGD(learning_rate=100)
model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
history=model.fit(img,output,batch_size=1,epochs=1)
print('\nmodel output after:')
print(model.predict(img))

print('\n1st convolutional layer, 1st kernel weights:')
print(np.squeeze(model.get_weights()[0][:,:,0,0]))
print('\n1st convolutional layer, 1st kernel bias:')
print(np.squeeze(model.get_weights()[1][0]))

print('\n1st convolutional layer, 2nd kernel weights:')
print(np.squeeze(model.get_weights()[0][:,:,0,1]))
print('\n1st convolutional layer, 2nd kernel bias:')
print(np.squeeze(model.get_weights()[1][1]))

print('\nMax Pool layer weights:')
print('None')
# print(np.squeeze(model.get_weights()[1][:,:,0,0]))
# print(np.squeeze(model.get_weights()[1][:,:,1,0]))
# print(np.squeeze(model.get_weights()[1]))

print('\nfully connected layer weights:')
print(np.squeeze(model.get_weights()[2]))
print('\nfully connected layer bias:')
print(np.squeeze(model.get_weights()[3]))