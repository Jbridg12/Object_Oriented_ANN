import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from parameters import generateExample2

# Create a feed forward network
model=Sequential()

# Add convolutional layers, flatten, and fully connected layer
model.add(layers.Conv2D(1,3,input_shape=(5,5,1),activation='sigmoid'))
model.add(layers.Flatten())
model.add(layers.Dense(1,activation='sigmoid'))

# Call weight/data generating function
l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, output = generateExample2()

#Set weights to desired values

#setting weights and bias of first layer.
l1k1=l1k1.reshape(3,3,1,1)
# l1k2=l1k2.reshape(3,3,1,1)

w1 = l1k1
print(f'W1: {w1}')
print(f'B1: {np.array([l1b1[0]])}')
# print(l1k1)
# w1=np.concatenate((l1k1),axis=1)
model.layers[0].set_weights([w1,np.array([l1b1[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)


#setting weights and bias of second layer.
# l2c1=l2c1.reshape(3,3,1,1)
# l2c2=l2c2.reshape(3,3,1,1)
#
# w1=np.concatenate((l2c1,l2c2),axis=2)
# model.layers[1].set_weights([w1,l2b])

#setting weights and bias of fully connected layer.
print(f'Dense: {l3}, Transpose: {np.transpose(l3)}')
print(f'Dense bias: {l3b}')
model.layers[2].set_weights([np.transpose(l3),l3b])

new_input = []
for i in input[:5]:
    new_input.append(i[:5])

input = np.array(new_input)

#Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
img=np.expand_dims(input,axis=(0,3))
print('input:')
print(input)
print()

print('output:')
print(output)
print()

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

# print('\n1st convolutional layer, 2nd kernel weights:')
# print(np.squeeze(model.get_weights()[0][:,:,0,1]))
# print('\n1st convolutional layer, 2nd kernel bias:')
# print(np.squeeze(model.get_weights()[1][1]))


# print('\n2nd convolutional layer weights:')
# print(np.squeeze(model.get_weights()[2][:,:,0,0]))
# print(np.squeeze(model.get_weights()[2][:,:,1,0]))
# print('\n2nd convolutional layer bias:')
# print(np.squeeze(model.get_weights()[3]))

print('\nfully connected layer weights:')
print(np.squeeze(model.get_weights()[2]))
print('\nfully connected layer bias:')
print(np.squeeze(model.get_weights()[3]))

"""
input:
[[0.1650159  0.39252924 0.09346037 0.82110566 0.15115202]
 [0.98762547 0.45630455 0.82612284 0.25137413 0.59737165]
 [0.59020136 0.03928177 0.35718176 0.07961309 0.30545992]
 [0.03995921 0.42949218 0.31492687 0.63649114 0.34634715]
 [0.76324059 0.87809664 0.41750914 0.60557756 0.51346663]]

output:
[0.31827281]

model output before:
[[0.99477]]
1/1 [==============================] - 0s 169ms/step - loss: 0.4576 - accuracy: 0.0000e+00

model output after:
[[0.29105]]

1st convolutional layer, 1st kernel weights:
[[ 0.67983 -0.10195  0.55631]
 [ 0.63335  0.38956  0.11743]
 [ 0.07641  0.66339  0.05399]]

1st convolutional layer, 1st kernel bias:
0.65133345

fully connected layer weights:
[-0.02314 -0.11798  0.18762 -0.47613  0.22919 -0.31653  0.09061 -0.33937
  0.22619]

fully connected layer bias:
-0.3787216

OUTPUT: [0.7245792734818262]
LOSS: 0.16508494226710851

"""
