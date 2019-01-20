import os
import numpy as np
import cv2
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten,Input,Conv2D
from keras.optimizers import Adam
import random

#工程1，训练出模型
i,j=0,0
img_rows, img_cols = 28, 28
data = np.zeros([8000, 1,img_rows, img_cols])
label = np.zeros([8000,10])
sum = 0
imgs = os.listdir("D:\pytorchpro\pro\mnist_digits_images")
print(len(os.listdir("D:\pytorchpro\pro\mnist_digits_images\\"+imgs[0])))
num = len(imgs)
for i in range(num):
    path="D:\pytorchpro\pro\mnist_digits_images\\"+imgs[i]
    pic=os.listdir(path)
    for j, val in enumerate(pic):
        data[j+sum, :, :,:] = cv2.resize(cv2.cvtColor(cv2.imread(path +"\\" +val), cv2.COLOR_BGR2GRAY), (img_rows, img_cols))/255
        label[j+sum,i:i+1] = i
        if (np.mod(j+sum, 100) == 0):
            print('第', j+sum, '个训练图片正在装载')
    sum += j+1 #每个i循环还要再加1的原因是，list的循环是从j=0开始的，所以要补加1个。

# print(data.shape)
#打散
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
#分配数量
train_data = data[:7000]
train_labels = label[:7000]
validation_data = data[7000:]
validation_labels = label[7000:]
# print(train_data.shape)
# print(train_labels.shape)
# print(validation_data.shape)
# print(validation_labels.shape)

#创建模型
model = Sequential()
# Conv layer 2 output shape (32, 28, 28)
model.add(Conv2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
    activation='relu',
    name="Dense_1_my"
))
# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
    name="pool_1_my",
))
# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first',activation='relu',name="Dense_2_my"))
# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10,name="dense10"))
model.add(Activation('softmax',name="softmax"))
# Another way to define your optimizer
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Training ------------')
# Another way to train the model
model.fit(train_data, train_labels, epochs=5, batch_size=64,)
print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(validation_data, validation_labels)
print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_my.png',show_shapes=True)
model.save('my_model.h5')
del model

#工程2，查看部分特征图
image = cv2.cvtColor(cv2.imread('2.bmp', 1), cv2.COLOR_BGR2GRAY)
print(image.shape)
myimage = np.zeros([1, 1,28, 28])
myimage[0,0,:,:] = cv2.resize(image, (28, 28))/255
#可以修改想要导入的模块
model = load_model('my_model.h5')
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1_my').output)
out = dense1_layer_model.predict(myimage)
print (type(out.shape))
num = out.shape[1]
print(num)
image_conv=[]
out = out.reshape(32,28,28)
for i in range(num):
    image_conv.append(out[i,:,:].reshape(28,28))
imgs = np.hstack(image_conv)
cv2.imshow("Dense_1_my", imgs)

pool1_layer_model = Model(inputs=model.input, outputs=model.get_layer('Dense_2_my').output)
out = pool1_layer_model.predict(myimage)
print(out.shape)
image_conv2=[]
out = out.reshape(64,14,14)
for i in range(64):
    image_conv2.append(out[i,:,:].reshape(14,14))
imgs2 = np.hstack(image_conv2)
cv2.imshow("Dense_2_my", imgs2)
cv2.waitKey(0)