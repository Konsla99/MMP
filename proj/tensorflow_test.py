import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers
# https://discuss.tensorflow.org/t/import-tensorflow-keras-shows-lint-error/17386
# 둘은 이제 별도의 모듈이라 keras 에서 가져올것
# from tensorflow import keras로  tensorflow에서 가져올수 있음 혹은 그냥 keras

# from tensorflow import keras
from keras import layers, models, optimizers
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2

tr_file = 'D:/code/mmp/proj/list_tr'
ts_file = 'D:/code/mmp/proj/list_ts'
n_epochs = int(input('Enter number of epochs:'))
b_size = int(input('Enter batch size:'))
n_tag = int(input('Enter number of classes:'))
l_rate = float(input('Enter learning rate:'))
SX = SY =64

def Getdata(list_file):
    fp = open(list_file, 'r')
    lines = fp.read().splitlines()
    inp=[]
    tag=[]
    for line in lines:
        img_file, id = line.split(' ')
        img = cv2.imread(img_file)
        img = cv2.resize(img, (SY,SX))
        inp.append(img/255)
        tag.append(np.uint8(id))
    fp.close()
    return inp, tag

def percentage_accuracy(y_true, y_pred):
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    return accuracy * 100

[tr_inp, tr_tag] = Getdata(tr_file)
tr_inp = np.array(tr_inp)
tr_tag = np.array(tr_tag)

[ts_inp, ts_tag] = Getdata(ts_file)
ts_inp = np.array(ts_inp)
ts_tag = np.array(ts_tag)

model = models.Sequential()
model.add(layers.Conv2D(5, (11, 11), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(7, (7, 7), activation='relu'))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(n_tag, activation='softmax'))

model.summary()

adam = optimizers.Adam(learning_rate=l_rate)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[percentage_accuracy])
model.fit(tr_inp, tr_tag, batch_size=b_size, epochs=n_epochs, verbose=1, shuffle=1)
test_loss, test_acc = model.evaluate(ts_inp, ts_tag, verbose=0)
print("Results for test data=")
print(test_acc)


pred = model.predict(ts_inp)
y_pred = np.argmax(pred, axis=1)
conf_mat = confusion_matrix(ts_tag, y_pred)
print(conf_mat)
