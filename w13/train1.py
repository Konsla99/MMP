import tensorflow as tf
from keras import layers, models, optimizers, activations
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt

tr_file = 'D:/code/mmp/w13/test/list_tr'
ts_file = 'D:/code/mmp/w13/test/list_ts'
n_epochs = int(input('Enter number of epochs:'))
b_size = int(input('Enter batch size:'))
n_tag = int(input('Enter number of classes:'))
l_rate = float(input('Enter learning rate:'))
SX = SY = 64

def Getdata(list_file):
    fp = open(list_file, 'r')
    lines = fp.read().splitlines()
    inp = []
    tag = []
    for line in lines:
        img_file, id = line.split(' ')
        img = cv2.imread(img_file)
        img = cv2.resize(img, (SY, SX))
        inp.append(img / 255)
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

# Leaky ReLU 활성화 함수 정의
def leaky_relu(x):
    return activations.relu(x, alpha=0.01)

model = models.Sequential()
model.add(layers.Conv2D(5, (11, 11), activation=leaky_relu, input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Conv2D(7, (7, 7), activation=leaky_relu))
model.add(layers.MaxPooling2D((4, 4)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation=leaky_relu))
model.add(layers.Dense(n_tag, activation='softmax'))

model.summary()

adam = optimizers.Adam(learning_rate=l_rate)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=[percentage_accuracy])

# 학습 과정 기록을 위한 콜백
class HistoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.train_accuracies.append(logs.get('percentage_accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_percentage_accuracy'))

history = HistoryCallback()
model.fit(tr_inp, tr_tag, batch_size=b_size, epochs=n_epochs, verbose=1, shuffle=True, validation_data=(ts_inp, ts_tag), callbacks=[history])

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(ts_inp, ts_tag, verbose=0)
print("Results for test data=")
print(test_acc)

# 예측 및 혼동 행렬 출력
pred = model.predict(ts_inp)
y_pred = np.argmax(pred, axis=1)
conf_mat = confusion_matrix(ts_tag, y_pred)
print(conf_mat)

# 학습 및 테스트 과정의 loss와 accuracy를 그래프로 출력
plt.figure(figsize=(12, 6))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(history.train_losses, label='Train Loss')
plt.plot(history.val_losses, label='validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)  # 격자선 추가


# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(history.train_accuracies, label='Train Accuracy')
plt.plot(history.val_accuracies, label='validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)  # 격자선 추가


plt.show()
