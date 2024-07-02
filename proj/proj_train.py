import tensorflow as tf
from keras import layers, models, optimizers, activations,regularizers
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

tr_file = 'D:/code/mmp/proj/list_tr'
ts_file = 'D:/code/mmp/proj/list_ts'
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

# model = models.Sequential()
# model.add(layers.Conv2D(5, (11, 11), activation=leaky_relu, input_shape=(64, 64, 3)))
# model.add(layers.MaxPooling2D((4, 4)))
# model.add(layers.Conv2D(7, (7, 7), activation=leaky_relu))
# model.add(layers.MaxPooling2D((4, 4)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation=leaky_relu))
# model.add(layers.Dense(n_tag, activation='softmax'))

#model = models.Sequential()

model = models.Sequential()

# # 첫 번째 합성곱 블록
# model.add(layers.Conv2D(32, (3, 3), activation=leaky_relu, input_shape=(64, 64, 3)))
# model.add(layers.MaxPooling2D((2, 2)))  # 풀 사이즈를 (2, 2)로 변경

# # 두 번째 합성곱 블록
# model.add(layers.Conv2D(64, (3, 3), activation=leaky_relu))
# model.add(layers.MaxPooling2D((2, 2)))  # 풀 사이즈를 (2, 2)로 변경

# # 세 번째 합성곱 블록
# model.add(layers.Conv2D(64, (3, 3), activation=leaky_relu))
# model.add(layers.MaxPooling2D((2, 2)))  # 풀 사이즈를 (2, 2)로 변경

# # 네 번째 합성곱 블록
# model.add(layers.Conv2D(128, (3, 3), activation=leaky_relu))
# model.add(layers.MaxPooling2D((2, 2)))  # 풀 사이즈를 (2, 2)로 변경

# # 플래튼 및 Dense 층
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(n_tag, activation='softmax'))



# # 첫 번째 합성곱 블록
# model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
# model.add(layers.LeakyReLU(alpha=0.01))
# model.add(layers.MaxPooling2D((2, 2)))
# #model.add(layers.Dropout(0.3))

# # 두 번째 합성곱 블록
# model.add(layers.Conv2D(64, (3, 3)))
# model.add(layers.LeakyReLU(alpha=0.01))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.3))

# # 세 번째 합성곱 블록
# model.add(layers.Conv2D(128, (3, 3)))
# model.add(layers.LeakyReLU(alpha=0.01))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.3))
# # 네 번째 합성곱 블록
# model.add(layers.Conv2D(128, (3, 3)))
# model.add(layers.LeakyReLU(alpha=0.01))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Dropout(0.3))
# # 플래튼 및 Dense 층
# model.add(layers.Flatten())
# model.add(layers.Dense(512))
# model.add(layers.LeakyReLU(alpha=0.01))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(n_tag, activation='softmax'))

model = models.Sequential()

# 첫 번째 합성곱 블록 (드롭아웃 없음)
model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), kernel_regularizer=regularizers.l2(0.002)))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))

# 두 번째 합성곱 블록
model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.0035)))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# 세 번째 합성곱 블록
model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.005)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# 네 번째 합성곱 블록
model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.006)))
model.add(layers.BatchNormalization())
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# 플래튼 및 Dense 층
model.add(layers.Flatten())
model.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.LeakyReLU(alpha=0.01))
model.add(layers.Dropout(0.5))
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
plt.plot(history.val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)  # 격자선 추가

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(history.train_accuracies, label='Train Accuracy')
plt.plot(history.val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)  # 격자선 추가

plt.show()

# 혼동 행렬 시각화
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_tag), yticklabels=range(n_tag))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
