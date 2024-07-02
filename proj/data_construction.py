import cv2
import numpy as np
import time

base_name = input('Enter base file name:')
delay = float(input('Enter capture delay (in sec):'))
num_class = int(input('Enter number of classes:'))
num_files = int(input('Enter number of images pre class:'))
list_file = input('Enter list file name:')

cap = cv2.VideoCapture(0,cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

fp = open(list_file,'w')
write = False
print("Press w to start..")
count_img = count_class = 0

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("captured", frame)
        if write:
            fname = base_name + ('_%03d_%03d'%(count_img, count_class)) + '.jpg'
            cv2.imwrite(fname, frame)
            print(fname)
            print('%s %d'%(fname, count_class), file=fp)
            count_img += 1

    if count_img==num_files:
        count_img = 0
        count_class += 1
        if count_class == num_class:
            break
        write = False
        print('Press w to restart')
    key=cv2.waitKey(33)
    if key == ord('w'):
        write = True
    
    time.sleep(delay)

fp.close()
cap.release()
cv2.destroyAllWindows()

# model = models.Sequential()
# model.add(layers.Conv2D(5, (11, 11), activation='relu', input_shape=(64, 64, 3)))
# model.add(layers.MaxPooling2D((4, 4)))
# model.add(layers.Conv2D(7, (7, 7), activation='relu'))
# model.add(layers.MaxPooling2D((4, 4)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, activation='relu'))
# model.add(layers.Dense(n_tag, activation='softmax'))
