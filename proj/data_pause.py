import cv2
import numpy as np
import time
# D:/code/mmp/proj/images/
base_name = input('Enter base file name:')
delay = float(input('Enter capture delay (in sec):'))
num_class = int(input('Enter number of classes:'))
num_files = int(input('Enter number of images per class:'))
list_file = input('Enter list file name:')

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

fp = open(list_file, 'w')
write = False
pause = False
print("Press w to start..")
count_img = count_class = 0

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("captured", frame)
        if write and not pause:
            fname = base_name + ('_%03d_%03d' % (count_img, count_class)) + '.jpg'
            cv2.imwrite(fname, frame)
            print(fname)
            print('%s %d' % (fname, count_class), file=fp)
            count_img += 1

            # 1/6마다 일시정지
            if count_img % (num_files // 6) == 0 and count_img != 0:
                pause = True
                print('Paused. Press p to continue...')

    if count_img == num_files:
        count_img = 0
        count_class += 1
        if count_class == num_class:
            break
        write = False
        print('Press w to restart')

    key = cv2.waitKey(33)
    if key == ord('w'):
        write = True
    elif key == ord('p'):
        pause = not pause  # p 키를 누르면 pause 상태를 토글합니다.

    time.sleep(delay)

fp.close()
cap.release()
cv2.destroyAllWindows()
