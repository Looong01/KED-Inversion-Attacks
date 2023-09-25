import os
import shutil

for i in range(1, 10):
    shutil.copy(os.path.join('/home/loong/Codes/KED/Datasets/Celeba/Img/img_align_celeba_png/', '00000' + str(i) + '.png'), '/home/loong/Codes/KED/Datasets/Celeba/Img/new_png/')
for i in range(10, 100):
    shutil.copy(os.path.join('/home/loong/Codes/KED/Datasets/Celeba/Img/img_align_celeba_png/', '0000' + str(i) + '.png'), '/home/loong/Codes/KED/Datasets/Celeba/Img/new_png/')
for i in range(100, 1000):
    shutil.copy(os.path.join('/home/loong/Codes/KED/Datasets/Celeba/Img/img_align_celeba_png/', '000' + str(i) + '.png'), '/home/loong/Codes/KED/Datasets/Celeba/Img/new_png/')
for i in range(1000, 5121):
    shutil.copy(os.path.join('/home/loong/Codes/KED/Datasets/Celeba/Img/img_align_celeba_png/', '00' + str(i) + '.png'), '/home/loong/Codes/KED/Datasets/Celeba/Img/new_png/')

for i in range(162771, 163539):
    shutil.copy(os.path.join('/home/loong/Codes/KED/Datasets/Celeba/Img/img_align_celeba_png/', str(i) + '.png'), '/home/loong/Codes/KED/Datasets/Celeba/Img/new_png/')

for i in range(182638, 183406):
    shutil.copy(os.path.join('/home/loong/Codes/KED/Datasets/Celeba/Img/img_align_celeba_png/', str(i) + '.png'), '/home/loong/Codes/KED/Datasets/Celeba/Img/new_png/')
