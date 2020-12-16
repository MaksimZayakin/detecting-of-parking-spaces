import os
import shutil
import random

# Задаем имя будущей сети
modelName = 'Car_park_detector'
# Задаем кол-во эпох обучения.
ep = 50

# Здесь задается кол-во картинок, необходимых для тренеровки, валидации и тестирования сети.
# 500 500 -> 1000 всего => 600 - тренеровочная выборка, 200 - валидационная выборка, 200 - тестовая выборка
num_train_with = 300
num_train_without = 300
num_val_with = 100
num_val_without = 100
num_test_with = 100
num_test_without = 100

# Далее задаем путь к папке, в которой находятся все картики.
base_dir = 'C:\\MyFiles\\Studying\\RIS-20-1m\\Coursework1Semester\\CourseworkPhotos'
parking_dir = os.path.join(base_dir, "Parkings")
notparking_dir = os.path.join(base_dir, "NotParkings")

# Задаем пути для папок с тренеровочными, валидационными и тестовыми картинками
training_dir = os.path.join(base_dir, 'train')
training_dir_parking = os.path.join(training_dir, 'train_parking')
training_dir_notparking = os.path.join(training_dir, 'train_notparking')
validation_dir = os.path.join(base_dir, 'val')
validation_dir_parking = os.path.join(validation_dir, 'validation_parking')
validation_dir_notparking = os.path.join(validation_dir, 'validation_notparking')
test_dir = os.path.join(base_dir, 'test')
test_dir_parking = os.path.join(test_dir, 'test_parking')
test_dir_notparking = os.path.join(test_dir, 'test_notparking')

# Удаление папок при запуске необходимо для того, чтобы если мы переобучаем сеть,
# то вручную не пришлось удалять папки заполненные картиками
shutil.rmtree(training_dir_parking, ignore_errors=True)
shutil.rmtree(training_dir_notparking, ignore_errors=True)
shutil.rmtree(validation_dir_parking, ignore_errors=True)
shutil.rmtree(validation_dir_notparking, ignore_errors=True)
shutil.rmtree(test_dir_parking, ignore_errors=True)
shutil.rmtree(test_dir_notparking, ignore_errors=True)

# Создание папок для хранения картинок по выборкам
os.makedirs(training_dir_parking)
os.makedirs(training_dir_notparking)
os.makedirs(validation_dir_parking)
os.makedirs(validation_dir_notparking)
os.makedirs(test_dir_parking)
os.makedirs(test_dir_notparking)

# Считываем картинки с парковками и без, для того, чтобы перемешать их
# и минимизировать свой субъективный вклад в обучение сети.
parking_imgs = os.listdir(path=parking_dir)
notparking_imgs = os.listdir(path=notparking_dir)
random.shuffle(parking_imgs)
random.shuffle(notparking_imgs)

# Далее раскидываем перемешанные картинки по соответсвующим папкам
for i in range(num_train_with):
    shutil.copy(os.path.join(parking_dir, parking_imgs[i]),
                os.path.join(training_dir_parking, str(parking_imgs[i])))
    shutil.copy(os.path.join(notparking_dir, notparking_imgs[i]),
                os.path.join(training_dir_notparking, str(notparking_imgs[i])))
for i in range(num_train_with, num_train_with + num_val_with):
    shutil.copy(os.path.join(parking_dir, parking_imgs[i]),
                os.path.join(validation_dir_parking, str(parking_imgs[i])))
    shutil.copy(os.path.join(notparking_dir, notparking_imgs[i]),
                os.path.join(validation_dir_notparking, str(notparking_imgs[i])))
for i in range(num_train_with + num_val_with, num_train_with + num_val_with + num_test_with):
    shutil.copy(os.path.join(parking_dir, parking_imgs[i]),
                os.path.join(test_dir_parking, str(parking_imgs[i])))
    shutil.copy(os.path.join(notparking_dir, notparking_imgs[i]),
                os.path.join(test_dir_notparking, str(notparking_imgs[i])))

# TODO: добавить сеть (скорее всего VGG16)

