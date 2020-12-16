import os
import shutil
import random
import pickle
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

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

# Для проверки добавляем вывод информации о кол-ве картинок
print('''Number of "parking" images:''', len(os.listdir(parking_dir)))
print('''Number of "not parking" images:''', len(os.listdir(notparking_dir)))

print('''Number of "training parking" images:''', len(os.listdir(training_dir_parking)))
print('''Number of "training not parking" images:''', len(os.listdir(training_dir_notparking)))

print('''Number of "validation parking" images:''', len(os.listdir(validation_dir_parking)))
print('''Number of "validation not parking" images:''', len(os.listdir(validation_dir_notparking)))

print('''Number of "test parking" images:''', len(os.listdir(test_dir_parking)))
print('''Number of "test not parking" images:''', len(os.listdir(test_dir_notparking)))

# Загружаем VGG16 (сверточная сеть для выделения признаков изображений), предворительно откидываем полносвязные слои
# и оставляем только сверточную основу. Полносвязные слои, которые отчечают за классификацию будем добавлять вручную
conv_base = VGG16(weights='imagenet',
                  include_top=False,  # откидываем верхние слои, отвечающие за классификацию.(полносвязные)
                  input_shape=(256, 256, 3))  # width height rgb


last_not_trainable_layer = 8  # Первые слои, которые отвечают за выделение простейших признаков, подходящих для любой
# задачи. Остальные слои будем переобучать

# Даем возможность изменять слои сверточной основы, за исплючение первых last_not_trainable_layer слоев.
conv_base.trainable = True
for layer in conv_base.layers[:last_not_trainable_layer]:
    layer.trainable = False
conv_base.summary()

# Далее сохраняем в файл описание архитектуры сверточной основы и число обучаемых параметров.
with open(modelName + '_conv_base_' + str(ep) + '.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    conv_base.summary(print_fn=lambda x: fh.write(x + '\n'))

# Теперь создаем модель, в которой будем объединять сверточные слои из VGG16 с другими слоями.
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())  # перевод картинок в вектор
model.add(layers.Dropout(0.5))  # отбросить половину нейронов, чтобы не переобучить нейросеть
model.add(layers.Dense(512, activation='relu'))  # функция активации релу, популярная функция активация при классификации
model.add(layers.Dense(1, activation='sigmoid'))  # функция активации сигмоид на последнем слое для бинарной классификации
model.summary()

# создаем картинку модели, а также сохраняем описание архитектуры полученныой нейросети
plot_model(model, to_file=modelName + '.png', show_shapes=True)
with open(modelName + '_' + str(ep) + '.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# производим настройку генераторов, которые используеются в функции обученя.
# Генератор подает картинки, изначально преобразуя их.
# Здесь сделано разделение на два генератора, для того, чтобы в дальшейнем можно было независимо
# корректирваоть их параметры, такие как
# rotation_range,
# width_shift_range,
# height_shift_range,
# shear_range,
# zoom_range,
# horizontal_flip,
# fill_mode
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# В генераторе будем задавать путь к директории с соответствующими картинками, размер картинок,
# кол-во изображений загруженных за раз (оптимально сделать чтобы batch_size был разве кол-ву картинок в
# папке, но это сильно повлияет на скорость обучения, поэтому выберем просто кратный размер),
# а также зададим бинарную классификацию
training_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(256, 256),
    batch_size=30,
    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=20,
    class_mode='binary')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=20,
    class_mode='binary')

# методом compile задаем параметры обучения сети.
# задаем функцию потерь – binary_crossentropy, оптимизатор – RMSprop (метод оптимизации), а также метрику по точности
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

# steps_per_epoch - сколько обращений к генератору картинок за эпоху (кол-во картинок деленное на batch_size).
history = model.fit_generator(
    training_generator,
    steps_per_epoch=20,
    epochs=ep,
    validation_data=validation_generator,
    validation_steps=10)  # то же самое что и steps_per_epoch, только для валидационных данных

# Прогонка сети по тестовым данным
test_loss, test_acc = model.evaluate_generator(test_generator, steps=10)
print('test acc:', test_acc)
f = open(modelName + '_' + str(ep) + '.txt', 'a')
f.write('test acc, loss: ' + str(test_acc) + ', ' + str(test_loss))


# Добавляем функцию для сглаживания графиков
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


with open('history_' + modelName + '_' + str(ep) + '.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Строим график точности
plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.minorticks_on()
plt.grid(which='major', color='k', linewidth=2)
plt.grid(which='minor', color='k', linestyle=':')
plt.legend()
plt.show()
plt.savefig('acc_' + modelName + '_' + str(ep) + '.pdf')
plt.figure()

# Строим график ошибки
plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.minorticks_on()
plt.grid(which='major', color='k', linewidth=2)
plt.grid(which='minor', color='k', linestyle=':')
plt.legend()
plt.show()
plt.savefig('loss_' + modelName + '_' + str(ep) + '.pdf')

# сохраняем сеть
model.save('model_' + modelName + '_1:' + str(last_not_trainable_layer) + '_' + str(ep) + '.h5')

