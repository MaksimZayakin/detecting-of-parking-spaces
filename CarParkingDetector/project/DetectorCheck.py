from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

#  Выполняем ручную проверку сети. Загружаем модель и прогоняем через нее заранее выбранную картинку.
model = load_model('C:\\MyFiles\\Studying\\RIS-20-1m\\Coursework1Semester\\\DetectorAndTestPhotos\\model_Car_park_detector_1_8_50.h5')
pic_size = 256
fileName = 'C:\\MyFiles\\Studying\\RIS-20-1m\\Coursework1Semester\\\DetectorAndTestPhotos\\test_photos\\parking\\516.jpeg'
img = load_img(fileName, target_size=(pic_size, pic_size))
input_arr = img_to_array(img)
input_arr = np.array([input_arr])
input_arr = input_arr.astype('float32')
input_arr /= 255.0
predictions = model.predict(input_arr)