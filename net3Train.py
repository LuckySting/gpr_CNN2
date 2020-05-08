import json

from keras.callbacks import ModelCheckpoint

from data import data_generator2
from net3 import net3

file = open('training_set6\in\y.json')
y_dict = json.load(file)
file.close()

train_generator = data_generator2(y_dict, 'training_set6', 0, 410, flip=True)
test_generator = data_generator2(y_dict, 'training_set6', 410, 10, flip=True)

weight_name = 'third_generation_1'

model = net3()

model_checkpoint = ModelCheckpoint('{}.hdf5'.format(weight_name), monitor='loss', verbose=1, save_best_only=True)

model.fit_generator(train_generator, steps_per_epoch=410, epochs=200, callbacks=[model_checkpoint])
result = model.predict_generator(test_generator, steps=5, verbose=1)
