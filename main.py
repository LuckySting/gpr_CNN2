from keras.callbacks import ModelCheckpoint

from data import data_generator, save_result
from unet import unet

train_generator = data_generator('training_set', 1, 200, flip=True)
test_generator = data_generator('training_set', 205, 5, flip=True)

weight_name = 'first_generation_4'

model = unet()

model_checkpoint = ModelCheckpoint('{}.hdf5'.format(weight_name), monitor='loss', verbose=1, save_best_only=True)

model.fit_generator(train_generator, steps_per_epoch=200, epochs=200, callbacks=[model_checkpoint])
result = model.predict_generator(test_generator, steps=5, verbose=1)

save_result(result, start=205, path='./{}'.format(weight_name))
