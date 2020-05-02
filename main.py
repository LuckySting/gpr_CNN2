from keras.callbacks import ModelCheckpoint

from data import data_generator, save_result
from unet import unet

train_generator = data_generator('training_set', 1, 15)
test_generator = data_generator('training_set', 1, 1)

model = unet()

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)

model.fit_generator(train_generator, steps_per_epoch=15, epochs=500, callbacks=[model_checkpoint])
result = model.predict_generator(test_generator, steps=1, verbose=1)
save_result('.', result)
