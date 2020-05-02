from keras.callbacks import ModelCheckpoint

from data import data_generator, save_result
from unet import unet

test_generator = data_generator('training_set', 1, 1)

model = unet()

model.load_weights('unet_membrane.hdf5')
result = model.predict_generator(test_generator, steps=1, verbose=1)
save_result('.', result)
