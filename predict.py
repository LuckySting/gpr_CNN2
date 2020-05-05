from data import data_generator, save_result
from unet import unet

test_generator = data_generator('training_set', 215, 2)

model = unet()
weight_name = 'first_generation_3'
model.load_weights('{}.hdf5'.format(weight_name))
result = model.predict_generator(test_generator, steps=2, verbose=1)
save_result('./{}'.format(weight_name), result, 215)

