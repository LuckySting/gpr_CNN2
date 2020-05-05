from data import data_generator, save_result
from unet import unet

test_generator = data_generator('training_set', 121, 3)

model = unet()
weight_name = 'first_generation_2'
model.load_weights('{}.hdf5'.format(weight_name))
result = model.predict_generator(test_generator, steps=3, verbose=1)
save_result('./{}'.format(weight_name), result, 121)

