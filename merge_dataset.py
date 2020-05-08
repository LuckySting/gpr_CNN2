import os
import re
from random import Random
from shutil import copyfile

random = Random()

def merge_dataset(path1, path2, output):
    dir1_x = [os.path.join(path1, 'x', p) for p in os.listdir(os.path.join(path1, 'x'))]
    dir1_y = [os.path.join(path1, 'y', p) for p in os.listdir(os.path.join(path1, 'y'))]
    dir2_x = [os.path.join(path2, 'x', p) for p in os.listdir(os.path.join(path2, 'x'))]
    dir2_y = [os.path.join(path2, 'y', p) for p in os.listdir(os.path.join(path2, 'y'))]
    files_x = (dir1_x + dir2_x)
    files_y = (dir1_y + dir2_y)
    random.shuffle(files_x)
    random.shuffle(files_y)
    for i in range(len(files_x)):
        f1 = files_x[i].split(os.path.sep)
        f1[0] = output
        f1[-1] = re.sub('\d+', str(i+1), f1[-1])
        f1 = os.path.join(*f1)
        f2 = files_y[i].split(os.path.sep)
        f2[0] = output
        f2[-1] = re.sub('\d+', str(i+1), f2[-1])
        f2 = os.path.join(*f2)
        copyfile(files_x[i], f1)
        copyfile(files_y[i], f2)





merge_dataset('training_set3', 'training_set4', 'training_set5')
