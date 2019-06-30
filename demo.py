import aug as am
import Helpers as hp
from util import *
import os
from os.path import join
from tqdm import tqdm

base_dir = os.getcwd()
data_base_dir = join(base_dir,'img')
save_base_dir = join(base_dir,'save')
save_crop_base_dir = join(base_dir,'save_crop')
save_annoation_base_dir = join(base_dir,'save_annotation')
check_dir(save_base_dir)
#check_dir(save_crop_base_dir)
#check_dir(save_annoation_base_dir)
imgs_dir = [f.strip() for f in open(join(base_dir,'train.txt')).readlines()]
labels_dir = hp.replace_labels(imgs_dir)
for image_dir,label_dir in tqdm(zip(imgs_dir,labels_dir)):
    am.copysmallobjects(image_dir,label_dir,save_base_dir,save_crop_base_dir,save_annoation_base_dir)
