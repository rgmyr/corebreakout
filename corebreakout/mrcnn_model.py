"""Dataset + Config + Model definition"""
import os
import glob
import numpy as np
from skimage import io

from mrcnn.utils import Dataset
from mrcnn.config import Config
import mrcnn.model as modellib
"""
from corebreakout.utils import make_labels

DROPBOX_DIR = '/home/'+os.environ['USER']+'/Dropbox/core_data/'
#DROPBOX_DIR = '/home/ross/Dropbox/core_data/'
DATA_DIR = DROPBOX_DIR + 'segmentation/blue_mask/'

#+++++++++++++++#
# Configuration #
#+++++++++++++++#

class CoreConfig(Config):
    '''Override some base config values.'''
    NAME = 'core'

    # TODO: STD_DEVs?

    # Model Config
    NUM_CLASSES = 1 + 1 # (background + column) ... add tablet / missing?
    BACKBONE = 'resnet101'
    RPN_ANCHOR_SCALES = (64, 128, 192, 320, 352)
    RPN_ANCHOR_RATIOS = [1, 4, 7] # [0.5, 1, 2]
    RPN_NMS_THRESHOLD = 0.7 # 0.7

    DETECTION_MAX_INSTANCES = 6
    DETECTION_MIN_CONFIDENCE = 0.95

    # Training Config
    STEPS_PER_EPOCH = 76
    VALIDATION_STEPS = 5

    # ... for Ross' home
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # ... for CoRE machine
    #GPU_COUNT = 2
    #IMAGES_PER_GPU = 1 # (Maybe can get away with 2?)

def yaml_to_CoreConfig(configpath):
    '''Create a CoreConfig from saved yaml file.'''
    pass


#++++++++++++++++++++#
# Dataset Definition #
#++++++++++++++++++++#

class CoreDataset(Dataset):

    def load_core_images(self, data_dir, subset):

        self.add_class("core", 1, "core_column") # self.add_class("core", 2, "info_tablet")

        assert subset in ['coarser_train', 'train', 'val']
        data_dir = os.path.join(data_dir, subset)

        img_files = [f for f in sorted(glob.glob(data_dir+'/*.jp*g')) if 'Label' not in f]

        for f in img_files:
            self.add_image('core', image_id=f.split('/')[-1].split('.')[0], path=f)

    def load_mask(self, image_id):

        img_path = self.image_info[image_id]['path']
        root, ext = tuple(img_path.split('.')[-2:])
        mask_path = root+'_BlueLabel.'+ext

        labels = make_labels(io.imread(mask_path))
        unique_l = np.unique(labels).tolist()[1:]

        mask = np.zeros([labels.shape[0],labels.shape[1],len(unique_l)], dtype=np.bool)
        for i, l in enumerate(unique_l):
            mask[:,:,i] = (labels == l)

        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        '''Return the path of the image.'''
        info = self.image_info[image_id]
        if info["source"] == "core":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


#++++++++++++#
#    Model   #
#++++++++++++#

def load_model(model_path, weights_path):
    config = CoreConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_path)
    model.load_weights(weights_path, by_name=True)
    return model
"""
