import os, sys

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/mask_rcnn/'))
from mrcnn import utils
from mrcnn import visualize

# we need access to Ian's code
sys.path.append(os.path.join(os.path.dirname(__file__), '../../external/ian/'))
import figure5

import numpy as np




class AngleDataset(utils.Dataset):
    
    def generate(self, count, flags=[True,False,False]):
        '''
        '''
        SETNAME = 'stimuli'
        
        self.add_class(SETNAME, 1, "angle")

        for i in range(count):
            sparse, mask, angles, parameters = self.random_image(flags)
            img = mask.copy()
            img[img>0] = 1 # re-set to binary
            self.add_image(SETNAME, image_id=i, path=None,
                          image=img, sparse=sparse, parameters=parameters,
                          mask=mask,
                          angles=angles)

            
    def load_image(self, image_id):
        '''
        '''
        info = self.image_info[image_id]

        image = info['image']
        
        loaded_img_3D = np.stack((image,)*3, -1)
        
        return (loaded_img_3D*255).astype(np.uint8)
        
            
    def load_mask(self, image_id):
        '''
        '''
        info = self.image_info[image_id]
        mask = info['mask']
        
        mask2 = np.zeros((mask.shape[0],mask.shape[1], 4), dtype=np.uint8)

        for i in range(0,4):
            filtered_mask = mask.copy()
            filtered_mask[filtered_mask!=(i+1)] = 0
            filtered_mask[filtered_mask==(i+1)] = 1
            mask2[:,:, i] = filtered_mask
        
        return mask2, np.array([1,1,1,1]) # it is always class 1 but four times 
                                        # and each channel of the mask needs to have a single angle
    
    
    def random_image(self,flags=[True,False,False]):
        '''
        '''
        
        return figure5.Figure5.angle(flags)


    @staticmethod
    def createSets(howmany=(500,50,50), flags=[True,False,False]):

        dataset_train = AngleDataset()
        dataset_train.generate(howmany[0], flags)
        dataset_train.prepare()

        dataset_val = AngleDataset()
        dataset_val.generate(howmany[1], flags)
        dataset_val.prepare()

        dataset_test = AngleDataset()
        dataset_test.generate(howmany[2], flags)
        dataset_test.prepare()

        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def show(which, howmany=4):

        image_ids = np.random.choice(which.image_ids, howmany)
        for image_id in image_ids:
            image = which.load_image(image_id)
            mask, class_ids = which.load_mask(image_id)
            visualize.display_top_masks(image, mask, class_ids, which.class_names)

