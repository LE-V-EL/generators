import os, sys

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/mask_rcnn/'))
from mrcnn import utils
from mrcnn import visualize

# we need access to Ian's code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/ian/'))
from figure5 import Figure5

import numpy as np
import json
from statistics import mean 


class PartitionedDataset:

    class AngleDataset(utils.Dataset):

        def __init__(self, p_dataset, count):
            '''
            '''
            # allow the inner class to access the outer class's data
            self.p_dataset = p_dataset
            self.count = count
            self.label_distribution = []
            super().__init__()


        def generate(self):
            '''
            '''
            SETNAME = 'stimuli'
            
            self.add_class(SETNAME, 1, "angle")

            for i in range(self.count):

                sparse, mask, angles, parameters = self.next_image()

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
        
        
        def random_image(self):
            '''
            '''
            
            return Figure5.angle(self.p_dataset.flags)



        def next_image(self):
            '''
            '''
            
            sparse, mask, angles, parameters = self.random_image()

            # so we can use the nice numpy operations
            label = np.asarray(angles)

            while not self.validate_label(label):
                sparse, mask, angles, parameters = self.random_image()
                label = np.asarray(angles)

            self.add_label(label)

            return sparse, mask, angles, parameters



        def validate_label(self, label):
            '''
            '''
            return self.p_dataset.check_label_euclid(label) and self.check_distribution(label)



        def check_distribution(self, label):
            '''
            '''

            # we dont care until we reach larger amounts
            if sum(self.label_distribution) < 1000:
                return True
            
            # not adding anything over 110% of the mean amount in each angle bucket
            threshold = mean(self.label_distribution) * 1.1
            
            for element in label:
                
                if element > len(self.label_distribution):
                    self.extend_label_distribution(element)
                    # recalculate after extending
                    threshold = mean(self.label_distribution) * 1.1


                if self.label_distribution[element - 1] > threshold:
                    return False
            
            return True



        def extend_label_distribution(self, element):
            '''
            '''
            
            while len(self.label_distribution) < element:
                self.label_distribution.append(0)



        def add_label(self, label):
            '''
            '''

            for element in label:
                
                if element > len(self.label_distribution):
                    self.extend_label_distribution(element)

                self.label_distribution[element - 1] += 1

            self.p_dataset.add_label(label)



        @staticmethod
        def show(which, howmany=4):
            '''
            '''

            image_ids = np.random.choice(which.image_ids, howmany)
            for image_id in image_ids:
                image = which.load_image(image_id)
                mask, class_ids = which.load_mask(image_id)
                visualize.display_top_masks(image, mask, class_ids, which.class_names)



    def __init__(self, 
        counts             = {"train": 500, "val": 50, "test": 50}, 
        flags              = [True,False,False], 
        distance_threshold = 5):
        '''
        '''

        self.flags = flags
        self.distance_threshold = distance_threshold


        self.__dataset = {}
        self.labels = []

        for key in counts:
            self.__dataset[key] = self.AngleDataset(self, counts[key])
            self.__dataset[key].generate()
            self.__dataset[key].prepare()



    def dataset(self, name):
        '''
        '''
        return self.__dataset[name]



    def check_label_euclid(self, label):
        '''
        '''
        for existing_label in self.labels:
            dist = np.linalg.norm(existing_label - label)
            if dist < self.distance_threshold:
                return False
        return True



    def add_label(self, label):
        '''
        '''
        self.labels.append(label)



