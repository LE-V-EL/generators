import os, sys

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/mask_rcnn/'))
from mrcnn import utils
from mrcnn import visualize

# we need access to Ian's code
#sys.path.append(os.path.join(os.path.dirname(__file__), 'external/ian/'))
from angle import Figure5

import numpy as np
import json
from operator import add, sub
from datetime import datetime
from statistics import mean 
import pickle

class PartitionedDataset:

    class SubDataset(utils.Dataset):

        def __init__(self, p_dataset, count):
            '''
            '''
            # allow the inner class to access the outer class's data
            self.p_dataset = p_dataset
            self.count = count
            self.label_distribution = {}
            
            self.label = []
            self.image = []
            self.mask  = []
            self.bbox  = []
            
            super().__init__()


        def generate(self):
            '''
            '''
            SETNAME = 'stimuli'
            
            self.add_class(SETNAME, 1, self.p_dataset.mode)

            for i in range(self.count):

                sparse, mask, label, parameters = self.next_image()

                image = mask.copy()
                image[image>0] = 1 # re-set to binary

                if self.p_dataset.to_file:

                    self.label.append(label)
                    self.image.append(image)
                    self.mask.append(mask)
                    
                    # copied from maskrcnn/utils.py
                    n_mask = np.array(mask)
                    _idx = np.sum(n_mask, axis=(0, 1)) > 0
                    n_mask = n_mask[:, :, _idx]

                    bbox = utils.extract_bboxes(n_mask)

                    self.bbox.append(bbox)
                
                else:
                    self.add_image(SETNAME, image_id=i, path=None,
                                   image=image, sparse=sparse, parameters=parameters,
                                   mask=mask,
                                   angles=label)

                
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

            image_function = getattr(self.p_dataset, self.p_dataset.mode)
            
            return image_function(flags=self.p_dataset.flags)



        def next_image(self):
            '''
            '''
            
            sparse, mask, label, parameters = self.random_image()

            # so we can use the nice numpy operations
            label = np.asarray(label)
            
            self.p_dataset.add_itteration()
            
            while not self.validate_label(label):
                
                sparse, mask, label, parameters = self.random_image()
                
                label = np.asarray(label)
                
                self.p_dataset.add_itteration()

            self.add_label(label)

            return sparse, mask, label, parameters



        def validate_label(self, label):
            '''
            '''
            return self.p_dataset.check_label_euclid(label) and self.check_distribution(label)



        def check_distribution(self, label):
            '''
            '''

            # we dont care until we reach larger amounts
            if sum(self.label_distribution.values()) < 1000:
                return True
            
            # not adding anything over 110% of the mean amount in each angle bucket
            threshold = mean(self.label_distribution.values()) * 1.1
            
            for element in label:

                if self.label_distribution[str(element)] > threshold:
                    return False
            
            return True



        def add_label(self, label):
            '''
            '''

            for element in label:

                self.label_distribution[str(element)] = 1 + self.label_distribution.get(str(element), 0)

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

    #####################################################################################
    # PartitionedDataset
    #####################################################################################

    mode_list = [
        'position_non_aligned_scale',
        'position_common_scale',
        'angle',
        'length',
        'direction',
        'area',
        'volume',
        'curvature'
    ]

    def __init__(self, 
        counts             = {"train": 500, "val": 50, "test": 50}, 
        flags              = [True,False,False], 
        distance_threshold = 3.0,
        naive              = False,
        to_file            = True,
        batch              = True,
        mode               = 'angle'):
        '''
        '''

        self.counts             = counts
        self.flags              = flags
        self.distance_threshold = distance_threshold
        self.naive              = naive
        self.to_file            = to_file
        self.batch              = batch 
        self.mode               = mode

        self.__dataset    = {}
        self.labels       = []
        self.euclid_table = {}
        self.itterations  = 0



        for function in PartitionedDataset.mode_list:
            setattr(self, function, getattr(Figure5, function))



    def generate(self):

        startTime = datetime.now()

        for key in self.counts:

            if self.to_file:

                self.generate_to_file(key)

            else:
                self.__dataset[key] = self.SubDataset(self, self.counts[key])
                self.__dataset[key].generate()
                self.__dataset[key].prepare()

            print("Finished Generating: ", key)

        print("Evaluation time ", datetime.now() - startTime)



    def generate_to_file(self, key):
        '''
        '''

        folder = "output/" + self.mode + "/"

        if not os.path.exists(folder):
            os.makedirs(folder)

        count = self.counts[key]
        current_dataset = 0
        
        while count > 0:
            
            dataset_count = 0

            if not self.batch:
                dataset_count = count
            elif count >= 10000:
                dataset_count = 10000
            else:
                dataset_count = count
            
            dataset = self.SubDataset(self, dataset_count)
            dataset.generate()
            dataset.prepare()
            dataset.p_dataset = None


            data = []


            for attribute in ['image', 'mask', 'label', 'bbox']:

                data.append(getattr(dataset, attribute))



            file = folder + key + "_" + str(current_dataset) + ".npy"
            
            np.save(file, np.asarray(data))



            del dataset
            
            if self.batch:
                count -= 10000
            else:
                count = 0
            current_dataset += 1



    def dataset(self, name):
        '''
        '''
        return self.__dataset[name]



    def check_label_euclid(self, label):
        '''
        Function to check if a label is within a certain euclidian distance
        of a label already added to the dataset, and so be invalid to add.
        This can be done in a naive fashion or one where all invalid labels
        are memoized.
        '''
        if self.naive:
            return self.check_label_euclid_naive(label)
        else:
            return self.check_label_euclid_memo(label)


    def check_label_euclid_naive(self, label, label_set=None, print_failure=False):
        '''
        '''
        if label_set is None:
            label_set = self.labels

        for existing_label in label_set:
            dist = np.linalg.norm(existing_label - label)
            if dist < self.distance_threshold:
                if print_failure:
                    print("Naive Validation Failure")
                    print("Label to Add :", label)
                    print("Existing Label: ", existing_label)
                return False
        return True



    def check_label_euclid_memo(self, label):
        
        return not self.euclid_table.get("-".join(label.astype(str)))



    def validate_labels(self):
        '''
        This checks if all labels in the PartitonedDataset are not within
        the euclidian distance threshold of each other, useful for testing 
        that the memoized process for adding labels to the table is working
        properly.
        '''
        for index in range(len(self.labels) - 1):

            if not index % 1000:
                print("validating: ", index)

            if not self.check_label_euclid_naive(self.labels[index], self.labels[index + 1:], print_failure=True):
                return False
        
        return True


    def add_label(self, label):
        '''
        Adds label to overall PartitionedDataset. If it is not in naive mode
        then all of the potencial labels that are within the euclidian distance
        threshold of the label to add are calculated and added to the memoized
        lookup table.
        '''
        self.labels.append(label)

        if not len(self.labels) % 1000:
            print("labels: ", len(self.labels))
        
        if not self.naive:
            self.add_labels_within_threshold(label)



    def add_labels_within_threshold(self, label):
        '''
        Adds all possibly labels within the euclidian distance threshold
        to the memoized euclidian distance lookup table.
        '''
        self.__add_labels_within_threshold(label, label, 0, 0)



    def __add_labels_within_threshold(self, base_label, current_label, index, old_dist):

        for op in [add, sub]:

            dist = old_dist

            next_label = current_label.copy();

            while dist < self.distance_threshold:


                self.add_euclid_label(next_label)

                if index + 1 < len(base_label):
                    self.__add_labels_within_threshold(base_label, next_label.copy(), index + 1, dist)

                next_label[index] = op(next_label[index], 1)

                dist = np.linalg.norm(base_label - next_label)



    def add_euclid_label(self, label):
        '''
        Adds a label to the euclidian distance lookup table.
        '''
        self.euclid_table["-".join(label.astype(str))] = True



    def add_itteration(self):
        self.itterations += 1
        if not self.itterations % 1000:
            print("itteration: ", self.itterations)



