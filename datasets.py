import os, sys

import numpy as np

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/mask_rcnn/'))
from mrcnn import utils
from mrcnn import visualize

from angle import Figure5

from operator import add, sub
from datetime import datetime
from statistics import mean


class DatasetGenerator:


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

    dataset_components = [
        'label',
        'image',
        'mask',
        'bbox',
        'sparse',
        'parameters'
    ]

#####################################################################################
# SubDataset
#####################################################################################

    class SubDataset:


        def __init__(self, p_dataset, count):
            '''
            '''
            # allow the inner class to access the outer class's data
            self.p_dataset = p_dataset
            self.count = count
            self.label_distribution = {}

            for component in DatasetGenerator.dataset_components:
                setattr(self, component, [])

        def generate(self):
            '''
            '''
            for i in range(self.count):

                sparse, premask, label, parameters = self.next_image()

                sparse = np.asarray(sparse, dtype='uint16')

                image = np.asarray(premask.copy(), dtype='float32')
                image[image>0] = 1 # re-set to binary

                image =  np.asarray(image, dtype='float32')

                # adding 5% noise
                image += np.random.uniform(0, 0.05, image.shape)

                mask = np.zeros((premask.shape[0], premask.shape[1], len(label)), dtype=np.uint8)

                sparse = np.asarray(sparse, dtype="uint16")

                for i in range(len(label)):
                    filtered_mask = premask.copy()
                    filtered_mask[filtered_mask!=(i+1)] = 0
                    filtered_mask[filtered_mask==(i+1)] = 1
                    mask[:,:, i] = filtered_mask

                self.label.append(label)
                self.image.append(image)
                self.mask.append(mask)
                self.sparse.append(sparse)

                bbox = np.asarray(utils.extract_bboxes(mask), dtype='uint16')

                self.bbox.append(bbox)

                self.parameters.append(parameters)



        def random_image(self):
            '''
            '''
            image_function = getattr(self.p_dataset, self.p_dataset.mode)
            
            return image_function(flags=self.p_dataset.flags)



        def next_image(self):
            '''
            '''
            sparse, premask, label, parameters = self.random_image()

            # so we can use the nice numpy operations
            label = np.asarray(label, dtype='uint8')
            
            self.p_dataset.add_itteration()
            
            while not self.validate_label(label):
                
                sparse, premask, label, parameters = self.random_image()
                
                label = np.asarray(label, dtype='uint8')
                
                self.p_dataset.add_itteration()

            self.add_label(label)

            return sparse, premask, label, parameters



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

                if self.label_distribution.get(str(element), 0) > threshold:
                    return False
            
            return True



        def add_label(self, label):
            '''
            '''

            for element in label:

                self.label_distribution[str(element)] = 1 + self.label_distribution.get(str(element), 0)

            self.p_dataset.add_label(label)


#####################################################################################
# DatasetGenerator
#####################################################################################

    def __init__(self,
        counts             = {"train": 500, "val": 50, "test": 50},
        flags              = [True,False,False],
        distance_threshold = 3.0,
        naive              = False,
        batch              = True,
        mode               = 'angle'):

        self.counts             = counts
        self.flags              = flags
        self.distance_threshold = distance_threshold
        self.naive              = naive
        self.batch              = batch
        self.mode               = mode

        self.folder = "output/" + self.mode + "/"

        self.labels       = []
        self.euclid_table = {}
        self.itterations  = 0

        for function in DatasetGenerator.mode_list:
            setattr(self, function, getattr(Figure5, function))



    def generate(self):
        '''
        '''
        startTime = datetime.now()

        print("Generating dataset of class: ", self.mode)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        for key in self.counts:

            count = self.counts[key]
            dataset_number = 0

            while count > 0:
                
                dataset_count = 0

                if not self.batch:
                    dataset_count = count
                elif count >= 10000:
                    dataset_count = 10000
                else:
                    dataset_count = count

                self.generate_subdataset_and_save(key, dataset_count, dataset_number)
                
                if self.batch:
                    count -= 10000
                else:
                    count = 0
                dataset_number += 1

            print("Finished Generating: ", key)

        print("Evaluation time ", datetime.now() - startTime)



    def generate_subdataset_and_save(self, name, dataset_count, dataset_number):
        '''
        '''
        dataset = self.SubDataset(self, dataset_count)
        dataset.generate()
        dataset.p_dataset = None

        # pulling out data and transforming it to save
        save_data = dict( map(
            lambda component : ( component, np.stack( getattr(dataset, component), axis=0 ) ),
            DatasetGenerator.dataset_components
        ))

        file = self.folder + name + "_" + str(dataset_number) + ".npz"

        np.savez(file, **save_data)

        del dataset




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
        This checks if all labels in the DatasetGenerator are not within
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
        Adds label to overall DatasetGenerator. If it is not in naive mode
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

        # exception for length because there is too little space to do 
        # the full euclidian distance caluclation and allow for a dataset
        # of any size
        if self.mode == "length":
            self.add_euclid_label(label)
        else:
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



#####################################################################################
# DatasetFromFile
#####################################################################################

class DatasetFromFile(utils.Dataset):

    def __init__(self, file):
        
        super().__init__()

        self.file = file



    def load_from_file(self):
        '''
        '''
        data = np.load(self.file)

        # pulling all data out of npz file
        loaded_data = dict(map(lambda component: ( component, data[component] ), data.keys()))

        SETNAME = 'stimuli'
        
        self.add_class(SETNAME, 1, "angle")

        for i in range(len(loaded_data['label'])):

            # pulling out all the info for one image
            image_data = dict( map(lambda component: ( component, loaded_data[component][i] ), loaded_data.keys()))

            self.add_image(SETNAME,
                image_id   = i,
                path       = None,
                **image_data)

        del data

        self.prepare()

        return self



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
        mask = self.image_info[image_id]['mask']

        # it is always class 1 but for the amount of stimuli
        return mask, np.ones(mask.shape[2], dtype='uint8')



    def show(self, howmany=1):
        '''
        '''
        image_ids = np.random.choice(self.image_ids, howmany)
        
        for image_id in image_ids:
            
            image = self.load_image(image_id)
            
            mask, class_ids = self.load_mask(image_id)

            visualize.display_top_masks(image, mask, class_ids, self.class_names)



    def show_bbox(self, image_id):
        '''
        '''
        image = self.load_image(image_id)

        mask, class_ids = self.load_mask(image_id)

        bbox = self.image_info[image_id]['bbox']

        visualize.display_instances(image, bbox, mask, class_ids,
                            self.class_names, figsize=(8, 8))
