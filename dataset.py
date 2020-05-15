import os, sys

import numpy as np
import pickle

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/mask_rcnn/'))
from mrcnn import utils
from mrcnn import visualize

from figure5 import Figure5

from operator import add, sub
from datetime import datetime
from statistics import mean
from math import ceil

class DatasetGenerator:


    data_class_list = [
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
        'sparse'
    ]

    dataset_metadata = [
        'parameters',
        'data_class',
        'distance_threshold',
        'count',
        'source',
        'image_size'
    ]

    max_distance_threshold = 2

#####################################################################################
# SubDataset
#####################################################################################

    class SubDataset:


        def __init__(self, p_dataset, count):
            '''
            '''
            # allow the inner class to access the outer class's data
            self.p_dataset = p_dataset

            # metadata to save later
            self.data_class         = self.p_dataset.data_class
            self.distance_threshold = self.p_dataset.distance_threshold
            self.parameters         = None
            self.label_distribution = {}
            self.count              = count
            self.source             = 'LE-V-EL.org'
            self.image_size         = Figure5.SIZE

            # dataset components
            for component in DatasetGenerator.dataset_components:
                setattr(self, component, [])

            if self.p_dataset.data_class == 'area' or self.p_dataset.data_class == 'curvature':
                self.label_type = 'float32'
            else:
                self.label_type = 'uint16'


        def generate(self):
            '''
            '''
            for i in range(self.count):

                sparse, premask, label, parameters = self.next_image()

                sparse = np.asarray(sparse, dtype='uint16')

                image = np.asarray(premask.copy(), dtype='float32')
                image[image>0] = 1 # re-set to binary

                image = np.asarray(image, dtype='float32')

                # adding 5% noise
                image += np.random.uniform(0, 0.05, image.shape)


                # there is the scale object to account for in position_non_aligned_scale and position_common_scale
                extra = 0
                if (self.p_dataset.data_class == 'position_non_aligned_scale' 
                    or self.p_dataset.data_class == 'position_common_scale'):
                    extra = 1

                mask = np.full((premask.shape[0], premask.shape[1], len(label) + extra), False, dtype='bool')


                for i in range(len(label) + extra):
                    filtered_mask = premask.copy()
                    filtered_mask[filtered_mask!=(i+1)] = False
                    filtered_mask[filtered_mask==(i+1)] = True
                    mask[:,:, i] = filtered_mask

                self.label.append(label)
                self.image.append(image)
                self.mask.append(mask)
                self.sparse.append(sparse)

                bbox = np.asarray(utils.extract_bboxes(mask), dtype='uint16')

                self.bbox.append(bbox)

                if self.parameters is None:
                    self.parameters = parameters



        def next_image(self):
            '''
            '''
            is_valid_label = False

            sparse, premask, label, parameters, validation_label = None, None, None, None, None
            
            while not is_valid_label:

                sparse, premask, prelabel, parameters = self.p_dataset.random_image()

                # so we can use the nice numpy operations
                label = np.asarray(prelabel, dtype=self.label_type)

                validation_label = self.p_dataset.modify_label_for_validation(label)

                self.p_dataset.add_itteration()

                is_valid_label = self.validate_label(validation_label)


            self.add_label(validation_label)

            return sparse, premask, label, parameters


        def validate_label(self, label):
            '''
            '''
            
            if not self.p_dataset.check_label_euclid(label):
                self.p_dataset.euclid_failure()
                return False

            if not self.check_distribution(label):
                self.p_dataset.distribution_failure()
                return False

            return True



        def check_distribution(self, label):
            '''
            '''

            # we dont care until we reach larger amounts
            if sum(self.label_distribution.values()) < 1000:
                return True
            
            # not adding anything over 110% of the mean amount in each angle bucket
            threshold = mean(self.label_distribution.values()) * 1.1

            for element in label:

                key = str(element)

                if self.label_distribution.get(key, 0) > threshold:
                    return False
            
            return True



        def add_label(self, label):
            '''
            '''

            for element in label:

                key = str(element)

                self.label_distribution[key] = 1 + self.label_distribution.get(key, 0)

            self.p_dataset.add_label(label)



#####################################################################################
# DatasetGenerator
#####################################################################################

    def __init__(self,
        counts             = {"train": 500, "val": 50, "test": 50},
        flags              = [True,False,False],
        naive              = False,
        batch              = True,
        data_class         = 'angle',
        verbose            = True):

        self.counts             = counts
        self.flags              = flags
        self.naive              = naive
        self.batch              = batch
        self.data_class         = data_class
        self.verbose            = verbose

        self.folder = "output/" + self.data_class + "/"

        self.labels              = []
        self.euclid_table        = {}
        self.itterations         = 0
        self.failed_euclid       = 0
        self.failed_distribution = 0

        for function in DatasetGenerator.data_class_list:
            setattr(self, function, getattr(Figure5, function))

        self.distance_threshold = self.__calculate_distance_threshold()


    def __calculate_distance_threshold(self):

        _, _, label, parameters =  self.random_image()

        threshold = 0

        labels_in_threshold = 1

        total_count = sum(self.counts.values())

        next_threshold = threshold

        if parameters < total_count:
            print(self.data_class, "has parameters", parameters, "which is too small for dataset of size", total_count)

        while threshold < DatasetGenerator.max_distance_threshold and (parameters / labels_in_threshold > total_count):

            threshold = next_threshold
            next_threshold += 1

            labels_in_threshold = len(DatasetGenerator.get_labels_within_threshold(
                np.asarray([next_threshold * 2] * len(label)),
                next_threshold))

        return threshold




    def random_image(self):
        '''
        '''
        image_function = getattr(self, self.data_class)
        
        return image_function(flags=self.flags)


    def generate(self):
        '''
        '''
        startTime = datetime.now()

        print("Generating dataset of class: ", self.data_class)

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

        metadata = dict( map(
            lambda component : ( component, str(getattr(dataset, component)) ),
            DatasetGenerator.dataset_metadata
        ))

        save_data['metadata'] = np.asarray(list(metadata.items()), dtype="str")

        file = self.folder + name + "_" + str(dataset_number) + ".npz"

        np.savez(file, **save_data)

        pickle.dump({self.data_class : dataset.label_distribution}, 
            open(self.folder + name + "_label_distribution_" + str(dataset_number) + ".p", "wb"))

        del dataset


    def modify_label_for_validation(self, label):
        '''
        '''
        return np.asarray([self.modify_element_for_validation(element) for element in label])


    def modify_element_for_validation(self, element):
        # curvature is small floats so we need to make them larger and round
        if self.data_class == 'curvature':
            return int(ceil(element * 1000))
        # area is large floats so we can just round
        elif self.data_class == 'area':
            return int(ceil(element))
        else:
            return element



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

        if self.verbose and not len(self.labels) % 1000:
            print("Labels Accepted: ", len(self.labels))
        
        if not self.naive:
            self.add_labels_within_threshold(label)



    def add_labels_within_threshold(self, label):
        '''
        Adds all possibly labels within the euclidian distance threshold
        to the memoized euclidian distance lookup table.
        '''
        labels = DatasetGenerator.get_labels_within_threshold(label, self.distance_threshold)

        for label in labels:
            self.add_euclid_label(label)


    @staticmethod
    def get_labels_within_threshold(label, distance_threshold):
        return DatasetGenerator.__get_labels_within_threshold(label, label, 0, 0, distance_threshold)



    @staticmethod
    def __get_labels_within_threshold(base_label, current_label, index, old_dist, distance_threshold):

        labels_in_threshold = []

        for op in [add, sub]:

            dist = old_dist

            next_label = current_label.copy();

            while dist <= distance_threshold:

                labels_in_threshold.append(next_label)

                if index + 1 < len(base_label):

                    labels_found = DatasetGenerator.__get_labels_within_threshold(
                        base_label,
                        next_label.copy(),
                        index + 1, dist,
                        distance_threshold
                    )

                    for label in labels_found:
                        labels_in_threshold.append(label)

                next_label[index] = op(next_label[index], 1)

                dist = np.linalg.norm(base_label - next_label)

        return labels_in_threshold



    def add_euclid_label(self, label):
        '''
        Adds a label to the euclidian distance lookup table.
        '''
        self.euclid_table["-".join(label.astype(str))] = True



    def add_itteration(self):
        self.itterations += 1
        if self.verbose and not self.itterations % 1000:
            print("Generation Loop Itteration: ", self.itterations)

    def euclid_failure(self): 
        self.failed_euclid += 1
        if self.verbose and not self.failed_euclid % 1000:
            print("Labels Rejected by Euclidian Distance: ", self.failed_euclid)

    def distribution_failure(self): 
        self.failed_distribution += 1
        if self.verbose and not self.failed_distribution % 1000:
            print("Labels Rejected by Uneven Label Distribution: ", self.failed_distribution)



#####################################################################################
# DatasetFromFile
#####################################################################################

class DatasetFromFile(utils.Dataset):

    def __init__(self, file):
        
        super().__init__()

        self.file = file
        self.metadata = None



    def load_from_file(self):
        '''
        '''
        data = np.load(self.file)

        # pulling all data out of npz file
        loaded_data = dict(map(lambda component: ( component, data[component] ), data.keys()))

        self.metadata = dict(loaded_data.pop('metadata'))

        SETNAME = 'stimuli'
        
        self.add_class(SETNAME, 1, self.metadata['data_class'])

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
