import os, sys

# we need access to the MaskR-CNN code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/mask_rcnn/'))
from mrcnn import utils
from mrcnn import visualize

# we need access to Ian's code
sys.path.append(os.path.join(os.path.dirname(__file__), 'external/ian/'))
import figure5

import numpy as np
import json

# train_target  = 80000
# val_target    = 20000
# test_target   = 20000

# MAX_ANGLE  = 90
# HEIGHT     = 100
# WIDTH      = 150
# NUM_ANGLES = 4

# folder = "output/"

# all_counter   = 0

# train_data = np.zeros((train_target, HEIGHT, WIDTH), dtype=np.float32)
# train_label = np.zeros((train_target, NUM_ANGLES), dtype=np.float32)
# train_counter = 0

# val_data = np.zeros((val_target, HEIGHT, WIDTH), dtype=np.float32)
# val_label = np.zeros((val_target, NUM_ANGLES), dtype=np.float32)
# val_counter = 0

# test_data = np.zeros((test_target, HEIGHT, WIDTH), dtype=np.float32)
# test_label = np.zeros((test_target, NUM_ANGLES), dtype=np.float32)
# test_counter = 0

# train_angles = np.zeros(MAX_ANGLE)
# val_angles   = np.zeros(MAX_ANGLE)
# test_angles  = np.zeros(MAX_ANGLE)

# used = set()

# iteration_counter = 0

# while train_counter < train_target or val_counter < val_target or test_counter < test_target:
    
#     # sanity checking when running
#     iteration_counter += 1
#     if not iteration_counter % 100000:
#         print(iteration_counter)

#     sparse, image, label, parameters = Figure5.angle(flags=[True, False, False])

#     np_image = image.astype(np.float32)
    
#     # adding noise
#     np_image += np.random.uniform(0, 0.05, (HEIGHT, WIDTH))
    
#     sorted_label = tuple(np.sort(label))
    
#     if sorted_label not in used:
                                                                               
#         if train_counter < train_target and check_distribution(label, train_angles, train_counter):
            
#             for angle in label: 
#                 train_angles[angle - 1] += 1

#             train_data[train_counter] = np_image
#             train_label[train_counter] = label
            
#             pylab.imsave(folder + "train/" + str(train_counter) + ".png", image)
            
#             used.add(sorted_label)
#             train_counter += 1
#             all_counter += 1

#         #repeat process with other 2 sets of data
#         elif val_counter < val_target and check_distribution(label, val_angles, val_counter):
            
#             for angle in label: 
#                 val_angles[angle - 1] += 1

#             val_data[val_counter] = np_image
#             val_label[val_counter] = label
            
#             pylab.imsave(folder + "val/" + str(val_counter) + ".png", image)
            
#             used.add(sorted_label)
#             val_counter += 1
#             all_counter += 1

#         elif test_counter < test_target and check_distribution(label, test_angles, test_counter):
            
#             for angle in label: 
#                 test_angles[angle - 1] += 1

#             test_data[test_counter] = np_image
#             test_label[test_counter] = label
            
#             pylab.imsave(folder + "test/" + str(test_counter) + ".png", image)
            
#             used.add(sorted_label)
#             test_counter += 1
#             all_counter += 1
        
# np.save(folder + "train_data.npy",  train_data)
# np.save(folder + "train_label.npy", train_label)
# np.save(folder + "val_data.npy",    val_data)
# np.save(folder + "val_label.npy",   val_label)
# np.save(folder + "test_data.npy",   test_data)
# np.save(folder + "test_label.npy",  test_label)



class PartitionedDataset:

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
        def show(which, howmany=4):

            image_ids = np.random.choice(which.image_ids, howmany)
            for image_id in image_ids:
                image = which.load_image(image_id)
                mask, class_ids = which.load_mask(image_id)
                visualize.display_top_masks(image, mask, class_ids, which.class_names)


    def __init__(self, datasets={"train": 500, "val": 50, "test": 50}, flags=[True,False,False]):

        self.datasets = {}

        for key in datasets:
            self.datasets[key] = self.AngleDataset()
            self.datasets[key].generate(datasets[key], flags)
            self.datasets[key].prepare()


    def check_distribution(label, ang_counter, total_counter):
        
        # we dont care until we reach larger amounts
        if total_counter < 1000:
            return True
        
        # not adding anything over 110% of the average amount in each angle bucket
        threshold = (total_counter / len(ang_counter) * NUM_ANGLES) * 1.1
        
        for angle in label:
            if ang_counter[angle - 1] > threshold:
                return False
        
        return True

