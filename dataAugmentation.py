import torch
import torchio as tio
import numpy as np
import itertools
import random

class MRIDataAugmentation():
    def __init__(self, imgShape=(169, 208, 179), augProb = 0.5, smallBlockFactor=6):
        self.augProb = augProb
        small_block_height = imgShape[0] // smallBlockFactor
        small_block_width = imgShape[1] // smallBlockFactor
        small_block_depth = imgShape[2] // smallBlockFactor

        self.indices_block_small = [((small_block_height*i, small_block_height*(i+1)),
                                     (small_block_width*j, small_block_width*(j+1)),
                                     (small_block_depth*k, small_block_depth*(k+1)))
                                    for i, j, k in itertools.product(range(smallBlockFactor), range(smallBlockFactor), range(smallBlockFactor))]
    
    """torchio API for augmentation"""
    def augmentData_single(self, image):
        if np.random.random() < self.augProb:
            image = tio.ScalarImage(tensor=image)
            translation = tio.RandomAffine(
                scales=(0.9, 1.1),  # scale images with a factor between 0.9 and 1.1
                degrees=(10),  # rotate images by -10 to 10 degrees
                translation=(-10, 10),  # translate images by -10 to 10 pixels
                isotropic=False,  # if True, the scaling factor will be the same for all dimensions
                default_pad_value='mean',  # use the mean intensity of the image for padding
            )
            transformed_image = translation(image).tensor
            return transformed_image
        
    """Dropblocks with gradients"""
    def dropblock_grad_guided(self, imgs, grads, drop_rate):
        imgs = imgs.detach() # torch.Size([1, 169, 208, 179])
        for i in range(imgs.shape[0]):
            imgs[i, :, :, :, :] = self.dropblock_grad_single(imgs[i, :, :, :, :], grads[i, :, :, :, :], drop_rate)
        return imgs

    def dropblock_grad_single(self, img, grad, drop_rate):
        # mean value of gradients within one block
        num_drop_blocks = int(drop_rate * 6**3)
        block_means = list(torch.mean(torch.abs(grad[indices_set[0][0]:indices_set[0][1], indices_set[1][0]:indices_set[1][1],
                                          indices_set[2][0]:indices_set[2][1]])) for indices_set in self.indices_block_small)

        # get the remaining block indices 5%
        indices = [idx for idx in range(len(block_means))]

        # drop the blocks with the topk avg gradients
        largest_grad_indxs = torch.topk(torch.tensor([block_means[idx] for idx in indices]), num_drop_blocks)[1]
        indices_idx = torch.tensor([indices[idx] for idx in largest_grad_indxs])

        drop_indices = indices_idx.int()
        block_indices = [self.indices_block_small[k] for k in drop_indices]

        for block_idx in block_indices:
            img[0, block_idx[0][0]:block_idx[0][1], block_idx[1][0]:block_idx[1][1],
            block_idx[2][0]:block_idx[2][1]] = \
                torch.rand(size=(block_idx[0][1] - block_idx[0][0], block_idx[1][1] - block_idx[1][0], block_idx[2][1] - block_idx[2][0]))

        return img
    
    """Dropblocks randomly"""
    def dropblock_random(self, imgs, drop_rate):
        imgs = imgs.detach() # torch.Size([1, 169, 208, 179])
        for i in range(imgs.shape[0]):
            imgs[i, :, :, :, :] = self.dropblock_random_single(imgs[i, :, :, :, :], drop_rate)
        return imgs

    def dropblock_random_single(self, img, drop_rate):
        num_drop_blocks = int(drop_rate * 6**3)
        indices = [idx for idx in range(len(self.indices_block_small))]
        sampled_indices = random.sample(indices, num_drop_blocks)
        indices_idx = torch.tensor([indices[idx] for idx in sampled_indices])

        drop_indices = indices_idx.int()
        block_indices = [self.indices_block_small[k] for k in drop_indices]
        
        for block_idx in block_indices:
            img[0, block_idx[0][0]:block_idx[0][1], block_idx[1][0]:block_idx[1][1],
            block_idx[2][0]:block_idx[2][1]] = \
                torch.rand(size=(block_idx[0][1] - block_idx[0][0], block_idx[1][1] - block_idx[1][0], block_idx[2][1] - block_idx[2][0]))
        # print("y-------------------------------------------------y")
        return img
