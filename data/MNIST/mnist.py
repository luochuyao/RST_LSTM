import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import cv2
import torch
import torch.utils.data as data

def load_moving_mnist(training_num=10000):
    """Load the mnist dataset

    Parameters
    ----------
    training_num

    Returns
    -------

    """
    data_path = os.path.join("/home/ices/PycharmProject/FST_ConvRNN/data/MNIST/", "mnist.npz")
    if not os.path.isfile(data_path):
        origin = (
            'https://github.com/sxjscience/mxnet/raw/master/example/bayesian-methods/mnist.npz'
        )
        print('Downloading data from %s to %s' % (origin, data_path))

        from urllib import request
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context  # Not verify
        data_file = request.urlopen(origin)
        with open(data_path, 'wb') as output:
            output.write(data_file.read())
        print('Done!')

    shuffled_indices = np.random.permutation(training_num)
    dat = np.load(data_path)
    X = dat['X'][:training_num]
    Y = dat['Y'][:training_num]

    X_test = dat['X_test'][:4000]
    Y_test = dat['Y_test'][:4000]

    X_validation = X[shuffled_indices[:int(training_num / 5)]]
    Y_validation = Y[shuffled_indices[:int(training_num / 5)]]

    X_train = X[shuffled_indices[int(training_num / 5):]]
    Y_train = Y[shuffled_indices[int(training_num / 5):]]

    Y_train = Y_train.reshape((Y_train.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    Y_validation = Y_validation.reshape((Y_validation.shape[0],))

    return X_train,Y_train,X_validation,Y_validation, X_test,Y_test

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root):
    # Load the fixed dataset
    filename = 'mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train, n_frames_input, n_frames_output, num_objects,
                 transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):

        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        # if self.transform is not None:
        #     images = self.transform(images)
        images = images.transpose(0,3,1,2)
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        frozen = input[-1]
        # add a wall to input data
        # pad = np.zeros_like(input[:, 0])
        # pad[:, 0] = 1
        # pad[:, pad.shape[1] - 1] = 1
        # pad[:, :, 0] = 1
        # pad[:, :, pad.shape[2] - 1] = 1
        #
        # input = np.concatenate((input, np.expand_dims(pad, 1)), 1)

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        # print()
        # print(input.size())
        # print(output.size())

        out = [idx, output, input, frozen]
        return out

    def __len__(self):
        return self.length



def crop_mnist_digit(digit_img, tol=5):
    """Return the cropped version of the mnist digit

    Parameters
    ----------
    digit_img : np.ndarray
        Shape: ()

    Returns
    -------

    """
    tol = float(tol) / float(255)
    mask = digit_img > tol
    return digit_img[np.ix_(mask.any(1), mask.any(0))]

# class MovingMNISTAdvancedIterator(object):
#
#     def __init__(self,
#                  digit_num=3,
#                  distractor_num=0,
#                  img_size=64,
#                  distractor_size=5,
#                  max_velocity_scale=3.6,
#                  initial_velocity_range=(0.0, 3.6),
#                  acceleration_range=(0.0, 0.0),
#                  scale_variation_range=(1 / 1.1, 1.1),
#                  rotation_angle_range=(-30, 30),
#                  global_rotation_angle_range=(-30, 30),
#                  illumination_factor_range=(0.6, 1.0),
#                  period=5,
#                  global_rotation_prob=0.5,
#                  index_range=(0, 40000)):
#         """
#
#         Parameters
#         ----------
#         digit_num : int
#             Number of digits
#         distractor_num : int
#             Number of distractors
#         img_size : int
#             Size of the image
#         distractor_size : int
#             Size of the distractors
#         max_velocity_scale : float
#             Maximum scale of the velocity
#         initial_velocity_range : tuple
#         acceleration_range
#         scale_variation_range
#         rotation_angle_range
#         period : period of the
#         index_range
#         """
#         self.mnist_train_img, self.mnist_train_label,\
#         self.mnist_test_img, self.mnist_test_label = load_mnist()
#         self._digit_num = digit_num
#         self._img_size = img_size
#         self._distractor_size = distractor_size
#         self._distractor_num = distractor_num
#         self._max_velocity_scale = max_velocity_scale
#         self._initial_velocity_range = initial_velocity_range
#         self._acceleration_range = acceleration_range
#         self._scale_variation_range = scale_variation_range
#         self._rotation_angle_range = rotation_angle_range
#         self._illumination_factor_range = illumination_factor_range
#         self._period = period
#         self._global_rotation_angle_range = global_rotation_angle_range
#         self._global_rotation_prob = global_rotation_prob
#         self._index_range = index_range
#         self._h5py_f = None
#         self._seq = None
#         self._motion_vectors = None
#         self.replay = None
#         self.replay_index = 0
#         self.replay_numsamples = -1
#
#     def _choose_distractors(self, distractor_seeds):
#         """Choose the distractors
#
#         We use the similar approach as
#          https://github.com/deepmind/mnist-cluttered/blob/master/mnist_cluttered.lua
#         Returns
#         -------
#         ret : list
#             list of distractor images
#         """
#         ret = []
#         for i in range(self._distractor_num):
#             ind = math.floor(distractor_seeds[i, 2] * self._index_range[1])
#             distractor_img = self.mnist_train_img[ind].reshape((28, 28))
#             distractor_h_begin = math.floor(distractor_seeds[i, 3] * (28 - self._distractor_size))
#             distractor_w_begin = math.floor(distractor_seeds[i, 4] * (28 - self._distractor_size))
#             distractor_img = distractor_img[
#                 distractor_h_begin:distractor_h_begin + self._distractor_size,
#                 distractor_w_begin:distractor_w_begin + self._distractor_size]
#             ret.append(distractor_img)
#         return ret
#
#     def draw_distractors(self, canvas_img, distractor_seeds):
#         """
#
#         Parameters
#         ----------
#         canvas_img
#
#         Returns
#         -------
#
#         """
#         distractor_imgs = self._choose_distractors(distractor_seeds)
#         for i, img in enumerate(distractor_imgs):
#             r_begin = math.floor(distractor_seeds[i][0] * (self._img_size - img.shape[0]))
#             c_begin = math.floor(distractor_seeds[i][1] * (self._img_size - img.shape[1]))
#             canvas_img[r_begin:r_begin + img.shape[0], c_begin:c_begin +
#                        img.shape[1]] = img
#         return canvas_img
#
#     def draw_imgs(self,
#                   base_img,
#                   affine_transforms,
#                   prev_affine_transforms=None):
#         """
#
#         Parameters
#         ----------
#         base_img : list
#             Inner Shape: (H, W)
#         affine_transforms : np.ndarray
#             Shape: (digit_num, 2, 3)
#         prev_affine_transforms : np.ndarray
#             Shape: (digit_num, 2, 3)
#
#         Returns
#         -------
#
#         """
#         canvas_img = np.zeros(
#             (self._img_size, self._img_size), dtype=np.float32)
#         for i in range(self._digit_num):
#             tmp_img = cv2.warpAffine(base_img[i], affine_transforms[i],
#                                      (self._img_size, self._img_size))
#             canvas_img = np.maximum(canvas_img, tmp_img)
#         return canvas_img
#
#     def _find_center(self, img):
#         x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
#         raise NotImplementedError
#
#     def _bounce_border(self, inner_boundary, affine_transform, digit_shift,
#                        velocity, img_h, img_w):
#         # top-left, top-right, down-left, down-right
#         center = affine_transform.dot(
#             np.array([img_w / 2.0, img_h / 2.0, 1], dtype=np.float32))
#         new_velocity = velocity.copy()
#         new_center = center.copy()
#         if center[0] < inner_boundary[0]:
#             new_velocity[0] = -new_velocity[0]
#             new_center[0] = inner_boundary[0]
#         if center[0] > inner_boundary[2]:
#             new_velocity[0] = -new_velocity[0]
#             new_center[0] = inner_boundary[2]
#         if center[1] < inner_boundary[1]:
#             new_velocity[1] = -new_velocity[1]
#             new_center[1] = inner_boundary[1]
#         if center[1] > inner_boundary[3]:
#             new_velocity[1] = -new_velocity[1]
#             new_center[1] = inner_boundary[3]
#         affine_transform[:, 2] += new_center - center
#         digit_shift += new_center - center
#         return affine_transform, digit_shift, new_velocity
#
#     def sample(self, batch_size, seqlen, random=True):
#         """
#
#         Parameters
#         ----------
#         batch_size : int
#         seqlen : int
#         random: take random samples from loaded parameters. Ignored if no parameters are loaded.
#
#         Returns
#         -------
#         seq : np.ndarray
#             Shape: (seqlen, batch_size, 1, H, W)
#         motion_vectors : np.ndarray
#             Shape: (seqlen, batch_size, 2, H, W)
#         """
#
#         if self.replay is not None:
#             if random is True:
#                 self.replay_index = np.random.randint(self.replay_numsamples - batch_size)
#             elif self.replay_index + batch_size > self.replay_numsamples:
#                 raise IndexError("Not enough pre-generated parameters to create new sample.")
#
#         seq = np.zeros(
#             (seqlen, batch_size, 1, self._img_size, self._img_size),
#             dtype=np.float32)
#         motion_vectors = np.zeros(
#             (seqlen, batch_size, 2, self._img_size, self._img_size),
#             dtype=np.float32)
#         inner_boundary = np.array(
#             [10, 10, self._img_size - 10, self._img_size - 10],
#             dtype=np.float32)
#         for b in range(batch_size):
#             affine_transforms = np.zeros(
#                 (seqlen, self._digit_num, 2, 3), dtype=np.float32)
#             appearance_variants = np.ones(
#                 (seqlen, self._digit_num), dtype=np.float32)
#             scale = np.ones((seqlen, self._digit_num), dtype=np.float32)
#             rotation_angle = np.zeros(
#                 (seqlen, self._digit_num), dtype=np.float32)
#             init_velocity = np.zeros(
#                 shape=(self._digit_num, 2), dtype=np.float32)
#             velocity = np.zeros((seqlen, self._digit_num, 2), dtype=np.float32)
#             digit_shift = np.zeros(
#                 (seqlen, self._digit_num, 2), dtype=np.float32)
#
#             if self.replay is not None:
#                 digit_indices = self.replay["digit_indices"][self.replay_index
#                                                              + b]
#                 appearance_mult = self.replay["appearance_mult"][
#                     self.replay_index + b]
#                 scale_variation = self.replay["scale_variation"][
#                     self.replay_index + b]
#                 base_rotation_angle = self.replay["base_rotation_angle"][
#                     self.replay_index + b]
#                 affine_transforms_multipliers = self.replay[
#                     "affine_transforms_multipliers"][self.replay_index + b]
#                 init_velocity_angle = self.replay["init_velocity_angle"][
#                     self.replay_index + b]
#                 init_velocity_magnitude = self.replay[
#                     "init_velocity_magnitude"][self.replay_index + b]
#                 distractor_seeds = self.replay[
#                     "distractor_seeds"][self.replay_index + b]
#
#                 assert(distractor_seeds.shape[0] == seqlen)
#
#             else:
#                 digit_indices = np.random.randint(
#                     low=self._index_range[0],
#                     high=self._index_range[1],
#                     size=self._digit_num)
#                 appearance_mult = np.random.uniform(
#                     low=self._illumination_factor_range[0],
#                     high=self._illumination_factor_range[1])
#                 scale_variation = np.random.uniform(
#                     low=self._scale_variation_range[0],
#                     high=self._scale_variation_range[1],
#                     size=(self._digit_num, ))
#                 base_rotation_angle = np.random.uniform(
#                     low=self._rotation_angle_range[0],
#                     high=self._rotation_angle_range[1],
#                     size=(self._digit_num, ))
#                 affine_transforms_multipliers = np.random.uniform(
#                     size=(self._digit_num, 2))
#                 init_velocity_angle = np.random.uniform(size=(
#                     self._digit_num, )) * (2 * np.pi)
#                 init_velocity_magnitude = np.random.uniform(
#                     low=self._initial_velocity_range[0],
#                     high=self._initial_velocity_range[1],
#                     size=self._digit_num)
#                 distractor_seeds = np.random.uniform(
#                     size=(seqlen, self._distractor_num, 5))
#
#             base_digit_img = [
#                 crop_mnist_digit(self.mnist_train_img[i].reshape((28, 28)))
#                 for i in digit_indices
#             ]
#
#             for i in range(1, seqlen):
#                 appearance_variants[i, :] = appearance_variants[i - 1, :] *\
#                                             (appearance_mult ** -(2 * ((i // 5) % 2) - 1))
#
#             for i in range(1, seqlen):
#                 base_factor = (2 * ((i // 5) % 2) - 1)
#                 scale[i, :] = scale[i - 1, :] * (scale_variation**base_factor)
#                 rotation_angle[i, :] = rotation_angle[
#                     i - 1, :] + base_rotation_angle
#
#             affine_transforms[0, :, 0, 0] = 1.0
#             affine_transforms[0, :, 1, 1] = 1.0
#             for i in range(self._digit_num):
#                 affine_transforms[0, i, 0, 2] = affine_transforms_multipliers[i, 0] *\
#                     (self._img_size - base_digit_img[i].shape[1])
#                 affine_transforms[0, i, 1, 2] = affine_transforms_multipliers[i, 1] *\
#                     (self._img_size - base_digit_img[i].shape[0])
#
#             init_velocity[:, 0] = init_velocity_magnitude * np.cos(
#                 init_velocity_angle)
#             init_velocity[:, 1] = init_velocity_magnitude * np.sin(
#                 init_velocity_angle)
#             curr_velocity = init_velocity
#
#             # base_acceleration_angle = np.random.random() * 2 * np.pi
#             # base_acceleration_magnitude = np.random.uniform(low=self._acceleration_range[0],
#             #                                                 high=self._acceleration_range[1],
#             #                                                 size=self._digit_num)
#             # base_acceleration = np.zeros(shape=(self._digit_num, 2), dtype=np.float32)
#             # base_acceleration[:, 0] = base_acceleration_magnitude * np.cos(init_velocity_angle)
#             # base_acceleration[:, 1] = base_acceleration_magnitude * np.sin(init_velocity_angle)
#
#             for i in range(self._digit_num):
#                 digit_shift[0, i, 0] = affine_transforms[
#                     0, i, 0, 2]  #+ (base_digit_img[i].shape[1] / 2.0)
#                 digit_shift[0, i, 1] = affine_transforms[
#                     0, i, 1, 2]  #+ (base_digit_img[i].shape[0] / 2.0)
#
#             for i in range(seqlen - 1):
#                 velocity[i, :, :] = curr_velocity
#                 #curr_velocity += base_acceleration * (2 * ((i / 5) % 2) - 1)
#                 curr_velocity = np.clip(
#                     curr_velocity,
#                     a_min=-self._max_velocity_scale,
#                     a_max=self._max_velocity_scale)
#                 for j in range(self._digit_num):
#                     digit_shift[i + 1, j, :] = digit_shift[
#                         i, j, :] + curr_velocity[j]
#                     rotation_mat = cv2.getRotationMatrix2D(
#                         center=(base_digit_img[j].shape[1] / 2.0,
#                                 base_digit_img[j].shape[0] / 2.0),
#                         angle=rotation_angle[i + 1, j],
#                         scale=scale[i + 1, j])
#                     affine_transforms[i + 1, j, :, :2] = rotation_mat[:, :2]
#                     affine_transforms[i + 1, j, :, 2] = digit_shift[
#                         i + 1, j, :] + rotation_mat[:, 2]
#                     affine_transforms[i + 1, j, :, :], digit_shift[i + 1, j, :], curr_velocity[j] =\
#                         self._bounce_border(inner_boundary=inner_boundary,
#                                             affine_transform=affine_transforms[i + 1, j, :, :],
#                                             digit_shift=digit_shift[i + 1, j, :],
#                                             velocity=curr_velocity[j],
#                                             img_h=base_digit_img[j].shape[0],
#                                             img_w=base_digit_img[j].shape[1])
#             for i in range(seqlen):
#                 seq[i, b, 0, :, :] = self.draw_imgs(
#                     base_img=[
#                         base_digit_img[j] * appearance_variants[i, j]
#                         for j in range(self._digit_num)
#                     ],
#                     affine_transforms=affine_transforms[i])
#                 self.draw_distractors(seq[i, b, 0, :, :], distractor_seeds[i])
#
#         self.replay_index += batch_size
#
#         return seq, motion_vectors
#
#     def load(self, file):
#         """Initialize to draw samples from pre-computed parameters.
#
#         Args:
#             file: Either the file name (string) or an open file (file-like
#                 object) from which the data will be loaded.
#         """
#         self.replay_index = 0
#         with np.load(file) as f:
#             self.replay = dict(f)
#
#         assert(self.replay["distractor_seeds"].shape[2] == self._distractor_num)
#
#         num_samples, seqlen = self.replay["distractor_seeds"].shape[0:2]
#         self.replay_numsamples = num_samples
#         return num_samples, seqlen
#
#     def save(self, seqlen, num_samples=10000, file=None):
#         """Draw random numbers for num_samples sequences and save them.
#
#         This initializes the state of MovingMNISTAdvancedIterator to generate
#         sequences based on the hereby drawn parameters.
#
#         Note that each call to sample(batch_size, seqlen) will use batch_size
#         of the num_samples parameters.
#
#         Args:
#             num_samples: Number of unique MovingMNISTAdvanced sequences to draw
#                 parameters for
#             file: Either the file name (string) or an open file (file-like
#                 object) where the data will be saved. If file is a string or a
#                 Path, the .npz extension will be appended to the file name if
#                 it is not already there.
#
#         """
#         if file is None:
#             file = "mnist_{}".format(num_samples)
#
#         self.replay = dict()
#         self.replay["digit_indices"] = np.random.randint(
#             low=self._index_range[0],
#             high=self._index_range[1],
#             size=(num_samples, self._digit_num))
#         self.replay["appearance_mult"] = np.random.uniform(
#             low=self._illumination_factor_range[0],
#             high=self._illumination_factor_range[1],
#             size=(num_samples, ))
#         self.replay["scale_variation"] = np.random.uniform(
#             low=self._scale_variation_range[0],
#             high=self._scale_variation_range[1],
#             size=(num_samples, self._digit_num))
#         self.replay["base_rotation_angle"] = np.random.uniform(
#             low=self._rotation_angle_range[0],
#             high=self._rotation_angle_range[1],
#             size=(num_samples, self._digit_num))
#         self.replay["affine_transforms_multipliers"] = np.random.uniform(
#             size=(num_samples, self._digit_num, 2))
#         self.replay["init_velocity_angle"] = np.random.uniform(
#             size=(num_samples, self._digit_num)) * 2 * np.pi
#         self.replay["init_velocity_magnitude"] = np.random.uniform(
#             low=self._initial_velocity_range[0],
#             high=self._initial_velocity_range[1],
#             size=(num_samples, self._digit_num))
#         self.replay["distractor_seeds"] = np.random.uniform(
#             size=(num_samples, seqlen, self._distractor_num, 5))
#
#         self.replay_numsamples = num_samples
#
#         np.savez_compressed(file=file, **self.replay)



class MovingMNISTAdvancedIterator(data.Dataset):
    def __init__(self,
                 root,
                 n_frames_input,
                 n_frames_output,
                 dataset,
                 data_source,
                 digit_num=3,
                 distractor_num=0,
                 img_size=64,
                 distractor_size=5,
                 max_velocity_scale=3.6,
                 initial_velocity_range=(0.0, 3.6),
                 acceleration_range=(0.0, 0.0),
                 scale_variation_range=(1 / 1.1, 1.1),
                 rotation_angle_range=(-30, 30),
                 global_rotation_angle_range=(-30, 30),
                 illumination_factor_range=(0.6, 1.0),
                 period=5,
                 global_rotation_prob=0.5,
                 ):
        """

        Parameters
        ----------
        digit_num : int
            Number of digits
        distractor_num : int
            Number of distractors
        img_size : int
            Size of the image
        distractor_size : int
            Size of the distractors
        max_velocity_scale : float
            Maximum scale of the velocity
        initial_velocity_range : tuple
        acceleration_range
        scale_variation_range
        rotation_angle_range
        period : period of the
        index_range
        """
        self.root = root
        self.mnist_train_img, self.mnist_train_label,\
        self.mnist_validation_img,self.mnist_validation_label,\
        self.mnist_test_img, self.mnist_test_label, = data_source
        if dataset == 'train':
            self.dataset = self.mnist_train_img
            self._index_range = (0,8000)
        elif dataset == 'validation':
            self.dataset = self.mnist_validation_img
            self._index_range = (0, 2000)
        elif dataset == 'test':
            self.dataset = self.mnist_test_img
            self._index_range = (0, 4000)
        else:
            raise ('dataset error')
        self._length = self.dataset.shape[0]
        self._digit_num = digit_num
        self._img_size = img_size
        self._distractor_size = distractor_size
        self._distractor_num = distractor_num
        self._max_velocity_scale = max_velocity_scale
        self._initial_velocity_range = initial_velocity_range
        self._acceleration_range = acceleration_range
        self._scale_variation_range = scale_variation_range
        self._rotation_angle_range = rotation_angle_range
        self._illumination_factor_range = illumination_factor_range
        self._period = period
        self._global_rotation_angle_range = global_rotation_angle_range
        self._global_rotation_prob = global_rotation_prob

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self._h5py_f = None
        self._seq = None
        self._motion_vectors = None
        # self.replay = None
        self.replay_index = 0
        self.replay_numsamples = -1

    def _choose_distractors(self, distractor_seeds):
        """Choose the distractors

        We use the similar approach as
         https://github.com/deepmind/mnist-cluttered/blob/master/mnist_cluttered.lua
        Returns
        -------
        ret : list
            list of distractor images
        """
        ret = []
        for i in range(self._distractor_num):
            ind = math.floor(distractor_seeds[i, 2] * self._index_range[1])
            distractor_img = self.dataset[ind].reshape((28, 28))
            distractor_h_begin = math.floor(distractor_seeds[i, 3] * (28 - self._distractor_size))
            distractor_w_begin = math.floor(distractor_seeds[i, 4] * (28 - self._distractor_size))
            distractor_img = distractor_img[
                distractor_h_begin:distractor_h_begin + self._distractor_size,
                distractor_w_begin:distractor_w_begin + self._distractor_size]
            ret.append(distractor_img)
        return ret

    def draw_distractors(self, canvas_img, distractor_seeds):
        """

        Parameters
        ----------
        canvas_img

        Returns
        -------

        """
        distractor_imgs = self._choose_distractors(distractor_seeds)
        for i, img in enumerate(distractor_imgs):
            r_begin = math.floor(distractor_seeds[i][0] * (self._img_size - img.shape[0]))
            c_begin = math.floor(distractor_seeds[i][1] * (self._img_size - img.shape[1]))
            canvas_img[r_begin:r_begin + img.shape[0], c_begin:c_begin +
                       img.shape[1]] = img
        return canvas_img

    def draw_imgs(self,
                  base_img,
                  affine_transforms,
                  prev_affine_transforms=None):
        """

        Parameters
        ----------
        base_img : list
            Inner Shape: (H, W)
        affine_transforms : np.ndarray
            Shape: (digit_num, 2, 3)
        prev_affine_transforms : np.ndarray
            Shape: (digit_num, 2, 3)

        Returns
        -------

        """
        canvas_img = np.zeros(
            (self._img_size, self._img_size), dtype=np.float32)
        for i in range(self._digit_num):
            tmp_img = cv2.warpAffine(base_img[i], affine_transforms[i],
                                     (self._img_size, self._img_size))
            canvas_img = np.maximum(canvas_img, tmp_img)
        return canvas_img

    def _find_center(self, img):
        x, y = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
        raise NotImplementedError

    def _bounce_border(self, inner_boundary, affine_transform, digit_shift,
                       velocity, img_h, img_w):
        # top-left, top-right, down-left, down-right
        center = affine_transform.dot(
            np.array([img_w / 2.0, img_h / 2.0, 1], dtype=np.float32))
        new_velocity = velocity.copy()
        new_center = center.copy()
        if center[0] < inner_boundary[0]:
            new_velocity[0] = -new_velocity[0]
            new_center[0] = inner_boundary[0]
        if center[0] > inner_boundary[2]:
            new_velocity[0] = -new_velocity[0]
            new_center[0] = inner_boundary[2]
        if center[1] < inner_boundary[1]:
            new_velocity[1] = -new_velocity[1]
            new_center[1] = inner_boundary[1]
        if center[1] > inner_boundary[3]:
            new_velocity[1] = -new_velocity[1]
            new_center[1] = inner_boundary[3]
        affine_transform[:, 2] += new_center - center
        digit_shift += new_center - center
        return affine_transform, digit_shift, new_velocity

    def __getitem__(self, idx):

        """

        Parameters
        ----------
        batch_size : int
        seqlen : int
        random: take random samples from loaded parameters. Ignored if no parameters are loaded.

        Returns
        -------
        seq : np.ndarray
            Shape: (seqlen, batch_size, 1, H, W)
        motion_vectors : np.ndarray
            Shape: (seqlen, batch_size, 2, H, W)
        """

        # if self.replay is not None:
        #     if random is True:
        #         self.replay_index = np.random.randint(self.replay_numsamples - batch_size)
        #     elif self.replay_index + batch_size > self.replay_numsamples:
        #         raise IndexError("Not enough pre-generated parameters to create new sample.")
        seqlen = self.n_frames_input+self.n_frames_output
        seq = np.zeros(
            (seqlen, 1, self._img_size, self._img_size),
            dtype=np.float32)
        motion_vectors = np.zeros(
            (seqlen, 2, self._img_size, self._img_size),
            dtype=np.float32)
        inner_boundary = np.array(
            [10, 10, self._img_size - 10, self._img_size - 10],
            dtype=np.float32)
        affine_transforms = np.zeros(
            (seqlen, self._digit_num, 2, 3), dtype=np.float32)
        appearance_variants = np.ones(
            (seqlen, self._digit_num), dtype=np.float32)
        scale = np.ones((seqlen, self._digit_num), dtype=np.float32)
        rotation_angle = np.zeros(
            (seqlen, self._digit_num), dtype=np.float32)
        init_velocity = np.zeros(
            shape=(self._digit_num, 2), dtype=np.float32)
        velocity = np.zeros((seqlen, self._digit_num, 2), dtype=np.float32)
        digit_shift = np.zeros(
            (seqlen, self._digit_num, 2), dtype=np.float32)

        # if self.replay is not None:
        #     digit_indices = self.replay["digit_indices"][self.replay_index
        #                                                  + b]
        #     appearance_mult = self.replay["appearance_mult"][
        #         self.replay_index + b]
        #     scale_variation = self.replay["scale_variation"][
        #         self.replay_index + b]
        #     base_rotation_angle = self.replay["base_rotation_angle"][
        #         self.replay_index + b]
        #     affine_transforms_multipliers = self.replay[
        #         "affine_transforms_multipliers"][self.replay_index + b]
        #     init_velocity_angle = self.replay["init_velocity_angle"][
        #         self.replay_index + b]
        #     init_velocity_magnitude = self.replay[
        #         "init_velocity_magnitude"][self.replay_index + b]
        #     distractor_seeds = self.replay[
        #         "distractor_seeds"][self.replay_index + b]
        #
        #     assert(distractor_seeds.shape[0] == seqlen)
        #
        # else:
        digit_indices = np.random.randint(
            low=self._index_range[0],
            high=self._index_range[1],
            size=self._digit_num)
        appearance_mult = np.random.uniform(
            low=self._illumination_factor_range[0],
            high=self._illumination_factor_range[1])
        scale_variation = np.random.uniform(
            low=self._scale_variation_range[0],
            high=self._scale_variation_range[1],
            size=(self._digit_num, ))
        base_rotation_angle = np.random.uniform(
            low=self._rotation_angle_range[0],
            high=self._rotation_angle_range[1],
            size=(self._digit_num, ))
        affine_transforms_multipliers = np.random.uniform(
            size=(self._digit_num, 2))
        init_velocity_angle = np.random.uniform(size=(
            self._digit_num, )) * (2 * np.pi)
        init_velocity_magnitude = np.random.uniform(
            low=self._initial_velocity_range[0],
            high=self._initial_velocity_range[1],
            size=self._digit_num)
        distractor_seeds = np.random.uniform(
            size=(seqlen, self._distractor_num, 5))

        base_digit_img = [
            crop_mnist_digit(self.dataset[i].reshape((28, 28)))
            for i in digit_indices
        ]

        for i in range(1, seqlen):
            appearance_variants[i, :] = appearance_variants[i - 1, :] *\
                                        (appearance_mult ** -(2 * ((i // 5) % 2) - 1))

        for i in range(1, seqlen):
            base_factor = (2 * ((i // 5) % 2) - 1)
            scale[i, :] = scale[i - 1, :] * (scale_variation**base_factor)
            rotation_angle[i, :] = rotation_angle[
                i - 1, :] + base_rotation_angle

        affine_transforms[0, :, 0, 0] = 1.0
        affine_transforms[0, :, 1, 1] = 1.0
        for i in range(self._digit_num):
            affine_transforms[0, i, 0, 2] = affine_transforms_multipliers[i, 0] *\
                (self._img_size - base_digit_img[i].shape[1])
            affine_transforms[0, i, 1, 2] = affine_transforms_multipliers[i, 1] *\
                (self._img_size - base_digit_img[i].shape[0])

        init_velocity[:, 0] = init_velocity_magnitude * np.cos(
            init_velocity_angle)
        init_velocity[:, 1] = init_velocity_magnitude * np.sin(
            init_velocity_angle)
        curr_velocity = init_velocity

        # base_acceleration_angle = np.random.random() * 2 * np.pi
        # base_acceleration_magnitude = np.random.uniform(low=self._acceleration_range[0],
        #                                                 high=self._acceleration_range[1],
        #                                                 size=self._digit_num)
        # base_acceleration = np.zeros(shape=(self._digit_num, 2), dtype=np.float32)
        # base_acceleration[:, 0] = base_acceleration_magnitude * np.cos(init_velocity_angle)
        # base_acceleration[:, 1] = base_acceleration_magnitude * np.sin(init_velocity_angle)

        for i in range(self._digit_num):
            digit_shift[0, i, 0] = affine_transforms[
                0, i, 0, 2]  #+ (base_digit_img[i].shape[1] / 2.0)
            digit_shift[0, i, 1] = affine_transforms[
                0, i, 1, 2]  #+ (base_digit_img[i].shape[0] / 2.0)

        for i in range(seqlen - 1):
            velocity[i, :, :] = curr_velocity
            #curr_velocity += base_acceleration * (2 * ((i / 5) % 2) - 1)
            curr_velocity = np.clip(
                curr_velocity,
                a_min=-self._max_velocity_scale,
                a_max=self._max_velocity_scale)
            for j in range(self._digit_num):
                digit_shift[i + 1, j, :] = digit_shift[
                    i, j, :] + curr_velocity[j]
                rotation_mat = cv2.getRotationMatrix2D(
                    center=(base_digit_img[j].shape[1] / 2.0,
                            base_digit_img[j].shape[0] / 2.0),
                    angle=rotation_angle[i + 1, j],
                    scale=scale[i + 1, j])
                affine_transforms[i + 1, j, :, :2] = rotation_mat[:, :2]
                affine_transforms[i + 1, j, :, 2] = digit_shift[
                    i + 1, j, :] + rotation_mat[:, 2]
                affine_transforms[i + 1, j, :, :], digit_shift[i + 1, j, :], curr_velocity[j] =\
                    self._bounce_border(inner_boundary=inner_boundary,
                                        affine_transform=affine_transforms[i + 1, j, :, :],
                                        digit_shift=digit_shift[i + 1, j, :],
                                        velocity=curr_velocity[j],
                                        img_h=base_digit_img[j].shape[0],
                                        img_w=base_digit_img[j].shape[1])

        for i in range(seqlen):
            seq[i, 0, :, :] = self.draw_imgs(
                base_img=[
                    base_digit_img[j] * appearance_variants[i, j]
                    for j in range(self._digit_num)
                ],
                affine_transforms=affine_transforms[i])
            self.draw_distractors(seq[i, 0, :, :], distractor_seeds[i])

        input = seq[:self.n_frames_input]
        output = seq[self.n_frames_input:]
        out = [idx, output, input,motion_vectors]
        return out


    def load(self, file):
        """Initialize to draw samples from pre-computed parameters.

        Args:
            file: Either the file name (string) or an open file (file-like
                object) from which the data will be loaded.
        """
        self.replay_index = 0
        with np.load(file) as f:
            self.replay = dict(f)

        assert(self.replay["distractor_seeds"].shape[2] == self._distractor_num)
        num_samples, seqlen = self.replay["distractor_seeds"].shape[0:2]
        self.replay_numsamples = num_samples
        return num_samples, seqlen



    def __len__(self):
        return self._length

    def save(self, seqlen, num_samples=10000, file=None):
        """Draw random numbers for num_samples sequences and save them.

        This initializes the state of MovingMNISTAdvancedIterator to generate
        sequences based on the hereby drawn parameters.

        Note that each call to sample(batch_size, seqlen) will use batch_size
        of the num_samples parameters.

        Args:
            num_samples: Number of unique MovingMNISTAdvanced sequences to draw
                parameters for
            file: Either the file name (string) or an open file (file-like
                object) where the data will be saved. If file is a string or a
                Path, the .npz extension will be appended to the file name if
                it is not already there.

        """
        if file is None:
            file = "mnist_{}".format(num_samples)

        self.replay = dict()
        self.replay["digit_indices"] = np.random.randint(
            low=self._index_range[0],
            high=self._index_range[1],
            size=(num_samples, self._digit_num))
        self.replay["appearance_mult"] = np.random.uniform(
            low=self._illumination_factor_range[0],
            high=self._illumination_factor_range[1],
            size=(num_samples, ))
        self.replay["scale_variation"] = np.random.uniform(
            low=self._scale_variation_range[0],
            high=self._scale_variation_range[1],
            size=(num_samples, self._digit_num))
        self.replay["base_rotation_angle"] = np.random.uniform(
            low=self._rotation_angle_range[0],
            high=self._rotation_angle_range[1],
            size=(num_samples, self._digit_num))
        self.replay["affine_transforms_multipliers"] = np.random.uniform(
            size=(num_samples, self._digit_num, 2))
        self.replay["init_velocity_angle"] = np.random.uniform(
            size=(num_samples, self._digit_num)) * 2 * np.pi
        self.replay["init_velocity_magnitude"] = np.random.uniform(
            low=self._initial_velocity_range[0],
            high=self._initial_velocity_range[1],
            size=(num_samples, self._digit_num))
        self.replay["distractor_seeds"] = np.random.uniform(
            size=(num_samples, seqlen, self._distractor_num, 5))

        self.replay_numsamples = num_samples

        np.savez_compressed(file=file, **self.replay)


if __name__ == '__main__':
    import yaml
    path = '/home/ices/PycharmProject/FST_ConvRNN/experiment/MNIST/config/dec_ConvGRU_MNIST.yml'
    f = open(path)
    info = yaml.safe_load(f)
    print(info)
    datasource = load_moving_mnist()
    trainFolder = MovingMNIST(
        is_train=True,
        root=info['Folder_path'],
        n_frames_input=info['DATA']['INPUT_SEQ_LEN'],
        n_frames_output=info['DATA']['OUTPUT_SEQ_LEN'],
        num_objects=info['DATA']['NUM_DIGITS']
    )
    trainFolder_ = MovingMNISTAdvancedIterator(
        root=info['Folder_path'],
        n_frames_input=info['DATA']['INPUT_SEQ_LEN'],
        n_frames_output=info['DATA']['OUTPUT_SEQ_LEN'],
        dataset='train',
        data_source=datasource
    )
    validationFolder_ = MovingMNISTAdvancedIterator(
        root=info['Folder_path'],
        n_frames_input=info['DATA']['INPUT_SEQ_LEN'],
        n_frames_output=info['DATA']['OUTPUT_SEQ_LEN'],
        dataset='validation',
        data_source=datasource
    )
    testFolder_ = MovingMNISTAdvancedIterator(
        root=info['Folder_path'],
        n_frames_input=info['DATA']['INPUT_SEQ_LEN'],
        n_frames_output=info['DATA']['OUTPUT_SEQ_LEN'],
        dataset='test',
        data_source=datasource
    )
    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=info['TRAIN']['BATCH_SIZE'],
        shuffle=info['TRAIN']['SHUFFLE'],
        num_workers=1
    )
    trainLoader_ = torch.utils.data.DataLoader(
        trainFolder_,
        batch_size=info['TRAIN']['BATCH_SIZE'],
        shuffle=info['TRAIN']['SHUFFLE'],
        num_workers=1
    )
    for i, (idx, img, delta, _) in enumerate(trainLoader):
        if torch.cuda.is_available():
            in_frame_dat = delta.cuda()
            target_frame_dat = img.cuda()

        print(in_frame_dat.size())
        print(target_frame_dat.size())
        print(np.max(in_frame_dat.data.cpu().numpy()))
        print(np.min(in_frame_dat.data.cpu().numpy()))
        break

    # for i, (idx, img, delta, _) in enumerate(trainLoader_):
    #     if torch.cuda.is_available():
    #         in_frame_dat = delta.cuda()
    #         target_frame_dat = img.cuda()
    #         in_frame_dat = 2 * (in_frame_dat/255.0 - 0.5)
    #         target_frame_dat = 2 * (target_frame_dat / 255.0 - 0.5)
    #     print(in_frame_dat.size())
    #     print(target_frame_dat.size())
    #     print(np.max(in_frame_dat.data.cpu().numpy()))
    #     print(np.min(in_frame_dat.data.cpu().numpy()))
    #     break

