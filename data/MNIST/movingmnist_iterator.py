import logging
import cv2
import numpy as np
import math
import os
from scipy.misc import *


logger = logging.getLogger(__name__)


def load_mnist(training_num=10000):
    """Load the mnist dataset

    Parameters
    ----------
    training_num

    Returns
    -------

    """
    data_path = os.path.join("/home/ices/PycharmProject/RST_LSTM_/data/MNIST/", "mnist.npz")
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
    dat = np.load(data_path)
    X = dat['X'][:training_num]
    Y = dat['Y'][:training_num]
    X_test = dat['X_test']
    Y_test = dat['Y_test']
    Y = Y.reshape((Y.shape[0],))
    Y_test = Y_test.reshape((Y_test.shape[0],))
    return X, Y, X_test, Y_test


def move_step(v0, p0, bounding_box):
    xmin, xmax, ymin, ymax = bounding_box
    assert (p0[0] >= xmin) and (p0[0] <= xmax) and (p0[1] >= ymin) and (p0[1] <= ymax)
    v = v0.copy()
    assert v[0] != 0.0 and v[1] != 0.0
    p = v0 + p0
    while (p[0] < xmin) or (p[0] > xmax) or (p[1] < ymin) or (p[1] > ymax):
        vx, vy = v
        x, y = p
        dist = np.zeros((4,))
        dist[0] = abs(x - xmin) if ymin <= (xmin - x) * vy / vx + y <= ymax else np.inf
        dist[1] = abs(x - xmax) if ymin <= (xmax - x) * vy / vx + y <= ymax else np.inf
        dist[2] = abs((y - ymin) * vx / vy) if xmin <= (ymin - y) * vx / vy + x <= xmax else np.inf
        dist[3] = abs((y - ymax) * vx / vy) if xmin <= (ymax - y) * vx / vy + x <= xmax else np.inf
        n = np.argmin(dist)
        if n == 0:
            v[0] = -v[0]
            p[0] = 2 * xmin - p[0]
        elif n == 1:
            v[0] = -v[0]
            p[0] = 2 * xmax - p[0]
        elif n == 2:
            v[1] = -v[1]
            p[1] = 2 * ymin - p[1]
        elif n == 3:
            v[1] = -v[1]
            p[1] = 2 * ymax - p[1]
        else:
            assert False
    return v, p


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


class MovingMNISTAdvancedIterator(object):

    def __init__(self,
                 data_num = 8000,
                 n_frames_input=5,
                 n_frames_output=10,
                 digit_num=3,
                 distractor_num=None,
                 img_size=64,
                 distractor_size=5,
                 max_velocity_scale=3.6,
                 initial_velocity_range=(0.0, 3.6),
                 acceleration_range=(0.0, 0.0),
                 scale_variation_range=(1 / 1.1, 1.1),
                 rotation_angle_range=(-30, 30),
                 global_rotation_angle_range=(-30, 30),
                 illumination_factor_range=(0.6, 1.0),
                 global_rotation_prob=0.5):
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
        self.data_num = data_num
        self.mnist_train_img, self.mnist_train_label,\
        self.mnist_test_img, self.mnist_test_label = load_mnist(data_num)
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self._digit_num = digit_num
        self._img_size = img_size
        self._distractor_size = distractor_size
        self._distractor_num = 0
        self._max_velocity_scale = max_velocity_scale
        self._initial_velocity_range = initial_velocity_range
        self._acceleration_range = acceleration_range
        self._scale_variation_range = scale_variation_range
        self._rotation_angle_range = rotation_angle_range
        self._illumination_factor_range = illumination_factor_range
        self._global_rotation_angle_range = global_rotation_angle_range
        self._global_rotation_prob = global_rotation_prob
        self._index_range = (0,data_num)
        self._h5py_f = None
        self._seq = None
        self._motion_vectors = None
        self.replay = None
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
            distractor_img = self.mnist_train_img[ind].reshape((28, 28))
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

    def sample(self, batch_size, random=True):
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
        seqlen = self.n_frames_input+self.n_frames_output
        if self.replay is not None:
            if random is True:
                self.replay_index = np.random.randint(self.replay_numsamples - batch_size)
            elif self.replay_index + batch_size > self.replay_numsamples:
                raise IndexError("Not enough pre-generated parameters to create new sample.")

        seq = np.zeros(
            (seqlen, batch_size, 1, self._img_size, self._img_size),
            dtype=np.float32)
        motion_vectors = np.zeros(
            (seqlen, batch_size, 2, self._img_size, self._img_size),
            dtype=np.float32)
        inner_boundary = np.array(
            [10, 10, self._img_size - 10, self._img_size - 10],
            dtype=np.float32)
        for b in range(batch_size):
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

            if self.replay is not None:
                digit_indices = self.replay["digit_indices"][self.replay_index
                                                             + b]
                appearance_mult = self.replay["appearance_mult"][
                    self.replay_index + b]
                scale_variation = self.replay["scale_variation"][
                    self.replay_index + b]
                base_rotation_angle = self.replay["base_rotation_angle"][
                    self.replay_index + b]
                affine_transforms_multipliers = self.replay[
                    "affine_transforms_multipliers"][self.replay_index + b]
                init_velocity_angle = self.replay["init_velocity_angle"][
                    self.replay_index + b]
                init_velocity_magnitude = self.replay[
                    "init_velocity_magnitude"][self.replay_index + b]
                distractor_seeds = self.replay[
                    "distractor_seeds"][self.replay_index + b]

                assert(distractor_seeds.shape[0] == seqlen)

            else:
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
                crop_mnist_digit(self.mnist_train_img[i].reshape((28, 28)))
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
                seq[i, b, 0, :, :] = self.draw_imgs(
                    base_img=[
                        base_digit_img[j] * appearance_variants[i, j]
                        for j in range(self._digit_num)
                    ],
                    affine_transforms=affine_transforms[i])
                self.draw_distractors(seq[i, b, 0, :, :], distractor_seeds[i])

        self.replay_index += batch_size

        return seq, motion_vectors

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

    train_parameter = {
        'max_velocity_scale':3.6,
        'initial_velocity_range':[0.0,3.6],
        'scale_variation_range':[0.9,1.1],
        'rotation_angle_range':[-30,30],
        'global_rotation_angle_range':[-20,20],
        'illumination_factor_range':[0.6,1.0],
        'max_range':[100,200]
    }
    test_set1_parameter = {
        'max_velocity_scale':4.6,
        'initial_velocity_range':[0.0,4.6],
        'scale_variation_range':[0.8,1.2],
        'rotation_angle_range':[-45,45],
        'global_rotation_angle_range':[-45,45],
        'illumination_factor_range':[0.8,1.2],
        'max_range':[80,220]
    }
    test_set2_parameter = {
        'max_velocity_scale': 5.6,
        'initial_velocity_range': [2.0, 5.6],
        'scale_variation_range': [0.7, 1.3],
        'rotation_angle_range': [-45, 45],
        'global_rotation_angle_range': [-30, 30],
        'illumination_factor_range': [0.7, 1.3],
        'max_range': [80, 220]
    }
    test_set3_parameter = {
        'max_velocity_scale': 5.6,
        'initial_velocity_range': [0.0, 5.6],
        'scale_variation_range': [0.6, 1.4],
        'rotation_angle_range': [-45, 45],
        'global_rotation_angle_range': [-45, 45],
        'illumination_factor_range': [0.8, 1.2],
        'max_range': [80, 220]
    }

    test_set4_parameter = {
        'max_velocity_scale': 4.6,
        'initial_velocity_range': [2.0, 4.6],
        'scale_variation_range': [0.5, 1.5],
        'rotation_angle_range': [-40, 40],
        'global_rotation_angle_range': [-30, 30],
        'illumination_factor_range': [0.8, 1.2],
        'max_range': [80, 220]
    }


    n_input = 10,
    n_output = 10,
    train_num = 8000
    validation_num = 2000
    test_num = 4000

    testset1_iter = MovingMNISTAdvancedIterator(
        data_num = test_num,
        n_frames_input=10,
        n_frames_output=10,
        max_velocity_scale=test_set1_parameter['max_velocity_scale'],
        initial_velocity_range=test_set1_parameter['initial_velocity_range'],
        scale_variation_range=test_set1_parameter['scale_variation_range'],
        rotation_angle_range=test_set1_parameter['rotation_angle_range'],
        global_rotation_angle_range=test_set1_parameter['global_rotation_angle_range'],
        illumination_factor_range=test_set1_parameter['illumination_factor_range'],
        global_rotation_prob=0.5
    )

    testset2_iter = MovingMNISTAdvancedIterator(
        data_num=test_num,
        n_frames_input=10,
        n_frames_output=10,
        max_velocity_scale=test_set2_parameter['max_velocity_scale'],
        initial_velocity_range=test_set2_parameter['initial_velocity_range'],
        scale_variation_range=test_set2_parameter['scale_variation_range'],
        rotation_angle_range=test_set2_parameter['rotation_angle_range'],
        global_rotation_angle_range=test_set2_parameter['global_rotation_angle_range'],
        illumination_factor_range=test_set2_parameter['illumination_factor_range'],
        global_rotation_prob=0.5
    )

    testset3_iter = MovingMNISTAdvancedIterator(
        data_num=test_num,
        n_frames_input=10,
        n_frames_output=10,
        max_velocity_scale=test_set3_parameter['max_velocity_scale'],
        initial_velocity_range=test_set3_parameter['initial_velocity_range'],
        scale_variation_range=test_set3_parameter['scale_variation_range'],
        rotation_angle_range=test_set3_parameter['rotation_angle_range'],
        global_rotation_angle_range=test_set3_parameter['global_rotation_angle_range'],
        illumination_factor_range=test_set3_parameter['illumination_factor_range'],
        global_rotation_prob=0.5
    )

    testset4_iter = MovingMNISTAdvancedIterator(
        data_num=test_num,
        n_frames_input=10,
        n_frames_output=10,
        max_velocity_scale=test_set4_parameter['max_velocity_scale'],
        initial_velocity_range=test_set4_parameter['initial_velocity_range'],
        scale_variation_range=test_set4_parameter['scale_variation_range'],
        rotation_angle_range=test_set4_parameter['rotation_angle_range'],
        global_rotation_angle_range=test_set4_parameter['global_rotation_angle_range'],
        illumination_factor_range=test_set4_parameter['illumination_factor_range'],
        global_rotation_prob=0.5
    )

    moving_iter = MovingMNISTAdvancedIterator(
        data_num = train_num,
        n_frames_input=10,
        n_frames_output=10,
        max_velocity_scale=train_parameter['max_velocity_scale'],
        initial_velocity_range=train_parameter['initial_velocity_range'],
        scale_variation_range=train_parameter['scale_variation_range'],
        rotation_angle_range=train_parameter['rotation_angle_range'],
        global_rotation_angle_range=train_parameter['global_rotation_angle_range'],
        illumination_factor_range=train_parameter['illumination_factor_range'],
        global_rotation_prob=0.5
    )
    validation_moving_iter = MovingMNISTAdvancedIterator(
        data_num=validation_num,
        n_frames_input=10,
        n_frames_output=10,
        max_velocity_scale=train_parameter['max_velocity_scale'],
        initial_velocity_range=train_parameter['initial_velocity_range'],
        scale_variation_range=train_parameter['scale_variation_range'],
        rotation_angle_range=train_parameter['rotation_angle_range'],
        global_rotation_angle_range=train_parameter['global_rotation_angle_range'],
        illumination_factor_range=train_parameter['illumination_factor_range'],
        global_rotation_prob=0.5
    )

    mnist_data_root_path = '/mnt/A/MNIST_dataset/'
    if os.path.exists(mnist_data_root_path):
        print('root folder exist')
    else:
        os.mkdir(mnist_data_root_path)

    train_root_path = mnist_data_root_path+'train/'
    if os.path.exists(train_root_path):
        print('train dataset fold exist')
    else:
        os.mkdir(train_root_path)
        for num in range(train_num):
            train_sample_path = train_root_path+'sample_'+str(num+1)+'/'
            seq_imgs, _ = moving_iter.sample(batch_size=1, random=False)
            seq = seq_imgs[:,0,0,:,:,]
            os.mkdir(train_sample_path)
            for img_index in range(20):
                img_path = train_sample_path+'img_'+str(img_index+1)+'.png'
                img = seq[img_index]
                imsave(img_path,img)
            print('training dataset generate complete ',(float)(100*(1+num)/train_num),'%')

    validation_root_path = mnist_data_root_path+'validation/'
    if os.path.exists(validation_root_path):
        print('validation dataset fold exist')
    else:
        os.mkdir(validation_root_path)
        for num in range(validation_num):
            validation_sample_path = validation_root_path+'sample_'+str(num+1)+'/'
            seq_imgs, _ = validation_moving_iter.sample(batch_size=1,random=False)
            seq = seq_imgs[:,0,0,:,:,]
            os.mkdir(validation_sample_path)
            for img_index in range(20):
                img_path = validation_sample_path+'img_'+str(img_index+1)+'.png'
                img = seq[img_index]
                imsave(img_path,img)
            print('validation dataset generate complete',(float)(100*(1+num)/validation_num),'%')

    test1_root_path = mnist_data_root_path + 'test_set1/'
    if os.path.exists(test1_root_path):
        print('test dataset 1 fold exist')
    else:
        os.mkdir(test1_root_path)
        for num in range(test_num):
            test_sample_path = test1_root_path + 'sample_' + str(num + 1) + '/'
            seq_imgs, _ = testset1_iter.sample(batch_size=1, random=False)
            seq = seq_imgs[:, 0, 0, :, :, ]
            os.mkdir(test_sample_path)
            for img_index in range(20):
                img_path = test_sample_path + 'img_' + str(img_index + 1) + '.png'
                img = seq[img_index]
                imsave(img_path, img)
            print('test dataset 1 generate complete', (float)(100 * (1 + num) / test_num), '%')

    test2_root_path = mnist_data_root_path + 'test_set2/'
    if os.path.exists(test2_root_path):
        print('test dataset 2 fold exist')
    else:
        os.mkdir(test2_root_path)
        for num in range(test_num):
            test_sample_path = test2_root_path + 'sample_' + str(num + 1) + '/'
            seq_imgs, _ = testset2_iter.sample(batch_size=1, random=False)
            seq = seq_imgs[:, 0, 0, :, :, ]
            os.mkdir(test_sample_path)
            for img_index in range(20):
                img_path = test_sample_path + 'img_' + str(img_index + 1) + '.png'
                img = seq[img_index]
                imsave(img_path, img)
            print('test dataset 2 generate complete', (float)(100 * (1 + num) / test_num), '%')

    test3_root_path = mnist_data_root_path + 'test_set3/'
    if os.path.exists(test3_root_path):
        print('test dataset 3 fold exist')
    else:
        os.mkdir(test3_root_path)
        for num in range(test_num):
            test_sample_path = test3_root_path + 'sample_' + str(num + 1) + '/'
            seq_imgs, _ = testset3_iter.sample(batch_size=1, random=False)
            seq = seq_imgs[:, 0, 0, :, :, ]
            os.mkdir(test_sample_path)
            for img_index in range(20):
                img_path = test_sample_path + 'img_' + str(img_index + 1) + '.png'
                img = seq[img_index]
                imsave(img_path, img)
            print('test dataset 3 generate complete', (float)(100 * (1 + num) / test_num), '%')

    test4_root_path = mnist_data_root_path + 'test_set4/'
    if os.path.exists(test4_root_path):
        print('test dataset 4 fold exist')
    else:
        os.mkdir(test4_root_path)
        for num in range(test_num):
            test_sample_path = test4_root_path + 'sample_' + str(num + 1) + '/'
            seq_imgs, _ = testset4_iter.sample(batch_size=1, random=False)
            seq = seq_imgs[:, 0, 0, :, :, ]
            os.mkdir(test_sample_path)
            for img_index in range(20):
                img_path = test_sample_path + 'img_' + str(img_index + 1) + '.png'
                img = seq[img_index]
                imsave(img_path, img)
            print('test dataset 4 generate complete', (float)(100 * (1 + num) / test_num), '%')
    # import argparse
    #
    # parser = argparse.ArgumentParser(
    #     description='Generate sample from MovingMNISTAdvancedIterator gifs.')
    #
    # parser.set_defaults(mode='test')
    # parser.add_argument(
    #     '--no-distractors',
    #     action='store_true',
    #     help="Don't load/generate/use parameters for distractors.")
    #
    # subparsers = parser.add_subparsers(help='Specify saving or loading mode.')
    # s = subparsers.add_parser('save', help='Generate a new dataset.')
    # s.add_argument(
    #     'sequences', type=int, help="Number of sequences to generate.")
    # s.add_argument('length', type=int, help="Length of each sequence.")
    # s.add_argument(
    #     'path', nargs='?', type=int, help="Path to the params file.")
    # s.set_defaults(mode='save')
    #
    # l = subparsers.add_parser('load', help='Load an existing dataset.')
    # l.add_argument(
    #     'path', nargs='?', type=int, help="Path to the params file.")
    # l.set_defaults(mode='load')
    #
    # args = parser.parse_args()
    #
    # distractor_num = 0 if args.no_distractors else 6
    # mnist_generator = MovingMNISTAdvancedIterator(
    #     distractor_num=distractor_num)
    #
    # batch_size = 1
    # if args.mode == 'test':
    #     seqlen = 100
    # elif args.mode == 'save':
    #     if args.path:
    #         fname = args.path
    #     else:
    #         fname = "params.npz"
    #
    #     print("Generating {} sequences of length {}. Saving to {}.".format(
    #         args.sequences, args.length, fname))
    #     seqlen = args.length
    #     mnist_generator.save(
    #         seqlen=seqlen, num_samples=args.sequences, file=fname)
    # elif args.mode == 'load':
    #     if args.path:
    #         fname = args.path
    #     else:
    #         fname = "params.npz"
    #     num_sequences, seqlen = mnist_generator.load(file=fname)
    #     print("Loaded {} sequences of length {}. Saving to {}.".format(
    #         num_sequences, seqlen, fname))
    #
    # seq, _ = mnist_generator.sample(batch_size=batch_size, seqlen=seqlen)
    #
    # print(seq.sum())
    #
    # save_gif(seq[:, 0, 0, :, :].astype(np.float32) / 255.0, "test.gif")
