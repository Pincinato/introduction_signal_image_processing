import numpy as np

def func(x_):
    y = x_ + np.sqrt(x_) + 1/np.power(x_, 2) + 10*np.sin(x_)
    return y

def set_red_pixel(image, line, column):
    ack = image
    if (image.shape[0] > line) and (image.shape[1] > column):
        image[line, column, 0] = 1
        image[line, column, 1] = 0
        image[line, column, 2] = 0
    return ack

def create_image(n, m):
    ack = np.zeros((n, m, 3))
    if (m > 0) and (n > 0):
        image = np.zeros((n, m, 3))
        image[:, :, 0] = np.random.randint(0, 2, size=(m, n))
        image[:, :, 1] = image[:, :, 0]
        image[:, :, 2] = image[:, :, 0]
        ack = set_red_pixel(image, np.random.randint(0, n, size=1), np.random.randint(0, m, size=1))
    return ack




def find_pixels(pixel_values, img):
    indexes_red_channel = img[:, :, 0] == pixel_values[0]
    indexes_green_channel = img[:, :, 1] == pixel_values[1]
    indexes_blue_channel = img[:, :, 2] == pixel_values[2]
    ack = indexes_red_channel & indexes_green_channel & indexes_blue_channel
    ack = np.where(ack == True)
    return ack


def euclidean_distance(a, b):
    return np.sqrt(np.sum(np.power(a - b, 2), axis=0))


def compute_distances(image):
    index_red = find_pixels(np.array([1, 0, 0]), image)
    indexes_white = find_pixels(np.array([1, 1, 1]), image)
    indexes_red = np.tile(index_red, (1, indexes_white[0].size))
    distances = euclidean_distance(indexes_red, np.array([indexes_white[0], indexes_white[1]]))
    return distances


def find_indexes_threshold(channel, tmin, tmax):
    index_min = channel >= tmin
    index_max = channel <= tmax
    ack = index_min & index_max
    return np.where(ack == True)


def apply_mask(image, mask):
    image_created = np.zeros(image.shape)
    indexes = np.where(mask == 1)
    image_created[indexes[0], indexes[1], :] = image[indexes[0], indexes[1], :]/255
    return image_created


def mse(a, b):
    ack = -1
    if (a.shape == b.shape) and (a.ndim == 1):
        ack = np.sum(np.power(a-b, 2))/a.shape
    return ack


def check_mse():
    ack = - 1
    test_array = np.random.randint(100, size=100)
    test_array_offset_2 = test_array + 2
    control = 0
    if mse(test_array, test_array) != 0:
        print("Error in computing mse")
        control += 1
    if mse(test_array, test_array_offset_2) != 4:
        print("Error in computing mse with offset 2")
        control += 1
    if control == 0:
        print("Mse function passed all test")
        ack = 0
    return ack


def find_better_coefficient(a_, x_, f_):
    mse_of_coefficients = np.zeros(a_.shape[0])
    for i in range(0, a_.shape[0]):
        mse_of_coefficients[i] = mse(a_[i]*x_, f_)
    ack = a_[np.argmin(mse_of_coefficients)]
    return ack


def find_better_linear_approximation(x_,  f_):
    a_ = np.arange(0.01, 0.5, 0.01)
    coefficient_a_ = find_better_coefficient(a_, x_, f_)
    return coefficient_a_*x_
