import numpy as np
import matplotlib.pyplot as plt
import statistics
# import functions


debug = 1
figure_number = 0


def update_figure_number():
    global figure_number
    figure_number += 1
    return figure_number


def func(x_):
    y = x_ + np.sqrt(x_) + 1/np.power(x_, 2) + 10*np.sin(x_)
    return y


def ex0(a, b, c):
    ack = -1
    if b >= a:
        x_ = np.arange(a, b, c)
        plt.figure(update_figure_number())
        plt.title('Ex 1')
        plt.plot(x_, func(x_))
        ack = 0
    return ack


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


def visualize_results(distances):
    mean = statistics.mean(distances)
    std = statistics.stdev(distances)
    median = statistics.median(distances)
    title = "Mean: " + str(mean) + " Standard variation: " + str(std) + " Median: " + str(median)
    plt.figure(update_figure_number())
    plt.hist(distances, bins=np.arange(100))
    plt.title(title)
    return


def find_indexes_threshold(channel, tmin, tmax):
    index_min = channel >= tmin
    index_max = channel <= tmax
    ack = index_min & index_max
    return np.where(ack == True)


def subplot_graph(subplot_option, image_plot, axis, title):
    plt.subplot(subplot_option[0], subplot_option[1], subplot_option[2])
    plt.imshow(image_plot)
    plt.axis(axis)  # this line removes the axis numbering
    plt.title(title)
    return


def plot_graph_2_curves(my_x1, my_y1, my_label1, my_x2, my_y2, my_label2, my_title):

    plt.plot(my_x1, my_y1, label=my_label1)
    plt.plot(my_x2, my_y2, label=my_label2)
    plt.xlabel('x axis label')
    plt.ylabel('y axis label')
    plt.title(my_title)
    plt.legend(loc=0)
    return


def apply_threshold(image, tmin, tmax):

    mask_rgb = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    mask_rgb[find_indexes_threshold((image[:, :, :]), tmin[:], tmax[:])] = 1
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[np.where((mask_rgb[:, :, 0] == 1) & (mask_rgb[:, :, 1] == 1) & (mask_rgb[:, :, 2] == 1))] = 1.0
    # graph and test
    if debug == 1:
        plt.figure(update_figure_number())
        subplot_graph([2, 3, 1], image[:, :, 0], 'off', 'Red')
        subplot_graph([2, 3, 2], image[:, :, 1], 'off', 'Green')
        subplot_graph([2, 3, 3], image[:, :, 2], 'off', 'Blue')
        subplot_graph([2, 3, 4], mask_rgb[:, :, 0], 'off', 'Red after threshold ')
        subplot_graph([2, 3, 5], mask_rgb[:, :, 1], 'off', 'Green after threshold ')
        subplot_graph([2, 3, 6], mask_rgb[:, :, 2], 'off', 'Blue after threshold ')
    return mask


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


def ex1_solver():
    # first exercise
    print("Ex 1")
    info = "Images " + str(figure_number + 1)
    ret = ex0(1, 10, 1)
    if ret == -1:
        print("Invalid parameters (ex0)")
    info += "-" + str(figure_number)
    print(info)
    return 0


def ex2_solver():
    # second exercise
    print("Ex 2 ")
    info = "Images " + str(figure_number + 1)
    img = create_image(100, 100)
    if debug == 1:
        plt.figure(update_figure_number())
        plt.title("Random white/black image  with one red pixel")
        plt.imshow(img)
    dist = compute_distances(img)
    visualize_results(dist)
    info += "-" + str(figure_number)
    print(info)
    return 0


def ex3_solver():
    # third exercise
    print("Ex 3")
    info = "Images " + str(figure_number + 1)
    loaded_image = plt.imread("stopturnsigns.jpg")
    if debug == 1:
        plt.figure(update_figure_number())
        plt.title("stopturnsigns.jpg")
        plt.imshow(loaded_image)
        my_mask = apply_threshold(loaded_image, [225, 30, 50], [253, 70, 85])
        new_image = apply_mask(loaded_image, my_mask)
        plt.figure(update_figure_number())
        plt.title("stopturnsigns.jpg after application of mask")
        plt.imshow(new_image)
        info += "-" + str(figure_number)
    print(info)
    return 0


def ex4_solver():
    # fourth exercise
    print("Ex 4")
    if check_mse() == 0:
        info = "Images " + str(figure_number + 1)
        x = np.arange(1, 201, 1)
        coefficient_a = 1.2
        f_x = 0.1*func(x)
        g_x = coefficient_a*x
        plt.figure(update_figure_number())
        plot_graph_2_curves(x, f_x, "f(x)", x, g_x, "g(x)", "Ex 4 f(x) and g(x)")
        g_x = find_better_linear_approximation(x, f_x)
        plt.figure(update_figure_number())
        plot_graph_2_curves(x, f_x, "f(x)", x, g_x, "g(x)", "Ex 4 f(x) and g(x) after coefficent's computation")
        info += "-" + str(figure_number)
        print(info)
    return 0


ex1_solver()
ex2_solver()
ex3_solver()
ex4_solver()
plt.show()
