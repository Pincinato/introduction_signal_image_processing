import numpy as np
import matplotlib.pyplot as plt
import statistics
import functions as fn

debug = 1
figure_number = 0


def update_figure_number():
    global figure_number
    figure_number += 1
    return figure_number


def ex0(a, b, c):
    ack = -1
    if b >= a:
        x_ = np.arange(a, b, c)
        plt.figure(update_figure_number())
        plt.title('Ex 1')
        plt.plot(x_, fn.func(x_))
        ack = 0
    return ack


def visualize_results(distances):
    mean = statistics.mean(distances)
    std = statistics.stdev(distances)
    median = statistics.median(distances)
    title = "Mean: " + str(mean) + " Standard variation: " + str(std) + " Median: " + str(median)
    plt.figure(update_figure_number())
    plt.hist(distances, bins=np.arange(100))
    plt.title(title)
    return


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
    mask_rgb[fn.find_indexes_threshold((image[:, :, :]), tmin[:], tmax[:])] = 1
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
    img = fn.create_image(100, 100)
    if debug == 1:
        plt.figure(update_figure_number())
        plt.title("Random white/black image  with one red pixel")
        plt.imshow(img)
    dist = fn.compute_distances(img)
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
        new_image = fn.apply_mask(loaded_image, my_mask)
        plt.figure(update_figure_number())
        plt.title("stopturnsigns.jpg after application of mask")
        plt.imshow(new_image)
        info += "-" + str(figure_number)
    print(info)
    return 0


def ex4_solver():
    # fourth exercise
    print("Ex 4")
    if fn.check_mse() == 0:
        info = "Images " + str(figure_number + 1)
        x = np.arange(1, 201, 1)
        coefficient_a = 1.2
        f_x = 0.1*fn.func(x)
        g_x = coefficient_a*x
        plt.figure(update_figure_number())
        plot_graph_2_curves(x, f_x, "f(x)", x, g_x, "g(x)", "Ex 4 f(x) and g(x)")
        g_x = fn.find_better_linear_approximation(x, f_x)
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
