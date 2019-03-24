import os
import numpy as np
import matplotlib.pyplot as plt


def test_interp():
    # Tests the interp() function with a known input and output
    # Leads to error if test fails

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
    x_new = np.array((0.5, 2.3, 3, 5.45))
    y_new_solution = np.array([0.2, 0.46, 0.6, 0.69])
    y_new_result = interp(y, x, x_new)
    np.testing.assert_almost_equal(y_new_solution, y_new_result)


def test_interp_1D():
    # Test the interp_1D() function with a known input and output
    # Leads to error if test fails

    y = np.array([0.2, 0.4, 0.6, 0.4, 0.6, 0.8, 1.0, 1.1])
    y_rescaled_solution = np.array([
        0.20000000000000001, 0.29333333333333333, 0.38666666666666671,
        0.47999999999999998, 0.57333333333333336, 0.53333333333333333,
        0.44000000000000006, 0.45333333333333331, 0.54666666666666663,
        0.64000000000000001, 0.73333333333333339, 0.82666666666666677,
        0.91999999999999993, 1.0066666666666666, 1.0533333333333335,
        1.1000000000000001
    ])
    y_rescaled_result = interp_1D(y, 2)
    np.testing.assert_almost_equal(y_rescaled_solution, y_rescaled_result)

def test_interp_2D():
    # Tests interp_2D() function with a known and unknown output
    # Leads to error if test fails

    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    matrix_scaled = np.array([[1., 1.4, 1.8, 2.2, 2.6, 3.],
                              [2., 2.4, 2.8, 3.2, 3.6, 4.],
                              [3., 3.4, 3.8, 4.2, 4.6, 5.],
                              [4., 4.4, 4.8, 5.2, 5.6, 6.]])

    result = interp_2D(matrix, 2)
    np.testing.assert_almost_equal(matrix_scaled, result)


def interp(y_vals, x_vals, x_new):
    # Computes interpolation at the given abscissas
    #
    # Inputs:
    #   x_vals: Given inputs abscissas, numpy array
    #   y_vals: Given input ordinates, numpy array
    #   x_new : New abscissas to find the respective interpolated ordinates, numpy
    #   arrays
    #
    # Outputs:
    #   y_new: Interpolated values, numpy array

    ################### PLEASE FILL IN THIS PART ###############################
    y_new = np.zeros(x_new.shape[0])
    x0 = np.concatenate(([0], x_vals) , axis=0);
    x1 = np.roll(x0,-1)
    y0 = np.concatenate(([0], y_vals) , axis=0);
    y1 = np.roll(y0,-1)
    for i in range(0, x_new.shape[0]):
        if x_new[i] >= np.max(x0):
            y_new[i] = y0[y0.shape[0] - 1]
        else:
            index = np.min(np.where(x0 > x_new[i])) -1
            if index == 0:
                y_new[i] = y0[1]
            else:
                y_new[i] = y1[index]*((x_new[i] - x0[index])/(x1[index] - x0[index]))
                y_new[i] = y_new[i] + y0[index]*(1 - (x_new[i] - x0[index])/(x1[index] - x0[index]))
    return y_new


def interp_1D(signal, scale_factor):
    # Linearly interpolates one dimensional signal by a given saling fcator
    #
    # Inputs:
    #   signal: A one dimensional signal to be samples from, numpy array
    #   scale_factor: scaling factor, float
    #
    # Outputs:
    #   signal_interp: Interpolated 1D signal, numpy array

    ################### PLEASE FILL IN THIS PART ###############################
    x = np.arange(0, signal.shape[0], 1)
    xnew = np.linspace(0,signal.shape[0]-1, signal.shape[0]*scale_factor)
    signal_interp = interp(signal, x, xnew)
    return signal_interp

def interp_2D(img, scale_factor):
    # Applies bilinear interpolation using 1D linear interpolation
    # It first interpolates in one dimension and passes to the next dimension
    #
    # Inputs:
    #   img: 2D signal/image (grayscale or RGB), numpy array
    #   scale_factor: Scaling factor, float
    #
    # Outputs:
    #   img_interp: interpolated image with the expected output shape, numpy array

    ################### PLEASE FILL IN THIS PART ###############################
    img_interp = np.zeros(( round(img.shape[0] * scale_factor), round(img.shape[1] * scale_factor)))
    for i in range(0, img.shape[0]):
        img_interp[i, :] = interp_1D(img[i, :], scale_factor)
    for i in range(0, img_interp.shape[1]):
        img_interp[:, i] = interp_1D(img_interp[np.arange(0, img.shape[0]), i], scale_factor)
    return img_interp


# set arguments
filename = 'bird.jpg'
#filename = 'butterfly.jpg'
#filename = 'monkey_face.jpg'
scale_factor = 2  # Scaling factor

# Before trying to directly test the bilinear interpolation on an image, we
# test the intermediate functions to see if the functions that are coded run
# correctly and give the expected results.

print('...................................................')
print('Testing test_interp()...')
test_interp()
print('done.')

print('Testing interp_1D()....')
test_interp_1D()
print('done.')

print('Testing interp_2D()....')
test_interp_2D()
print('done.')


print('Testing bilinear interpolation of an image...')


# Read image as a matrix, get image shapes before and after interpolation
img = (plt.imread(filename)).astype('float')  # need to convert to float
in_shape = img.shape  # Input image shape
# Apply bilinear interpolation
is_gray = not(np.min(in_shape) == 3)
if is_gray == True:
    img_int = interp_2D(img, scale_factor)
else:
    R = interp_2D(img[:, :, 0], scale_factor)
    G = interp_2D(img[:, :, 1], scale_factor)
    B = interp_2D(img[:, :, 2], scale_factor)
    img_int = np.zeros((R.shape[0], R.shape[1], 3))
    img_int[:, :, 0] = R
    img_int[:, :, 1] = G
    img_int[:, :, 2] = B
print('done.')
# Now, we save the interpolated image and show the results
print('Plotting and saving results...')
plt.figure()
if is_gray == True:
    plt.imshow(img_int.astype('uint8'), cmap="gray")  # Get back to uint8 data type
else:
    plt.imshow(img_int.astype('uint8'))
filename, _ = os.path.splitext(filename)
plt.savefig('{}_rescaled.jpg'.format(filename))
plt.close()
plt.figure()
plt.subplot(1, 2, 1)

if is_gray == True:
    plt.imshow(img.astype('uint8'), cmap="gray")
else:
    plt.imshow(img.astype('uint8'))

plt.title('Original')
plt.subplot(1, 2, 2)

if is_gray == True:
    plt.imshow(img_int.astype('uint8'), cmap="gray")
else:
   plt.imshow(img_int.astype('uint8'))
print("test")
plt.title('Rescaled by {:2f}'.format(scale_factor))
print('Do not forget to close the plot window --- it happens:) ')
plt.show()
print('done.')
