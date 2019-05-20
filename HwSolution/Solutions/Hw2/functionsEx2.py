
import numpy as np
import math
import functionsEx1 as pinc

def create_edge_magn_image(image, dx, dy):
    # this function created an edge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @gdx     : gradient along x axis
    # @gdy     : geadient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direction of gradients in every pixel

    # Computing mag in x and in y
    mag_x = np.asarray(pinc.myconv2(image, dx))[:, round(dx.shape[1]/2):image.shape[1] + round(dx.shape[1]/2)]
    mag_y = np.asarray(pinc.myconv2(image, dy))[round(dy.shape[0]/2):image.shape[0] + round(dy.shape[0]/2), :]
    # computing magnitude
    grad_mag_image = np.sqrt(np.power(mag_x,2) + np.power(mag_y,2))
    # correcting grad_mag_image shape
    grad_mag_image = grad_mag_image[:grad_mag_image.shape[0] - round(dy.shape[0]/2), :grad_mag_image.shape[1] - round(dx.shape[1]/2)]
    # computing angles
    grad_dir_image = np.angle(mag_x + 1j*mag_y)
    # correcting grad_dir_image shape
    grad_dir_image = grad_dir_image[:grad_dir_image.shape[0] - round(dy.shape[0]/2), :grad_dir_image.shape[1] - round(dx.shape[1]/2)]
    # making all angles positive
    indexes = np.where(grad_dir_image < 0)
    grad_dir_image[indexes[0], indexes[1]] = grad_dir_image[indexes[0], indexes[1]] + 2*math.pi
    return grad_mag_image, grad_dir_image


# Edge images of particular directions
def make_edge_map(image, dx, dy):
    # INPUTS
    # @image        : a 2D image
    # @gdx          : gradient along x axis
    # @gdy          : geadient along y axis
    # OUTPUTS:
    # @ edge maps   : a 3D array of shape (image.shape, 8) containing the edge maps on 8 orientations
    # creating grad_mag_image and grad_dir_image
    (grad_mag_image, grad_dir_image) = create_edge_magn_image(image, dx, dy)
    # selecting threshold
    r = 25
    # creating empty edge maps matrix
    edge_maps = np.zeros([image.shape[0], image.shape[1], 8])
    for i in range(8):
        # computing actual minimum and maximum angle
        minAngle = i*2*math.pi/8 - math.pi/8
        maxAngle = i*2*math.pi/8 + math.pi/8
        # condition for angles minAngle smaller than zero
        if i == 0:
            index = np.where((grad_mag_image > r) &
                             ((grad_dir_image >= 2*math.pi + minAngle) |
                              (grad_dir_image < maxAngle)))
        else:
            index = np.where((grad_mag_image > r) &
                             (grad_dir_image >= minAngle) &
                             (grad_dir_image < maxAngle))
        edge_maps[index[0], index[1], i] = 255
    return edge_maps

# Edge non max suppresion
def edge_non_max_suppression(img_edge_mag, edge_maps):
    # This function performs non maximum supresion, in order to reduce the width of the edge response
    # INPUTS
    # @img_edge_mag   : 2d image, with the magnitude of gradients in every pixel
    # @edge_maps      : 3d image, with the edge maps
    # OUTPUTS
    # @non_max_sup    : 2d image with the non max supresed pixels
    # coping img_edge_mag
    non_max_sup = np.array(img_edge_mag[:, :])
    # definition of the gradient length
    gHalfLength = 5
    for i in range(8):
        index = np.where(edge_maps[:, :, i] == 255)
        # definition of the gradients
        if (i == 0) or (i == 4):
            (gradX, gradY) = (1, 0)
        if (i == 1) or (i == 5):
            (gradX, gradY) = (1, 1)
        if (i == 2) or (i == 6):
            (gradX, gradY) = (0, 1)
        if (i == 3) or (i == 7):
            (gradX, gradY) = (-1, 1)
        for x in range(index[1].shape[0]):
            (newY, newX) = (index[0][x], index[1][x])
            # assuring that there will be not problem with index
            if (newX - gHalfLength*math.sqrt(gradX*gradX) >= 0) and (newX + gHalfLength*math.sqrt(gradX*gradX) < img_edge_mag.shape[1]) and \
                    (newY - gHalfLength*math.sqrt(gradY*gradY) >= 0) and (newY + gHalfLength*math.sqrt(gradY*gradY) < img_edge_mag.shape[0]):
                neighMaxX = newX - gHalfLength*gradX
                neighMaxY = newY - gHalfLength*gradY
                # Actually computing the max-non suppression
                for it in range(-gHalfLength, gHalfLength + 1):
                    if img_edge_mag[neighMaxY, neighMaxX] < img_edge_mag[newY + it*gradY, newX + it*gradX]:
                        non_max_sup[neighMaxY, neighMaxX] = 0
                        neighMaxX = newX + it*gradX
                        neighMaxY = newY + it*gradY
                    else:
                        non_max_sup[newY + it*gradY, newX + it*gradX] = 0
    return non_max_sup

