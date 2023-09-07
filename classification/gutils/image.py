import numpy as np
import cv2


def extract_largest_cc(mask: np.ndarray):
    mask = mask.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape, dtype=np.uint8)
    img2[output == max_label] = 1
    return img2


def smooth(mask: np.ndarray):
    mask = cv2.blur(mask, (9, 9))
    # mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask.astype('uint8')


def apply_mask_crop(image: np.ndarray, mask: np.ndarray, only_largest=False, smooth_mask=False):
    mask[mask > 0] = 1

    if only_largest:
        mask = extract_largest_cc(mask)

    if smooth_mask:
        mask = smooth(mask)

    masked_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(mask)

    masked_image = masked_image[y:y + h, x:x + w, :]

    nc = masked_image.shape[2]
    width_image = masked_image.shape[1]
    height_image = masked_image.shape[0]
    max_dim = max(width_image, height_image)
    padded_image = np.zeros((max_dim, max_dim, nc), dtype=np.uint8)
    yc = xc = max_dim // 2
    half_height = height_image // 2
    half_width = width_image // 2
    if max_dim == width_image:
        carry = 0 if half_height * 2 == masked_image.shape[0] else 1
        padded_image[yc - half_height:yc + half_height + carry, :, :] = masked_image
    elif max_dim == height_image:
        carry = 0 if half_width * 2 == masked_image.shape[1] else 1
        padded_image[:, xc - half_width:xc + half_width + carry, :] = masked_image
    return padded_image
