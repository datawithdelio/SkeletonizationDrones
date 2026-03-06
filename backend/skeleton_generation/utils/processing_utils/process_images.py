import cv2 as cv
import numpy as np
import base64


def get_pixel_coordinates(image, scale_x, scale_y):
    non_zero_indices = np.nonzero(np.any(image != [255, 255, 255], axis=-1))
    scaled_coords = zip(non_zero_indices[1] * scale_x, non_zero_indices[0] * scale_y)
    return list(scaled_coords)


def _largest_external_contours(binary_mask, min_area_ratio=0.0008):
    contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    min_area = max(25, int(binary_mask.shape[0] * binary_mask.shape[1] * min_area_ratio))
    return [c for c in contours if cv.contourArea(c) >= min_area]


def process_image(image):
    target_height = image.shape[0]

    aspect_ratio = image.shape[1] / image.shape[0]
    target_width = int(target_height * aspect_ratio)

    scale_x = target_width / image.shape[1]
    scale_y = target_height / image.shape[0]

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    border = np.concatenate((binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]))
    border_is_bright = np.mean(border) > 127
    if border_is_bright:
        binary = cv.bitwise_not(binary)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)

    contours = _largest_external_contours(binary)

    white_image = np.ones_like(image) * 255

    # Draw blue contours on the white image
    cv.drawContours(white_image, contours, -1, (255, 0, 0), 1)

    # Now, we'll convert this image to a base64 string
    _, buffer = cv.imencode(".png", white_image)
    outline_img_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    coordinates = get_pixel_coordinates(white_image, scale_x, scale_y)
    height = binary.shape[0]
    contour_strings = [
        "{:.7e} {:.7e}".format(float(point[0][0]), float(height - point[0][1]))
        for contour in contours
        for point in contour
    ]

    output = {
        "coordinates": coordinates,
        "contour_strings": contour_strings,
        "outline_img_base64": outline_img_base64,
    }
    return output
