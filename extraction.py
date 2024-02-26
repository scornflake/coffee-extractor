import cv2
from settings import Settings

import numpy as np
import pytesseract


def get_temperature_part_from_full_frame(frame, the_settings: Settings) -> int or None:
    # Extract the digital area from the frame, and find the temperature
    area = the_settings.digital_area
    digital_number_area = frame[area.top:area.bottom, area.left:area.right]

    # perspective correct the LCD
    top_left = list(area.top_left)
    top_right = list(area.top_right)
    bottom_left_skewed = list(area.bottom_left_skewed(the_settings.lcd_quad_skew))
    bottom_right_skewed = list(area.bottom_right_skewed(the_settings.lcd_quad_skew))

    pts1 = np.float32([top_left, top_right, bottom_left_skewed, bottom_right_skewed])
    pts2 = np.float32(
        [[0, 0], [area.width, 0], [0, area.height], [area.width, area.height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, matrix, (digital_number_area.shape[1], digital_number_area.shape[0]))


def extract_digits_from_readout(image, the_settings: Settings):
    # Invert the image
    inverted = cv2.bitwise_not(image)
    img_hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

    # Filter the image so that we're left with only cyan/green
    lower = np.array([the_settings.low_threshold.h, the_settings.low_threshold.s, the_settings.low_threshold.v])
    upper = np.array([the_settings.upper_threshold.h, the_settings.upper_threshold.s, the_settings.upper_threshold.v])
    mask = cv2.inRange(img_hsv, lower, upper)
    green_only = cv2.bitwise_and(inverted, inverted, mask=mask)

    return green_only


def parse_int_via_tesseract(image) -> int or None:
    custom_config = r'--psm 11 -c tessedit_char_whitelist=.0123456789'
    # text = pytesseract.image_to_string(image, config=custom_config, lang="letsgodigital")
    text = pytesseract.image_to_string(image, config=custom_config, lang="lets")

    # Strip any '.' from text
    text = text.replace('.', '')

    # Parse 'text' as an integer
    try:
        return int(text)
    except ValueError:
        return None


def try_to_get_number_from(contours, frame_number: int, image, thickness: int = 1,
                           write_detector_image: bool = True) -> int or None:
    img_copy = np.zeros_like(image)

    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), thickness)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    # img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_ELLIPSE, kernel)

    # cv2.drawContours(img_copy, contours, -1, (255, 255, 255), 2)
    if write_detector_image:
        write_image(img_copy, frame_number, f'contours_{thickness}')
    return parse_int_via_tesseract(img_copy)


# def find_temperature_of_frame2(frame_number, frame) -> int or None:
#     digital_number_area = get_temperature_part_from_full_frame(frame, the_settings=settings)
#     digital_number_area = extract_digits_from_readout(digital_number_area, sensitivity=10)
#
#     img = digital_number_area
#     img_copy = img.copy()
#     write_image(img_copy, frame_number, 'digital_area')
#
#     digital_number_area = cv2.cvtColor(digital_number_area, cv2.COLOR_BGR2GRAY)
#     contours, _ = cv2.findContours(digital_number_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     attempt_1 = try_to_get_number_from(contours, frame_number, digital_number_area, thickness=1)
#     return attempt_1


def new_image_from_contours(image, thickness: int = 2):
    digital_number_area = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(digital_number_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_copy = np.zeros_like(image)
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), thickness)
    return img_copy


def edged_image(image):
    gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17)
    return cv2.Canny(gray_image, 30, 200)


def make_threshold(digital_number_area, lower: int = 127):
    ret, digital_number_area = cv2.threshold(digital_number_area, lower, 255, cv2.THRESH_BINARY)
    return digital_number_area


def extract_lcd_and_ready_for_teseract(frame, the_settings: Settings):
    # Extract the digital area from the frame, and find the temperature
    extracted_frame = get_temperature_part_from_full_frame(frame, the_settings=the_settings)
    extracted_frame = extract_digits_from_readout(extracted_frame, the_settings=the_settings)
    blur = the_settings.lcd_blur_amount
    kernel = np.ones((blur, blur), np.float32) / 5
    extracted_frame = cv2.filter2D(extracted_frame, -1, kernel)
    extracted_frame = dilate_with_kernel(extracted_frame, kernel_size=2)
    return extracted_frame


def find_temperature_of_frame(frame_number, frame, the_settings: Settings, frame_handler) -> int or None:
    # Extract the digital area from the frame, and find the temperature
    digital_number_area = extract_lcd_and_ready_for_teseract(frame, the_settings=the_settings)

    # digital_number_area = new_image_from_contours(digital_number_area, thickness=1)
    # digital_number_area = make_threshold(digital_number_area)

    # Mess with it to make it easier for OCR to read
    # digital_number_area = dilate_with_kernel(digital_number_area, kernel_size=2)
    # digital_number_area = edged_image(digital_number_area)

    frame_handler(digital_number_area, frame_number)

    # If it's green at all - make it 100%
    # digital_number_area[np.where((digital_number_area > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return parse_int_via_tesseract(digital_number_area)


def dilate_with_kernel(image, kernel_size: int = 3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)
