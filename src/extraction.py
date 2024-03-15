import cv2
import imutils

from settings import Settings

import numpy as np
import pytesseract


def get_temperature_part_from_full_frame(frame, the_settings: Settings) -> int or None:
    # Extract the digital area from the frame, and find the temperature
    area = the_settings.digital_area
    digital_number_area = frame[area.top:area.bottom, area.left:area.right, :]

    # perspective correct the LCD
    top_left = list(area.top_left)
    top_right = list(area.top_right)
    bottom_left_skewed = list(area.bottom_left_skewed(the_settings.lcd_quad_skew))
    bottom_right_skewed = list(area.bottom_right_skewed(the_settings.lcd_quad_skew))

    pts1 = np.float32([top_left, top_right, bottom_left_skewed, bottom_right_skewed])
    pts2 = np.float32(
        [[0, 0], [area.width, 0], [0, area.height], [area.width, area.height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    digital_number_area = cv2.warpPerspective(frame, matrix,
                                              (digital_number_area.shape[1], digital_number_area.shape[0]))
    digital_number_area = cv2.cvtColor(digital_number_area, cv2.COLOR_BGR2RGB)
    # make image bigger, so that we can FAR better get contours out
    digital_number_area = cv2.resize(digital_number_area, (0, 0), fx=3, fy=3)

    return digital_number_area


def extract_digits_from_readout2(image, the_settings: Settings):
    # Converts images from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_red = np.array([the_settings.low_threshold.h, the_settings.low_threshold.s, the_settings.low_threshold.v])
    upper_red = np.array(
        [the_settings.upper_threshold.h, the_settings.upper_threshold.s, the_settings.upper_threshold.v])

    # This creates a mask of red coloured objects found in the frame.
    # i.e: digits, and little decimal place thingies
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # redraw the mask, using contours, so we get ignore small parts
    # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0]
    # if cnts is None or len(cnts) == 0:
    #     return None
    #
    # mask = np.ones(image.shape[:2], dtype="uint8") * 0
    # for c in cnts:
    #     # if the contour is not sufficiently large, ignore it
    #     if cv2.contourArea(c) > 100:
    #         cv2.drawContours(mask, [c], -1, 255, -1)
    #         continue

    # Blur the image
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.GaussianBlur(res, (the_settings.lcd_blur_amount, the_settings.lcd_blur_amount), 2)
    # Edge detection. Lines around the visible things.
    edged = cv2.Canny(res, 100, 255)

    # Dilate it , number of iterations will depend on the image
    dilate = cv2.dilate(edged, None, iterations=7)
    # perform erosion
    erode = cv2.erode(dilate, None, iterations=6)

    # make an empty mask. This is pure white.
    mask2 = np.ones(image.shape[:2], dtype="uint8") * 255

    # find contours in the final erode
    cnts = cv2.findContours(erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2(True) else cnts[1]
    if cnts is None or len(cnts) == 0:
        return None
        # raise ValueError("No contours found in image")

    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 600:
            cv2.drawContours(mask2, [c], -1, 0, -1)
            continue

    # Remove ignored contours
    newimage = cv2.bitwise_and(erode.copy(), dilate.copy(), mask=mask2)
    # return newimage
    # Again perform dilation and erosion
    # newimage = cv2.dilate(newimage, None, iterations=1)
    # newimage = cv2.erode(newimage, None, iterations=1)
    ret, newimage = cv2.threshold(newimage, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return newimage


def extract_digits_from_readout(image, the_settings: Settings):
    # Invert the image so that we are targeting GREEN
    # inverted = cv2.bitwise_not(image)
    inverted = image

    # Push into HSV for filtering
    img_hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

    # Filter the image so that we're left with only cyan/green
    lower = np.array([the_settings.low_threshold.h, the_settings.low_threshold.s, the_settings.low_threshold.v])
    upper = np.array([the_settings.upper_threshold.h, the_settings.upper_threshold.s, the_settings.upper_threshold.v])
    mask = cv2.inRange(img_hsv, lower, upper)
    just_the_digits = cv2.bitwise_and(inverted, inverted, mask=mask)

    return just_the_digits


def extract_digits_from_greyscale_readout(image, the_settings: Settings):
    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
    return mask


def parse_int_via_tesseract(image) -> int or None:
    custom_config = r'--psm 8 -c tessedit_char_whitelist=.0123456789'
    # text = pytesseract.image_to_string(image, config=custom_config, lang="letsgodigital")
    # text = pytesseract.image_to_string(image, config=custom_config, lang="lets")
    # text = pytesseract.image_to_string(image, config=custom_config, lang="genecafe")
    text = pytesseract.image_to_string(image, config=custom_config, lang="genecafefast")
    # text = pytesseract.image_to_string(image, config=custom_config)

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


def sharpen(image, amount=5):
    # sharpen the image
    blurred = cv2.GaussianBlur(image, (amount, amount), 0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def extract_lcd_and_ready_for_tesseract(frame, frame_number, the_settings: Settings, temps_handler=None):
    # Extract the digital area from the frame, and find the temperature
    extracted_frame = get_temperature_part_from_full_frame(frame, the_settings=the_settings)
    extracted_frame = extract_digits_from_readout(extracted_frame, the_settings=the_settings)
    extracted_frame = cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2GRAY)
    extracted_frame = extract_digits_from_greyscale_readout(extracted_frame, the_settings=the_settings)
    return extracted_frame
    # extracted_frame = sharpen(extracted_frame, amount=23)

    # Try a morphological close
    kernel1 = cv2.getStructuringElement(cv2.MORPH_OPEN, (3, 3))
    close = cv2.morphologyEx(extracted_frame, cv2.MORPH_OPEN, kernel1)
    return close
    # convert to greyscale
    # extracted_frame = sharpen(extracted_frame)

    div = np.float32(extracted_frame) / (close)
    extracted_frame = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    return extracted_frame

    # return extracted_frame
    # return extracted_frame
    # extracted_frame = new_image_from_contours(extracted_frame, thickness=3)

    if extracted_frame is None:
        print("No digits found in frame", frame_number)
        return None

    if temps_handler:
        temps_handler(extracted_frame, frame_number, False)

    # Dilate and then erode
    extracted_frame = cv2.dilate(extracted_frame, None, iterations=2)
    extracted_frame = cv2.erode(extracted_frame, None, iterations=3)
    # extracted_frame = sharpen(extracted_frame)
    # return extracted_frame

    blur = the_settings.lcd_blur_amount
    if blur > 0:
        kernel = np.ones((blur, blur), np.float32) / 5
        extracted_frame = cv2.filter2D(extracted_frame, -1, kernel)

    if temps_handler:
        temps_handler(extracted_frame, frame_number, True)

    return extracted_frame


def extract_lcd_and_ready_for_tesseract2(frame, frame_number, the_settings: Settings, temps_handler=None):
    # Extract the digital area from the frame, and find the temperature
    extracted_frame = get_temperature_part_from_full_frame(frame, the_settings=the_settings)
    extracted_frame = extract_digits_from_readout2(extracted_frame, the_settings=the_settings)

    if temps_handler:
        temps_handler(extracted_frame, frame_number, True)

    return extracted_frame


def find_temperature_of_frame(frame_number, frame, the_settings: Settings, frame_handler, temps_handler) -> int or None:
    # Extract the digital area from the frame, and find the temperature
    digital_number_area = extract_lcd_and_ready_for_tesseract(frame, frame_number, the_settings=the_settings,
                                                              temps_handler=temps_handler)

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
