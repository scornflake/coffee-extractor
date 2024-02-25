import argparse

import cv2
import os
import time
import json
import subprocess

import numpy as np
import pytesseract
import digital_area
from movie import Movie
from settings import Settings
from args import args

# Command line to parse movie file, and extract:
# - images every 15s
# - the audio, to its own file
# - create a json file with the following structure:
# {
#   "images": [
#     { "time": "00:00:15", "filename": "image1.png" },
#     { "time": "00:00:30", "filename": "image2.png" },
#     ...
#   ],
#   "audio": "audio.mp3",
#   "first_crack_audio": "first_crack.mp3",
#   "second_crack_audio": "second_crack.mp3",
#   "chamber_temps": [
#     { "time": "00:00:15", "temp": 23.5 },
#     { "time": "00:00:30", "temp": 24.0 },
#     ...
#   ]
# }

pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/5.3.4/bin/tesseract'

settings = Settings(args.input_spec)
start_frame_number = args.start
end_frame_number = args.end

json_output = {
    "images": []
}

movie = Movie(settings.movie_file)
output_images_dir = os.path.join(settings.output_dir, 'images')

digital_area = settings.digital_area


def test_mode():
    # Get an image, at say, the 1m mark.
    # Draw a rectangle around it based on input spec crop area, and output it to disk for the user to preview
    frame = movie.get_frame_number(4000)
    if frame:
        area = digital_area
        cv2.rectangle(frame, area.top_left, area.bottom_right, (0, 255, 0), 2)

        bottom_left = area.bottom_left_skewed(settings.lcd_quad_skew)
        bottom_right = area.bottom_right_skewed(settings.lcd_quad_skew)

        cv2.circle(frame, area.top_left, 5, (0, 0, 255), -1)
        cv2.circle(frame, area.top_right, 5, (0, 0, 255), -1)
        cv2.circle(frame, bottom_left, 5, (0, 0, 255), -1)
        cv2.circle(frame, bottom_right, 5, (0, 0, 255), -1)

        cv2.imshow('frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if args.test:
    print(f'Running in test mode')
    test_mode()
    exit(0)


def get_temperature_part_from_full_frame(frame) -> int or None:
    # Extract the digital area from the frame, and find the temperature
    digital_number_area = frame[digital_area.top:digital_area.bottom, digital_area.left:digital_area.right]

    # perspective correct the LCD
    top_left = list(digital_area.top_left)
    top_right = list(digital_area.top_right)
    bottom_left_skewed = list(digital_area.bottom_left_skewed(settings.lcd_quad_skew))
    bottom_right_skewed = list(digital_area.bottom_right_skewed(settings.lcd_quad_skew))

    pts1 = np.float32([top_left, top_right, bottom_left_skewed, bottom_right_skewed])
    pts2 = np.float32(
        [[0, 0], [digital_area.width, 0], [0, digital_area.height], [digital_area.width, digital_area.height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(frame, matrix, (digital_number_area.shape[1], digital_number_area.shape[0]))


def extract_digits_from_readout(image, sensitivity=0):
    # Invert the image
    inverted = cv2.bitwise_not(image)
    img_hsv = cv2.cvtColor(inverted, cv2.COLOR_BGR2HSV)

    # Filter the image so that we're left with only cyan/green
    lower = np.array([70, 200 - sensitivity, 100])
    upper = np.array([100, 255, 255])
    mask = cv2.inRange(img_hsv, lower, upper)
    green_only = cv2.bitwise_and(inverted, inverted, mask=mask)

    return green_only


def write_image(image, frame_number, identifier):
    hms_safe = movie.hhmmss(frame_number=frame_number)
    unique_name = os.path.join(output_images_dir, f'image_{hms_safe}_{identifier}.png')
    cv2.imwrite(unique_name, image)


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


def find_temperature_of_frame2(frame_number, frame) -> int or None:
    digital_number_area = get_temperature_part_from_full_frame(frame)
    digital_number_area = extract_digits_from_readout(digital_number_area, sensitivity=10)

    img = digital_number_area
    img_copy = img.copy()
    write_image(img_copy, frame_number, 'digital_area')

    digital_number_area = cv2.cvtColor(digital_number_area, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(digital_number_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    attempt_1 = try_to_get_number_from(contours, frame_number, digital_number_area, thickness=1)
    return attempt_1


def new_image_from_contours(image, thickness: int = 2):
    digital_number_area = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(digital_number_area, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_copy = np.zeros_like(image)
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            cv2.drawContours(img_copy, [cnt], -1, (255, 255, 255), thickness)
    return img_copy


# Output temps as seconds, temperature, as a .csv file
def write_times_to_own_csv(temperatures):
    temps_file = os.path.join(settings.output_dir, 'temps.csv')
    with open(temps_file, 'w') as f:
        for time_in_seconds, temp in temperatures.items():
            f.write(f'{time_in_seconds},{temp}\n')


def edged_image(image):
    gray_image = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17)
    return cv2.Canny(gray_image, 30, 200)


def make_threshold(digital_number_area, lower: int = 127):
    ret, digital_number_area = cv2.threshold(digital_number_area, lower, 255, cv2.THRESH_BINARY)
    return digital_number_area


def find_temperature_of_frame(frame_number, frame) -> int or None:
    # Extract the digital area from the frame, and find the temperature
    digital_number_area = get_temperature_part_from_full_frame(frame)
    digital_number_area = extract_digits_from_readout(digital_number_area, sensitivity=0)
    kernel = np.ones((3, 3), np.float32) / 5
    digital_number_area = cv2.filter2D(digital_number_area, -1, kernel)
    digital_number_area = dilate_with_kernel(digital_number_area, kernel_size=2)
    write_image(digital_number_area, frame_number, 'digits_actual')

    # digital_number_area = new_image_from_contours(digital_number_area, thickness=1)
    # digital_number_area = make_threshold(digital_number_area)

    # Mess with it to make it easier for OCR to read
    # digital_number_area = dilate_with_kernel(digital_number_area, kernel_size=2)
    # digital_number_area = edged_image(digital_number_area)

    write_image(digital_number_area, frame_number, 'teseract')

    # If it's green at all - make it 100%
    # digital_number_area[np.where((digital_number_area > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    return parse_int_via_tesseract(digital_number_area)


def dilate_with_kernel(image, kernel_size: int = 3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


time_last_value_set = time.time()


def is_temp_jmp_sensible(last_value, next_value) -> bool:
    global time_last_value_set

    # sensible = not too large of a jump. When temp is low, the jump can be larger.  When temp is high, the allowable jump is smaller
    # as the distance grows between last_value and current_value, the allowable jump becomes larger

    # Work out if a new value is allowable or not. We know the last_value, and num_seconds_since_last_value was set.

    distance = abs(last_value - next_value)
    sensible = False
    if last_value < 90:
        sensible = next_value < last_value * 2
    elif last_value < 120:
        sensible = next_value < last_value * 1.5
    elif last_value < 180:
        sensible = next_value < last_value * 1.3
    else:
        sensible = next_value < last_value * 1.2
    if sensible:
        time_last_value_set = time.time()
        print(f"**** SET: last: {last_value}, current: {next_value}, sensible: {sensible}")
    else:
        print(f"NOT SET: last: {last_value}, current: {next_value}, sensible: {sensible}")
    return sensible


# Extract images. One every 15s, using OpenCV
def extract_images_and_temps_from_video():
    print(
        f'Frame rate: {movie.frame_rate}, width: {movie.frame_width}, height: {movie.frame_height}, frame count: {movie.frame_count}, duration: {movie.duration}')

    # Extract video frames every 15s, writing each to the output directory/images

    # Remove images from output_images_dir
    for the_file in os.listdir(output_images_dir):
        file_path = os.path.join(output_images_dir, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    # Extract the images from the movie file
    os.makedirs(output_images_dir, exist_ok=True)
    the_images = []
    the_temps = {}
    iteration = 0
    last_temp = 25
    if start_frame_number and end_frame_number:
        frame_range = range(start_frame_number, end_frame_number, int(movie.frame_rate * 15))
    else:
        frame_range = range(0, movie.frame_count, int(movie.frame_rate * 15))
    for frame_number in frame_range:
        time_in_seconds = frame_number / movie.frame_rate
        frame = movie.get_frame_number(frame_number)
        if frame is not None:
            hms_safe = movie.hhmmss(frame_number, filename_safe=True)
            hms = movie.hhmmss(frame_number)
            image_file = os.path.join(output_images_dir, f'image_{hms_safe}.png')
            cv2.imwrite(image_file, frame)
            the_images.append({'hms': hms, 'filename': image_file, 'time': f'{time_in_seconds:.3f}'})
            print(f'Wrote frame {frame_number} to {image_file}')
            temperature = find_temperature_of_frame(frame_number, frame)

            # Temps must always increase, but not equal to the previous temp
            if temperature is not None:
                sensible = is_temp_jmp_sensible(last_temp, temperature)

                # print(f"is {temperature} sensible: {sensible}. In range: {last_temp * 0.5} < {temperature} < {last_temp * 1.5}")
                if last_temp < temperature < settings.target_temp and sensible:
                    the_temps[time_in_seconds] = temperature
                    last_temp = temperature
                    print(f"saw temp {temperature} at time {time_in_seconds}")
        else:
            print(f'Error reading frame {frame_number}')

        iteration += 1
        # if iteration > 5:
        #     return images

    return the_images, the_temps


def extract_audio_from_movie():
    # Extract the audio from the movie file, and write this as audio.wav to the output directory
    audio_file = os.path.join(settings.output_dir, 'audio.wav')
    subprocess.run(['ffmpeg', '-y', '-i', settings.movie_file, '-vn', audio_file])

    # Create first_crack and second_crack audio files, based on the input spec
    first_crack_start = settings.first_crack_start
    first_crack_end = settings.first_crack_end

    # Use ffmpeg to extract the audio from first_crack_start to first_crack_end
    first_crack_audio_file = os.path.join(settings.output_dir, 'first_crack.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', audio_file, '-ss', first_crack_start, '-to', first_crack_end, first_crack_audio_file])

    second_crack_start = settings.second_crack_start
    second_crack_end = settings.second_crack_end

    # Use ffmpeg to extract the audio from second_crack_start to second_crack_end
    second_crack_audio_file = os.path.join(settings.output_dir, 'second_crack.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', audio_file, '-ss', second_crack_start, '-to', second_crack_end, second_crack_audio_file])

    # Create 3 samples of audio that are 10s long each, that do not overlap with the first or second crack times
    # These will be used to train the AI
    # The first sample will be from 1:00 to 1:10
    # The second sample will be from 2:00 to 2:10
    # The third sample will be from 3:00 to 3:10
    sample1_audio_file = os.path.join(settings.output_dir, 'sample1.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', audio_file, '-ss', '60', '-to', '70', sample1_audio_file])

    sample2_audio_file = os.path.join(settings.output_dir, 'sample2.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', audio_file, '-ss', '120', '-to', '130', sample2_audio_file])

    sample3_audio_file = os.path.join(settings.output_dir, 'sample3.wav')
    subprocess.run(
        ['ffmpeg', '-y', '-i', audio_file, '-ss', '180', '-to', '190', sample3_audio_file])


images, temps = extract_images_and_temps_from_video()
json_output['images'] = images
json_output['chamber_temps'] = temps

# extract_audio_from_movie()
write_times_to_own_csv(temperatures=temps)

# Write this to metadata.json, within the output directory
metadata_file = os.path.join(settings.output_dir, 'metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(json_output, f, indent=2)
