import argparse

import cv2
import os
import time
import json
import subprocess

import pytesseract
from extraction import find_temperature_of_frame
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
# Make sure this dir exists
os.makedirs(output_images_dir, exist_ok=True)


def write_image(image, frame_number, identifier):
    hms_safe = movie.hhmmss(frame_number=frame_number)
    unique_name = os.path.join(output_images_dir, f'image_{hms_safe}_{identifier}.png')
    cv2.imwrite(unique_name, image)


# Output temps as seconds, temperature, as a .csv file
def write_times_to_own_csv(temperatures):
    temps_file = os.path.join(settings.output_dir, 'temps.csv')
    with open(temps_file, 'w') as f:
        for time_in_seconds, temp in temperatures.items():
            f.write(f'{time_in_seconds},{temp}\n')


time_last_value_set = time.time()


def is_temp_jmp_sensible(last_value, next_value) -> bool:
    global time_last_value_set

    # sensible = not too large of a jump. When temp is low, the jump can be larger.  When temp is high, the allowable jump is smaller
    # as the distance grows between last_value and current_value, the allowable jump becomes larger

    # Work out if a new value is allowable or not. We know the last_value, and num_seconds_since_last_value was set.

    distance = abs(last_value - next_value)
    sensible = False
    if last_value < 50:
        sensible = next_value < last_value * 3.5
    elif last_value < 90:
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

            def per_frame_handler(lcd_part, frame_number):
                write_image(lcd_part, frame_number, 'teseract')

            temperature = find_temperature_of_frame(frame_number, frame, settings, per_frame_handler)

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

extract_audio_from_movie()
write_times_to_own_csv(temperatures=temps)

# Write this to metadata.json, within the output directory
metadata_file = os.path.join(settings.output_dir, 'metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(json_output, f, indent=2)
