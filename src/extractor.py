import json
import os
import subprocess
import time

import cv2

from args import args
from audio.audio_utils import extract_percussive, save_audio_data_to_wav
from extraction import find_temperature_of_frame
from movie import Movie
from settings import Settings

# Command line to parse movie file, and extract:
# - images every 15s
# - the audio, to its own file
# - create a json file with some metadata

# def is_mac():
#     return platform.system() == 'Darwin'
#
#
# if is_mac():
#     pytesseract.pytesseract.tesseract_cmd = '/usr/local/Cellar/tesseract/5.3.4/bin/tesseract'

# current working dir
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

settings = Settings(args.input_spec)
start_frame_number = args.start
end_frame_number = args.end
save_images_to_tesseract = args.tesseract
save_temps = args.temps
extract_audio = args.audio
extract_video = args.video
do_a1_audio = args.a1

json_output = {
    "images": []
}

movie = Movie(settings.absolute_movie_file)


def clear_files_from(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def write_image(image, frame_number, identifier):
    hms_safe = movie.hhmmss(frame_number=frame_number)
    unique_name = settings.output_filename(f'images', f'{hms_safe}_{identifier}', 'png')
    cv2.imwrite(unique_name, image)


# Output temps as seconds, temperature, as a .csv file
# This is going to need work.
# TODO: needs start at zero, and intervals must be consistent
def write_times_to_own_csv(temperatures):
    temps_file = settings.output_filename('temps', extension='csv')
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
        sensible = next_value < last_value * 4
    elif last_value < 90:
        sensible = next_value < last_value * 3
    elif last_value < 120:
        sensible = next_value < last_value * 2
    elif last_value < 180:
        sensible = next_value < last_value * 1.5
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
    settings.clear_files_for('images')

    the_images = []
    the_temps = {}
    iteration = 0
    last_temp = 25
    if start_frame_number and end_frame_number:
        frame_range = range(start_frame_number, end_frame_number, int(movie.frame_rate * 15))
    else:
        # Load frames to get, from settings
        frame_range = settings.frame_numbers
        if frame_range is None or len(frame_range) < 32:
            frame_range = movie.get_series_of_quantized_frame_numbers(32, settings.frame_offset)
    for frame_number in frame_range:
        time_in_seconds = frame_number / movie.frame_rate
        frame = movie.get_frame_number(frame_number)
        if frame is not None:
            hms_safe = movie.hhmmss(frame_number, filename_safe=True)
            hms = movie.hhmmss(frame_number)
            image_file = settings.output_filename(f'images', f'{hms_safe}', 'png')
            cv2.imwrite(image_file, frame)
            the_images.append({'hms': hms, 'filename': image_file, 'time': f'{time_in_seconds:.3f}'})
            print(f'Wrote frame {frame_number} to {image_file}')

            def per_frame_handler(lcd_part, frame_number):
                if save_images_to_tesseract:
                    write_image(lcd_part, frame_number, 'tesseract')

            def per_temp_handler(temp, frame_number, blurred):
                if save_temps:
                    filename = "temp_" + ("blurred" if blurred else "normal")
                    write_image(temp, frame_number, filename)

            temperature = find_temperature_of_frame(frame_number, frame, settings, per_frame_handler, per_temp_handler)

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


# Audio, we want about 5 minutes of first_crack, second_crack
# First crack looks to be about 20-24s long. 4 recordings to get a minute, so about 20 total
# It's fine to have other stuff happening (voice, cars, etc) during first crack
#
# Then we want an equal number of recordings for background. Where first crack isn't happening.


def extract_audio_from_movie():

    # Extract the audio from the movie file, and write this as audio.wav to the output directory
    all_of_audio_file = settings.input_file_path("all_audio.wav")
    subprocess.run(['ffmpeg', '-y', '-i', settings.absolute_movie_file, '-vn', all_of_audio_file])

    def extract_sub_audio(output_label_name, start, end):
        output_file = settings.output_filename(output_label_name, extension='wav')
        subprocess.run(
            ['ffmpeg', '-y', '-i', all_of_audio_file, '-ss', start, '-to', end, output_file])

        if do_a1_audio:
            wav, sample_rate = extract_percussive(output_file)
            save_audio_data_to_wav(output_file, wav, sample_rate)

    # Create first_crack and second_crack audio files, bdased on the input spec
    first_crack_start = settings.first_crack_start
    first_crack_end = settings.first_crack_end

    # Use ffmpeg to extract the audio from first_crack_start to first_crack_end
    extract_sub_audio('first_crack', first_crack_start, first_crack_end)

    second_crack_start = settings.second_crack_start
    second_crack_end = settings.second_crack_end

    # Use ffmpeg to extract the audio from second_crack_start to second_crack_end
    extract_sub_audio('second_crack', second_crack_start, second_crack_end)

    # Create a sample of audio, which we'll presume to be background noise
    extract_sub_audio('background_noise', '60', '80')

    # Clean up, remove the old audio file
    os.remove(all_of_audio_file)


if extract_video:
    images, temps = extract_images_and_temps_from_video()
    json_output['images'] = images
    json_output['chamber_temps'] = temps
    write_times_to_own_csv(temperatures=temps)

if extract_audio:
    extract_audio_from_movie()

# Write this to metadata.json, within the output directory
metadata_file = settings.output_filename('metadata', extension='json')
with open(metadata_file, 'w') as f:
    json.dump(json_output, f, indent=2)
