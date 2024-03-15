# Input file specification as JSON
# {
#   "movie_file": "movie.mp4",
#   "output_dir": "output",
#   "first_crack_start": "00:05:00",
#   "first_crack_end": "00:05:30",
#   "second_crack_start": "00:06:00",
#   "second_crack_end": "00:06:30",
#   "target_temp": 242,
#   "digital_area": {
#     "left": 450,
#     "top": 780,
#     "width": 110,
#     "height": 60
#   }
# }
import json
import os

from digital_area import Area


class HSV:
    def __init__(self, h, s, v):
        self.h = h
        self.s = s
        self.v = v

    def to_json(self):
        return {
            "h": self.h,
            "s": self.s,
            "v": self.v,
        }

    @staticmethod
    def from_json(json):
        return HSV(json["h"], json["s"], json["v"])


class Settings:
    def __init__(self, spec_filename):
        # Read the input spec
        with open(spec_filename, 'r') as f:
            self.input_spec = json.load(f)

        self.ensure_input_spec_has_all_fields()
        self.spec_filename = spec_filename

        # Identifier is now based on where the input spec is located.
        # We have a convention of folders, where each roast is in a numbered folder, and input.spec.json is in that folder.
        # This means we can extract our own identifier from the folder name

        # the parent folder for the spec file (spec_filename) should be an integer number >= 1
        parent_folder_name = os.path.basename(os.path.dirname(spec_filename))

        self.identifier = int(parent_folder_name)
        self.identifier = "roast_" + str(self.identifier)

        self.movie_file = self.input_spec["movie_file"]

        # Again, by convention, the output_dir is a sibling to the roast file folders
        self.input_dir = os.path.dirname(spec_filename)
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(spec_filename)), "data-series")
        # this folder should exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.lcd_quad_skew = self.input_spec["lcd_quad_skew"] or 0
        self.target_temp = self.input_spec["target_temp"] or 0
        self.digital_area = Area.from_json(self.input_spec["digital_area"])
        self.first_crack_start = self.input_spec["first_crack_start"]
        self.first_crack_end = self.input_spec["first_crack_end"]
        self.second_crack_start = self.input_spec["second_crack_start"]
        self.second_crack_end = self.input_spec["second_crack_end"]
        self.target_temp = self.input_spec["target_temp"]

        self.low_threshold = HSV.from_json(self.input_spec["lcd_low_threshold"])
        self.upper_threshold = HSV.from_json(self.input_spec["lcd_upper_threshold"])
        self.lcd_blur_amount = self.input_spec["lcd_blur_amount"] or 3

        self.lcd_testing = self.input_spec.get("lcd_testing", {})
        # all the indexes of self.lcd_testing should be integers
        self.lcd_testing = {int(k): v for k, v in self.lcd_testing.items()}
        self.frame_offset = self.input_spec.get("frame_offset", 0)
        self.frame_numbers = self.input_spec.get('frame_numbers', [])

        self.validate()

    def save_values_to_input_spec(self):
        # Save the values to the input spec
        self.input_spec["movie_file"] = self.movie_file
        self.input_spec["lcd_quad_skew"] = self.lcd_quad_skew
        self.input_spec["target_temp"] = self.target_temp
        self.input_spec["digital_area"] = self.digital_area.to_json()
        self.input_spec["first_crack_start"] = self.first_crack_start
        self.input_spec["first_crack_end"] = self.first_crack_end
        self.input_spec["second_crack_start"] = self.second_crack_start
        self.input_spec["second_crack_end"] = self.second_crack_end
        self.input_spec["target_temp"] = self.target_temp

        self.input_spec["lcd_low_threshold"] = self.low_threshold.to_json()
        self.input_spec["lcd_upper_threshold"] = self.upper_threshold.to_json()
        self.input_spec["lcd_blur_amount"] = self.lcd_blur_amount

        self.input_spec["lcd_testing"] = self.lcd_testing
        self.input_spec["frame_offset"] = self.frame_offset
        self.input_spec["frame_numbers"] = self.frame_numbers

    def write_to_file(self):
        # Write the input spec to a file
        self.save_values_to_input_spec()
        folder_for_spec = os.getcwd()
        with open(os.path.join(folder_for_spec, self.spec_filename), 'w') as f:
            json.dump(self.input_spec, f, indent=2)

    def input_file_path(self, filename):
        return os.path.join(self.input_dir, filename)

    @property
    def absolute_movie_file(self):
        return self.input_file_path(self.movie_file)

    def validate(self):
        # all the parameters in the input file specification must exist and be valid
        if not self.movie_file:
            print('Error: movie_file is required')
            exit(1)

        fq_movie_path = self.input_file_path(self.movie_file)
        if not os.path.exists(fq_movie_path):
            print(f'Error: movie_file {fq_movie_path} does not exist')
            exit(1)

        # if not os.path.exists(self.output_dir):
        #     print(f'Error: output_dir {self.output_dir} does not exist, really. I meant it.')
        #     exit(1)

        if not self.first_crack_start:
            print('Error: first_crack_start is required')
            exit(1)

        if not self.first_crack_end:
            print('Error: first_crack_end is required')
            exit(1)

        if not self.second_crack_start:
            print('Error: second_crack_start is required')
            exit(1)

        if not self.second_crack_end:
            print('Error: second_crack_end is required')
            exit(1)

    def set_true_value_for_lcd(self, index: int, value: int):
        self.lcd_testing[index] = value

    def get_true_value_for_lcd(self, index: int) -> int or None:
        return self.lcd_testing.get(index, None)

    def output_filename(self, label, filename: str = None, extension: str = None):
        if filename is not None:
            actual_filename = f'{label}_{self.identifier}_{filename}.{extension}'
        else:
            actual_filename = f'{label}_{self.identifier}.{extension}'

        current_working = os.getcwd()
        full_path = os.path.join(current_working, self.output_dir, label, actual_filename)
        # check folder exists
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        return full_path

    def clear_files_for(self, label):
        folder = os.path.join(self.output_dir, label)
        # Ensure folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            if file_path.__contains__(self.identifier):
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)

    def ensure_input_spec_has_all_fields(self):
        # Ensure that all fields are present in the input spec
        if "lcd_quad_skew" not in self.input_spec:
            self.input_spec["lcd_quad_skew"] = -16

        if "lcd_blur_amount" not in self.input_spec:
            self.input_spec["lcd_blur_amount"] = 3

        if "target_temp" not in self.input_spec:
            self.input_spec["target_temp"] = 242

        if "lcd_low_threshold" not in self.input_spec:
            self.input_spec["lcd_low_threshold"] = HSV(70, 200, 100).to_json()

        if "lcd_upper_threshold" not in self.input_spec:
            self.input_spec["lcd_upper_threshold"] = HSV(100, 255, 255).to_json()

        if "digital_area" not in self.input_spec:
            self.input_spec["digital_area"] = Area(470, 795, 90, 42).to_json()

        if "output_dir" not in self.input_spec:
            self.input_spec["output_dir"] = "."

        # First and second crack values
        if "first_crack_start" not in self.input_spec:
            self.input_spec["first_crack_start"] = "00:05:00"

        if "first_crack_end" not in self.input_spec:
            self.input_spec["first_crack_end"] = "00:05:30"

        if "second_crack_start" not in self.input_spec:
            self.input_spec["second_crack_start"] = "00:06:00"

        if "second_crack_end" not in self.input_spec:
            self.input_spec["second_crack_end"] = "00:06:30"
