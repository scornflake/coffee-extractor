# generate a bunch of random numbers, from 0 to 250, as temperatures
# also add the three dots that we see in the temp readings from the roaster

import os
import subprocess
from multiprocessing import Process

# Seed random number generator for reproducibility
from random import seed, randint

seed(1)

"""
To get this to work, I had to install the fonts I wanted to use, on the OS
"""


class GroundTruthCreation:
    def __init__(self):
        # number of ground truths to create
        self.number_of_truths_to_create = 8000

        # Scan the fonts folder. Only the top level. Make up a list of fonts
        self.fonts_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'fonts')
        self.fonts = []
        for font in os.listdir(self.fonts_folder):
            if font.endswith('.ttf'):
                self.fonts.append(font)

        self.path_to_self = os.path.dirname(os.path.realpath(__file__))
        self.ground_truth_output_folder = os.path.join(self.path_to_self, 'data/genecafe-ground-truth')
        print(f"Will output to: {self.ground_truth_output_folder}")
        print(f"Using {len(self.fonts)} fonts.")

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.ground_truth_output_folder):
            os.makedirs(self.ground_truth_output_folder)

    def create_truth_i(self, iteration: int):
        filename = f"genecafe_{iteration}"
        ground_truth_file = os.path.join(self.ground_truth_output_folder, filename + ".gt.txt")
        with open(ground_truth_file, 'w') as f:
            # Create a random number from 0 to 250
            random_number = str(iteration % 250)
            # random_number = str(randint(0, 250))
            # put dots in between the digits, randomly
            f.write(random_number)

            # Now write another 10 numbers, random from 1-10000 as doubles
            for _ in range(20):
                f.write(" ")
                f.write(str(randint(1, 1000000) / 100))

        def create_image_from_text(iter: int):
            # Now use text2image to create an image of this text, using one of the fonts
            font_to_use = self.fonts[iter % len(self.fonts)]
            font_without_path = os.path.basename(font_to_use)
            name_of_font_excluding_extension = font_without_path.split('.')[0]
            use_distortion = iter % 2 == 0
            use_rotation = True
            exposure = randint(-10, 0)
            # noinspection PyListCreation
            args = [
                "text2image",
                "--fonts_dir=" + self.fonts_folder,
                f"--exposure={exposure}",
                "--font", name_of_font_excluding_extension,
                "--text", ground_truth_file,
                "--outputbase", os.path.join(self.ground_truth_output_folder, filename),
                "--ptsize=20",
                "--xsize=7000",
                "--ysize=300",
                "--unicharset_file=", os.path.join(self.path_to_self, 'data/Latin.unicharset'),
            ]

            args.append("--rotate_image={}".format("true" if use_rotation else "false"))
            args.append("--distort_image={}".format("true" if use_distortion else "false"))

            subprocess.run(args)

        create_image_from_text(iter=iteration)

    def process_chunk(self, i, chunk_size):
        start = i * chunk_size
        end = start + chunk_size
        for j in range(start, end):
            self.create_truth_i(j)

            # if j > 10:
            #     exit(0)

    def perform_training(self):
        # Clear out existing files in the ground truth folder
        for the_file in os.listdir(self.ground_truth_output_folder):
            file_path = os.path.join(self.ground_truth_output_folder, the_file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        number_of_chunks = min(self.number_of_truths_to_create, 9)
        chunk_size = self.number_of_truths_to_create // number_of_chunks

        # Chunk into 10 groups
        print(f"Will split into {number_of_chunks} jobs, each having {chunk_size} items")
        processes = []
        for i in range(number_of_chunks):
            processes = []
            p = Process(target=self.process_chunk, args=(i, chunk_size,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # # Clean up - remove any numbered *.gt.txt file that does not have both and box file
        # all_gt_txt_files = [f for f in os.listdir(self.ground_truth_output_folder) if f.endswith('.gt.txt')]
        #
        # for gt_txt_file in all_gt_txt_files:
        #     # files have names like genecafe_1.gt.txt, we want just the number from that filename
        #     number = gt_txt_file.split('_')[1].split('.')[0]
        #     # number = gt_txt_file.split('_')[1]
        #     box_file = os.path.join(self.ground_truth_output_folder, f"genecafe_{number}.box")
        #     if not os.path.exists(box_file):
        #         self.remove_data_for(gt_txt_file, box_file)
        #     else:
        #         # If the box file is empty, also remove it and associated
        #         with open(box_file, 'r') as f:
        #             contents = f.read()
        #             if len(contents.strip()) == 0:
        #                 self.remove_data_for(gt_txt_file, box_file)


if __name__ == '__main__':
    trainer = GroundTruthCreation()
    trainer.perform_training()
