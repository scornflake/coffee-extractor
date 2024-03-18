# Extract a series of roasts, to the output directory

import os
import argparse

parser = argparse.ArgumentParser(description='Run many extractions')
parser.add_argument('-s', "--start", type=int, help='The starting roast number', required=True)
parser.add_argument('-e', "--end", type=int, help='The ending roast number', required=True)
parser.add_argument("--temps", action=argparse.BooleanOptionalAction)
parser.add_argument("--audio", action=argparse.BooleanOptionalAction)
parser.add_argument("--video", action=argparse.BooleanOptionalAction)
parser.add_argument('--a1', type=int, help='Preprocess audio v1', action=argparse.BooleanOptionalAction)
parser.add_argument("input_data_folder", type=str, help='The input data folder, where the roast folders are located')
args = parser.parse_args()

# We must have start and end arguments specified
if not args.start:
    print('Error: start is required')
    exit(1)
if not args.end:
    print('Error: end is required')
    exit(1)

# --audio --video /input-data-folder/1/input.spec.json

print("Extracting roasts from", args.start, "to", args.end)
input_folder = args.input_data_folder
print("Input data folder is", input_folder)

# tessdata_local_path is at ../../data/resseract/tessdata
tessdata_local_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tesseract/tessdata")
print("should be in : ", tessdata_local_path)
os.environ["TESSDATA_PREFIX"] = tessdata_local_path

# Run each extraction, in turn
for i in range(args.start, args.end + 1):
    print("Extracting roast", i)
    specification = f"{args.input_data_folder}/{i}/input.spec.json"

    # the spec must exist
    if not os.path.exists(specification):
        print(f"Error: {specification} does not exist")
        exit(1)

    os.system(
        f"python3 extractor.py {'--temps' if args.temps else ''} {'--a1' if args.a1 else ''} {'--audio' if args.audio else ''} {'--video' if args.video else ''} {specification}")
