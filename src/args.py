import argparse

parser = argparse.ArgumentParser(
    description='Extract images and audio from a roast, and produce something useful for AI training')
parser.add_argument('input_spec', type=str, help='The input specification')
parser.add_argument('-s', "--start", type=int, help='The starting frame number', required=False)
parser.add_argument('-e', "--end", type=int, help='The ending frame number', required=False)
parser.add_argument("--tesseract", action=argparse.BooleanOptionalAction, help='Save images to tesseract')
parser.add_argument("--temps", action=argparse.BooleanOptionalAction)
parser.add_argument("--audio", action=argparse.BooleanOptionalAction)
parser.add_argument("--video", action=argparse.BooleanOptionalAction)
parser.add_argument('--a1', type=int, help='Preprocess audio v1', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# Check that required args are given
if not args.input_spec:
    print('Error: input_spec is required')
    exit(1)
