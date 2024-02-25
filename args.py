import argparse

parser = argparse.ArgumentParser(
    description='Extract images and audio from a roast, and produce something useful for AI training')
parser.add_argument('input_spec', type=str, help='The input specification')
parser.add_argument('-s', "--start", type=int, help='The starting frame number', required=False)
parser.add_argument('-e', "--end", type=int, help='The ending frame number', required=False)
parser.add_argument("-t", "--test", help="Run in test mode", default=False, action="store_true")
args = parser.parse_args()

# Check that required args are given
if not args.input_spec:
    print('Error: input_spec is required')
    exit(1)

