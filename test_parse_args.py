import argparse
parser = argparse.ArgumentParser()
parser.description='please enter two parameters a and b ...'
parser.add_argument("-i", "--inputA", help="set input file", dest="input", type=str, default="video1.mp4")
args = parser.parse_args()

print("intput file: ",args.input)
