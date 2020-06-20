import glob
import fileinput

read_file = glob.glob("*.txt")

with open("result.csv", "w") as outfile:
	for f in read_file:
		with open(f,"r") as infile:
			outfile.write(infile.read())