# WidthHeightRatio: 1.6050749039215912 | Pitch: 0.22355608196898985

import sys
import csv

text_file_name = sys.argv[1]
csv_file_name = sys.argv[2]

with open(str(csv_file_name), mode='w') as csv_file, open (str(text_file_name), mode='r') as text_file:
    test_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    lines = text_file.readlines()
    for line in lines:
        widthHeightRatio = line.split("|")[0].split(":")[1]
        pitch = line.split("|")[1].split(":")[1]
        test_writer.writerow([widthHeightRatio, pitch])
