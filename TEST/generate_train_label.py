import os
import csv
directory_list = [str(i) for i in range(27)]
output_file = 'training_labels'
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['image', 'label'])  # the header row

    for directory in directory_list:
        for filename in os.listdir(directory):
            image = filename
            label = directory
            writer.writerow([image, label])

print("csv file created successfully")

