import os
import csv

directory_list = [str(i) for i in range(27)]

csv_file_address = 'training_labels'
with open(csv_file_address, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image', 'Label'])  # header

    counter = 0
    for directory in directory_list:
        for image_name in os.listdir(directory):
            if image_name.endswith('.png'):
                image = image_name
                label = directory
                writer.writerow([image, label])
                counter += 1

print(f"{counter} training labels have been created as {csv_file_address}")