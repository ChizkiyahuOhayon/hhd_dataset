import os
import shutil

directory_list = [str(i) for i in range(27)]
destination_directory = 'Test'

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

for directory in directory_list:
    for image_name in os.listdir(directory):  # open as a list
        source_file = os.path.join(directory, image_name)
        target_file = os.path.join(destination_directory, image_name)
        shutil.move(source_file, target_file)

print(f"All the images have been moved to {destination_directory}")