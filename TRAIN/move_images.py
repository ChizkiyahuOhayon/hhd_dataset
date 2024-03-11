import os
import shutil
target_directory = "Train"
if not os.path.exists(target_directory):
    os.makedirs(target_directory)
directory_list = [str(i) for i in range(27)]

counter = 0
for directory in directory_list:
    for image_name in os.listdir(directory):
        source_file = os.path.join(directory, image_name)
        target_file = os.path.join(target_directory, image_name)
        shutil.copy(source_file, target_file)
        counter += 1

print(f"{counter} images have been copied to {target_directory}")