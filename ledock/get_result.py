import os
import shutil
import os

parent_folder = '/home/b519/wq/rlg_1/ledock/ligands/'

# List all items in the parent folder
items = os.listdir(parent_folder)

for item in items:
    # Create the full path to the item
    src_dir = os.path.join(parent_folder, item)
    dst_dir = '/home/b519/wq/rlg_1/ledock/final_result/folder/'
    new_name=src_dir.split('/home/b519/wq/rlg_1/ledock/ligands/')[1].split('/')[0]+'.mol2'
    # Get a list of all subdirectories in the source directory
    subdirs = [f.path for f in os.scandir(src_dir) if f.is_dir()]

    # Sort the list of subdirectories based on their modification time
    subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Find the latest .mol file in the latest subdirectory
    mol_file = None
    for f in os.listdir(subdirs[0]):
        if f.endswith('.mol2'):
            mol_file=f
            print(mol_file)
            os.rename(subdirs[0]+"/"+mol_file, subdirs[0]+"/"+new_name)
            destination_file = dst_dir + new_name
            shutil.copyfile(subdirs[0]+"/"+new_name, destination_file)

    # Move the latest .mol file to the destination directory
    # if mol_file is not None:
    #     shutil.move(src_dir+mol_file, dst_dir)
