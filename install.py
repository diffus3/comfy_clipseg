import os
import sys
import shutil
#import argparse

PATH = os.path.dirname(os.path.realpath(__file__))
PATH_EXTRAS = os.path.join('..', '..', 'comfy_extras', 'clipseg')
PATH_WEIGHTS = os.path.join(PATH_EXTRAS, "weights")
PATH_MODELS = os.path.join(PATH_EXTRAS, "models")

print(PATH_WEIGHTS)
print(PATH_MODELS)

print("Installing...")
#os.system(f'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"')
os.system(f'pip install -r "{PATH}/requirements.txt"')
os.system(f'pip install git+https://github.com/openai/CLIP.git')

weights_file_path = f'{PATH}/weights.zip'
if (not os.path.exists(weights_file_path)):
    #os.remove(weights_file_path)
    os.system(f'wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip')
    os.system(f'unzip -d weights -j weights.zip')
else:
    print(f'File exists: {weights_file_path}. Skipping download.')

#print(PATH_EXTRAS)
#print(PATH_WEIGHTS)
#print(PATH_MODELS)

#os.makedirs(PATH_EXTRAS, exist_ok=True)
#os.makedirs(PATH_WEIGHTS, exist_ok=True)
os.makedirs(PATH_MODELS, exist_ok=True)
#shutil.copytree('weights', PATH_WEIGHTS)
shutil.copytree('models', PATH_MODELS)

#wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
#unzip -d weights -j weights.zip
