import os
import gdown

def load_gdrive_file(file_id,
                     ending='',
                     output_folder=os.path.expanduser('~/.keras/datasets')):
  """Downloads files from google drive, caches files that are already downloaded."""
  filename = '{}.{}'.format(file_id, ending) if ending else file_id
  filename = os.path.join(output_folder, filename)
  if not os.path.exists(filename):
    print("Downloading model to ~/.keras/datasets")
    gdown.download('https://drive.google.com/uc?id={}'.format(file_id),
                   filename,
                   quiet=False)
  else:
      print("File already exists!")
  return filename

if __name__ == "__main__":
    load_gdrive_file("1Wu1p2U7BgK8TwzvTdLlBWCemo3oPN-ad.h5")