import os
import random
import shutil

def del_files(dir, per):
  for root, dirs, files in os.walk(dir):
    num_files_to_delete = int(len(files) * per)
    if num_files_to_delete > 0:
      files_to_delete = random.sample(files, num_files_to_delete)
      for file_name in files_to_delete:
        file_path = os.path.join(root, file_name)
        try:
          if os.path.isfile(file_path):
            os.remove(file_path)
          else:
            shutil.rmtree(file_path)
          print(f"Xóa: {file_path}")
        except Exception as e:
          print(f"Xóa lỗi {file_path}: {e}")

dts_path = "submain"
del_files(dts_path, 0.8)
