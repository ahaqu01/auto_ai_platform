import uuid

from App.settings import UPLOADS_DIR, FILE_PATH_PREFIX


def filename_transfer(sub_dir, filename):
    ext_name = filename.rsplit(".")[1]

    print(ext_name)

    new_filename = uuid.uuid4().hex + "." + ext_name

    save_path = UPLOADS_DIR + sub_dir  + new_filename

    upload_path = FILE_PATH_PREFIX + sub_dir + new_filename

    return save_path, upload_path


