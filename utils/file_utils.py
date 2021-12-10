import os


def create_path(path: str, *args, filename: str = None):

    path = os.path.join(path, *args)
    if not os.path.exists(path):
        os.makedirs(path)

    if filename is not None:
        path = os.path.join(path, filename)

    return path
