from PIL import Image


class ImageStorage:
    def save_image(self, image: Image, filename):
        raise NotImplementedError

    def get_image(self, filename):
        raise NotImplementedError


class LocalImageStorage(ImageStorage):
    def __init__(self, storage_path: str):
        import os
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        if storage_path[-1] != "/":
            storage_path += "/"
        self.storage_path = storage_path

    def save_image(self, image: Image, filename):
        image.save(self.storage_path + filename + ".jpg")

    def get_image(self, filename):
        return open(self.storage_path + filename + ".jpg", "rb").read()
