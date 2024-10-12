import PIL
import numpy as np
from io import BytesIO
from fastapi import UploadFile


def from_upload_file_to_np_array(
    file: UploadFile,
    max_height_for_compress: int | None = None,
):
    """
    This function reads the file from the FastAPI UploadFile object
    and returns a numpy array representing the image.

    If max_height_for_compress is provided, the image will be resized
    with a maximum height of max_height_for_compress before being
    converted to a numpy array. This can significantly reduce the time
    it takes to process the image with face_recognition.
    """
    b = BytesIO(file.file.read())
    image = PIL.Image.open(b).convert("RGB")

    if max_height_for_compress:
        image = compress_image(image, max_height_for_compress)

    return np.array(image)


def from_upload_file_to_pil_image(file: UploadFile):
    """
    This function reads the file from the FastAPI UploadFile object
    and returns a PIL Image object.
    """
    b = BytesIO(file.file.read())
    return PIL.Image.open(b).convert("RGB")


def compress_image(image, max_height=1000):
    """
    This function resizes the image to a maximum height of max_height
    Compressing a image before processing it with face_recognition can
    significantly reduce the time it takes to process the image.

    """

    if image.height > max_height:
        image = image.resize(
            (image.width * max_height // image.height, max_height)
        )
    return image
