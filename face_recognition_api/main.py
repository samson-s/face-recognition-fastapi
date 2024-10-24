import face_recognition
import uuid
from fastapi import FastAPI, UploadFile, Request, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional
from .vector_db import VectorDB
from .image_storage import LocalImageStorage, ImageStorage
from .image_processing import\
    from_upload_file_to_pil_image, \
    compress_image, \
    convert_locations_before_compress, \
    convert_locations_after_compress
import numpy as np


@asynccontextmanager
async def lifespan(app: FastAPI):
    vector_db = await VectorDB.create()
    app.state.vector_db = vector_db

    image_storage = LocalImageStorage(storage_path="images")
    app.state.image_storage = image_storage
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/find_faces")
async def find_faces(file: UploadFile):
    image = from_upload_file_to_pil_image(file)
    original_height = image.height

    image = compress_image(image, max_height=1000)
    imageArray = np.array(image)

    face_locations = face_recognition.face_locations(imageArray)
    face_locations = convert_locations_before_compress(
        face_locations, original_height, image.height
    )

    return {"face_locations": face_locations}


@app.post("/recognize_faces")
async def recognize_faces(request: Request, file: UploadFile):
    image = from_upload_file_to_pil_image(file)
    original_height = image.height

    image = compress_image(image, max_height=1000)
    imageArray = np.array(image)

    face_locations = face_recognition.face_locations(imageArray)
    original_face_locations = convert_locations_before_compress(
        face_locations, original_height, image.height
    )

    face_encodings = face_recognition.face_encodings(
        imageArray,
        known_face_locations=face_locations,
        num_jitters=1 if len(face_locations) > 5 else 10,
        model="large"
    )

    vector_db: VectorDB = request.app.state.vector_db

    results = []
    for i in range(len(original_face_locations)):
        compare_result = False

        found_faces = await vector_db.query(face_encodings[i], top_k=1)
        if found_faces:
            found_face = found_faces[0]
            # face_recognition.compare_faces takes a list of face encodings
            # and compares them to a known face encoding.
            # It returns a list of True/False values indicating whether or
            # not the known face matched the input face
            compare_result = face_recognition.compare_faces(
                [found_face.vector],
                face_encodings[i],
                0.4
            )[0]

        result = {
            'name': 'Unknown',
            'location': original_face_locations[i],
        }
        if compare_result:
            result['id'] = found_face.id
            result['name'] = found_face.payload['name']
            result['metadata'] = found_face.payload

        results.append(result)

    return results


@app.post("/add_face")
async def add_face(
    request: Request,
    file: UploadFile,
    name: str,
    location: str
):
    image = from_upload_file_to_pil_image(file)
    original_height = image.height

    image = compress_image(image, max_height=1000)
    image_array = np.array(image)

    # Assume and only one face is allowed
    if location:
        location = list(map(int, location.split(",")))
    else:
        location = face_recognition.face_locations(image_array)[0]

    location = convert_locations_after_compress(
        [location], original_height, image.height)[0]

    face_encodings = face_recognition.face_encodings(
        image_array,
        known_face_locations=[location],
        num_jitters=20,
        model="large"
    )
    face_encoding = face_encodings[0]

    vector_db: VectorDB = request.app.state.vector_db

    id = str(uuid.uuid4())
    payload = {"name": name}
    await vector_db.upsert(
        id, face_encoding.tolist(),
        payload
    )

    image_storage: ImageStorage = request.app.state.image_storage
    faceImage = image.crop((
        location[3], location[0], location[1], location[2]
    ))
    image_storage.save_image(faceImage, id)

    return {"id": id}


@app.get("/get_face_image")
async def get_face_image(request: Request, id: str):
    image_storage: ImageStorage = request.app.state.image_storage
    image = image_storage.get_image(id)
    return Response(content=image, media_type="image/jpg")


class UpdateFaceMetadataBody(BaseModel):
    id: str
    metadata: dict


@app.post("/update_face_metadata")
async def update_face_metadata(request: Request, body: UpdateFaceMetadataBody):
    vector_db: VectorDB = request.app.state.vector_db

    await vector_db.update_payload(
        body.id,
        body.metadata
    )

    return {"id": body.id}


class PointResponse(BaseModel):
    id: str
    payload: dict


class GetPointByPidResponse(BaseModel):
    points: list[PointResponse]
    next_offset: Optional[str] = Field(
        None, description="Next offset point's id"
    )


@app.get("/get_point_by_pid")
async def get_point_by_pid(
    request: Request,
    pid: str,
    limit: int = 10,
    offset: str | None = None
):
    vector_db: VectorDB = request.app.state.vector_db

    result = await vector_db.query_by_pid(pid, limit=limit, offset=offset)

    response = GetPointByPidResponse(
        points=[
            PointResponse(id=record.id, payload=record.payload)
            for record in result[0]
        ],
        next_offset=result[1]  # Next offset point's id
    )

    return response
