from qdrant_client import AsyncQdrantClient, models
from typing import List, Tuple, Optional
from face_recognition_api.settings import get_settings


class VectorDB:
    def __init__(self):
        settings = get_settings()

        if (settings.QDRANT_MODE == "local"):
            self.qdrant_client = AsyncQdrantClient(path=settings.QDRANT_PATH)
        elif (settings.QDRANT_MODE == "remote"):
            print("Connecting to Qdrant remote")
            print(f"Host: {settings.QDRANT_HOST}")
            print(f"Port: {settings.QDRANT_PORT}")
            print(f"GRPC Port: {settings.QDRANT_GRPC_PORT}")
            print(f"Prefer GRPC: {settings.QDRANT_PREFER_GRPC}")
            self.qdrant_client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                https=settings.QDRANT_HTTPS,
                port=settings.QDRANT_PORT,
                grpc_port=settings.QDRANT_GRPC_PORT,
                prefer_grpc=settings.QDRANT_PREFER_GRPC,
                api_key=settings.QDRANT_API_KEY,
            )

        self.collection = settings.QDRANT_COLLECTION

    @classmethod
    async def create(cls):
        self = cls()

        # Create collection if it doesn't exist
        collection_exists = await self.qdrant_client.collection_exists(
            self.collection
        )
        if not collection_exists:
            await self.qdrant_client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=128, distance=models.Distance.EUCLID,
                ),
            )
            await self.qdrant_client.create_payload_index(
                self.collection, field_name="pid", field_schema='keyword',
            )

        return self

    async def upsert(self, id, vector, metadata):
        await self.qdrant_client.upsert(
            collection_name=self.collection,
            points=[
                models.PointStruct(
                    id=id,
                    vector=vector,
                    payload=metadata,
                )
            ],
        )

    async def update_payload(self, id, payload):
        """
        Update payload by id (UUID)
        """
        await self.qdrant_client.set_payload(
            collection_name=self.collection,
            points=[id],
            payload=payload,
        )

    async def query(self, vector, top_k=5):
        """
        Query the collection by vector
        """
        result = await self.qdrant_client.query_points(
            self.collection, vector, limit=top_k, with_vectors=True
        )
        return result.points

    async def query_by_pid(
            self,
            pid,
            limit=10,
            offset=0
    ) -> Tuple[List[models.Record], Optional[str]]:
        """
        Get points by pid (Person ID)
        """
        result = await self.qdrant_client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="pid",
                    match=models.MatchValue(value=pid)
                )]
            ),
            limit=limit,
            offset=offset,
        )
        return result
