from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import os
import uuid
from features import compute_color_histogram

qdrant_client = QdrantClient(
    url="https://YOUR_QDRANT_CLUSTER_URL_HERE",
    api_key="YOUR_API_KEY_HERE",
)

def create_collection(collection_name="video-frames", vector_size=512):
    if collection_name not in [col.name for col in qdrant_client.get_collections().collections]:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"✅ Collection '{collection_name}' created.")
    else:
        print(f"✅ Collection '{collection_name}' already exists.")

def upload_frame_vectors(frame_folder: str, frame_data: list, collection_name="video-frames"):
    points = []
    for filename, timestamp in frame_data:
        image_path = os.path.join(frame_folder, filename)
        vector = compute_color_histogram(image_path)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "filename": filename,
                "timestamp": round(timestamp, 2)
            }
        )
        points.append(point)

    qdrant_client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Uploaded {len(points)} frames to Qdrant.")

def search_similar_frames(query_image_path: str, collection_name="video-frames", top_k=5):
    query_vector = compute_color_histogram(query_image_path)

    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k
    )

    return [
        {
            "score": round(point.score, 4),
            "filename": point.payload.get("filename", "unknown.jpg"),
            "timestamp": point.payload.get("timestamp", "N/A")  
        }
        for point in search_result
    ]
