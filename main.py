from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil

from video_utils import extract_frames
from qdrant_utils import (
    create_collection,
    upload_frame_vectors,
    search_similar_frames
)

app = FastAPI()

UPLOAD_DIR = "uploads"
FRAME_DIR = "frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_name = os.path.splitext(file.filename)[0]
    frame_output_dir = os.path.join(FRAME_DIR, video_name)
    frame_data = extract_frames(file_location, frame_output_dir)

    create_collection()
    upload_frame_vectors(frame_output_dir, frame_data)

    return {
        "message": f"Video '{file.filename}' uploaded, {len(frame_data)} frames extracted + vectors uploaded.",
        "frame_folder": frame_output_dir
    }

@app.post("/search/")
async def search_similar_images(file: UploadFile = File(...)):
    query_image_path = os.path.join("query_temp.jpg")
    
    try:
        with open(query_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"[DEBUG] Query image saved to: {query_image_path}")

        results = search_similar_frames(query_image_path)

        print(f"[DEBUG] Search returned {len(results)} results")
        
        for r in results:
            r["image_url"] = f"/frames/{r['filename']}"

        return {
            "query": file.filename,
            "results": results
        }

    except Exception as e:
        print(f"[ERROR] Search failed: {str(e)}")
        return {"error": f"Search failed: {str(e)}"}

app.mount("/frames", StaticFiles(directory="frames"), name="frames")
