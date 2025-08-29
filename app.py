
import os
import sys
import base64
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from src.components.face_detection import DetectFaces
from src.components.face_swapper import SwapFaces
from src.exceptions.exception import FaceDetectionException

app = FastAPI()

class FaceDetectionRequest(BaseModel):
    video_path: str

class FaceSwapperRequest(BaseModel):
    source_image_path: str
    video_path: str
    index: int = -1

@app.post("/face-detection")
async def face_detection(request: FaceDetectionRequest):
    try:
        detector = DetectFaces(video_path=request.video_path)
        artifacts, _ = detector.video_preprocessing()
        
        detected_faces_dir = artifacts.detected_faces_path
        face_files = [f for f in os.listdir(detected_faces_dir) if f.endswith(".jpg")]

        response_data = {"total_faces": len(face_files), "faces": []}
        
        for i, face_file in enumerate(face_files):
            face_path = os.path.join(detected_faces_dir, face_file)
            with open(face_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
            response_data["faces"].append({"index": i, "base64": encoded_string})
            
        return response_data

    except FaceDetectionException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/face-swapper")
async def face_swapper(request: FaceSwapperRequest):
    try:
        # Perform face detection to get clusters
        detector = DetectFaces(video_path=request.video_path)
        _, clusters = detector.video_preprocessing()

        if clusters is None:
            raise FaceDetectionException("No faces detected in video.", sys)

        # Perform face swapping
        swapper = SwapFaces(
            index=request.index,
            video_path=request.video_path,
            source_face_path=request.source_image_path,
            clusters=list(clusters.values())
        )
        
        output_video_path = swapper.swap_faces()
        final_video_path = swapper.insert_audio(output_video_path)

        return FileResponse(final_video_path, media_type="video/mp4", filename=os.path.basename(final_video_path))

    except FaceDetectionException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
