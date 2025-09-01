import os
import uvicorn
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
from pathlib import Path

# Assuming your pipeline is in src/pipeline/face_swap_video_pipeline.py
from src.pipeline.face_swap_video_pipeline import FaceSwapPipeline  
from src.loggings.logger import logger
from src.exceptions.exception import VideoProcessingException

app = FastAPI(
    title="Face Swapper API",
    description="An API to detect faces in a video and swap them with a source face.",
    version="1.0.0"
)

# Directory to store uploaded videos and outputs
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)



@app.post("/detect_faces/")
async def detect_faces_in_video(video: UploadFile = File(...)):
    """
    **Uploads a video and detects unique faces.**

    This endpoint takes a video file, saves it, and then runs the face detection
    and clustering pipeline. It returns information about the unique faces found
    in the video, allowing the user to select which face to swap in a subsequent
    request.

    - **Args**:
        - `video` (UploadFile): The video file to process.

    - **Returns**:
        - A JSON object with a status and a list of `detected_faces`. Each detected face includes its
          `index`, and an identifier for the representative face.
    """
    try:
        video_path = UPLOAD_DIR / video.filename
        
        # Save the uploaded video
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Initialize and run the detection pipeline
        pipeline = FaceSwapPipeline(video_path=str(video_path))
        
        # Run face detection
        if not pipeline.detect_faces():
            return JSONResponse(
                status_code=404,
                content={"message": "No valid faces were detected in the video."}
            )

        # Prepare the response data
        response_faces = []
        for label, faces in sorted(pipeline.clusters.items()):
            # Find the best representative face
            best_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
            
            # Ensure bbox is a Python list of integers
            bbox = [int(val) for val in best_face['bbox']]
            
            response_faces.append({
                "index": int(label),
                "representative_face_bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3]
                }
            })

        logger.info(f"Detected faces in video '{video.filename}'. Ready for swap.")
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Faces detected. Use the index to swap.",
                "video_path": str(video_path),
                "clusters": pipeline.clusters,
                "detected_faces": response_faces
            }
        )
    except VideoProcessingException as e:
        logger.error(f"Error during face detection: {e.args[0]}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred in face detection: {e.args[0]}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {e}"
        )

@app.post("/swap_faces/")
async def swap_faces_in_video(
    source_image: UploadFile = File(...),
    face_index: int = Form(...),
    video_path: str = Form(...),
    clusters: str = Form(...),
):
    """
    **Swaps faces in the previously uploaded video.**

    This endpoint takes a `source_image` and a `face_index` to perform the face-swapping operation
    on the video last uploaded via the /detect_faces/ endpoint. It returns the final output video.

    - **Args**:
        - `source_image` (UploadFile): The image containing the face to swap in.
        - `face_index` (int): The index of the person to swap. Use the indices
          from a previous detection run, or -1 for all.
        - `video_path` (str): The path to the video file.
        - `clusters` (str): A JSON string of the clusters data.

    - **Returns**:
        - The swapped video file.
    """
    try:
        # Save the uploaded source image to a temporary path
        source_image_path = UPLOAD_DIR / source_image.filename
        with source_image_path.open("wb") as buffer:
            shutil.copyfileobj(source_image.file, buffer)
        
        # Deserialize the clusters data
        import json
        clusters_dict = json.loads(clusters)

        # Initialize the pipeline
        pipeline = FaceSwapPipeline(video_path=video_path)
        
        # Call the swapping method from the pipeline
        swapping_artifact = pipeline.swap_faces(str(source_image_path), face_index, clusters_dict, pipeline.detection_artifact)
        
        if not swapping_artifact or not hasattr(swapping_artifact, 'final_output_video_path'):
            raise HTTPException(
                status_code=500,
                detail="Face swapping failed to generate an output video."
            )

        output_video_path = swapping_artifact.final_output_video_path

        logger.info(f"Face swapping complete. Output video at: {output_video_path}")
        return FileResponse(
            path=output_video_path,
            filename=Path(output_video_path).name,
            media_type="video/mp4"
        )

    except VideoProcessingException as e:
        logger.error(f"Error during face swapping: {e.args[0]}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during face swapping: {e.args[0]}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {e}"
        )

# Main entry point to run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)