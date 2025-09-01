from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from src.pipeline.face_swap_video_pipeline import FaceSwapPipeline
from src.loggings.logger import logger
from src.exceptions.exception import VideoProcessingException
import os
import base64
import json
from pathlib import Path

app = FastAPI(
    title="Video Face Swapper API",
    description="An API to detect faces in a video and swap them with a source face.",
    version="1.0.0"
)

# Directories for uploads and outputs
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Global variable to store the latest session data
latest_session_data = {}

@app.post("/upload-video/")
async def upload_video(video: UploadFile = File(...), source_image: UploadFile = File(...)):
    """
    Uploads a video and source image, detects faces in the video, and prepares for face swapping.
    
    Args:
        video (UploadFile): The video file to process.
        source_image (UploadFile): The image containing the face to swap in.
    
    Returns:
        JSONResponse: Contains detected face information, video path, source image path, and a message.
    """
    try:
        global latest_session_data

        # Validate file extensions
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .avi, .mov")
        if not source_image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Invalid image format. Supported: .jpg, .jpeg, .png")

        # Save video and source image
        video_path = UPLOAD_DIR / video.filename
        source_image_path = UPLOAD_DIR / source_image.filename

        with video_path.open("wb") as buffer:
            buffer.write(await video.read())
        with source_image_path.open("wb") as buffer:
            buffer.write(await source_image.read())

        # Initialize pipeline and detect faces
        logger.info(f"Initializing FaceSwapPipeline with video: {video_path}")
        pipeline = FaceSwapPipeline(video_path=str(video_path))
        if not pipeline.detect_faces():
            latest_session_data = {
                "video_path": str(video_path),
                "source_image_path": str(source_image_path),
                "detected_faces": [],
                "clusters": {}
            }
            return JSONResponse(
                status_code=404,
                content={
                    "message": "No faces detected in the video. Please try a different video.",
                    "detected_faces": [],
                    "video_path": str(video_path),
                    "source_image_path": str(source_image_path)
                }
            )

        # Prepare detected faces response
        detected_faces = []
        for label, faces in sorted(pipeline.clusters.items()):
            best_face = max(faces, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
            bbox = [int(val) for val in best_face['bbox']]
            # Convert representative image to base64 (if available)
            base64_face = ""
            face_image_path = best_face.get('image_path')
            if face_image_path and os.path.exists(face_image_path):
                try:
                    with open(face_image_path, "rb") as img_file:
                        base64_face = base64.b64encode(img_file.read()).decode("utf-8")
                except Exception as e:
                    logger.warning(f"Failed to encode face image {face_image_path}: {e}")

            detected_faces.append({
                "index": int(label) + 1,  # 1-based indexing for user
                "base64": f"data:image/jpeg;base64,{base64_face}" if base64_face else None,
                "bbox": {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}
            })

        # Store session data
        latest_session_data = {
            "video_path": str(video_path),
            "source_image_path": str(source_image_path),
            "detected_faces": detected_faces,
            "clusters": pipeline.clusters,
            "detection_artifact": pipeline.detection_artifact
        }

        logger.info(f"Detected {len(detected_faces)} faces in video '{video.filename}'.")
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Detected {len(detected_faces)} faces. Select a face to swap by index (e.g., 1 or -1 for all) in the /swap-faces/ endpoint.",
                "detected_faces": detected_faces,
                "video_path": str(video_path),
                "source_image_path": str(source_image_path)
            }
        )

    except VideoProcessingException as e:
        logger.error(f"Error in video upload endpoint: {e.args[0]}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video Processing Error: {e.args[0]}")
    except Exception as e:
        logger.error(f"Unexpected error in video upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/swap-faces/")
async def swap_faces(indices: int = Form(...)):  # Changed from str to int
    """
    Swaps faces in the previously uploaded video based on selected index.
    
    Args:
        indices (int): Single index of face to swap (e.g., 1) or -1 for all faces.
    
    Returns:
        FileResponse: The swapped video file.
    """
    try:
        global latest_session_data
        if not latest_session_data:
            logger.error("No session data available. Please upload video and source image first.")
            raise HTTPException(status_code=404, detail="No session data available. Please upload video and source image first.")

        # Validate session data
        required_keys = ["video_path", "source_image_path", "detected_faces", "clusters", "detection_artifact"]
        missing_keys = [key for key in required_keys if key not in latest_session_data]
        if missing_keys:
            logger.error(f"Missing session data keys: {missing_keys}")
            raise HTTPException(status_code=500, detail=f"Invalid session data: missing {missing_keys}")

        video_path = latest_session_data["video_path"]
        source_image_path = latest_session_data["source_image_path"]
        detected_faces = latest_session_data["detected_faces"]
        clusters = latest_session_data["clusters"]
        detection_artifact = latest_session_data["detection_artifact"]

        # Validate file existence
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
        if not os.path.exists(source_image_path):
            logger.error(f"Source image not found: {source_image_path}")
            raise HTTPException(status_code=404, detail=f"Source image not found: {source_image_path}")

        # Validate indices
        logger.info(f"Received indices: {indices}")
        face_index = indices
        if face_index != -1:  # Check for valid single index
            face_index = face_index - 1  # Convert to 0-based
            if face_index < 0 or face_index >= len(detected_faces):
                logger.warning(f"Invalid index: {indices}. Valid range: 1 to {len(detected_faces)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid index: {indices}. Valid range: 1 to {len(detected_faces)}"
                )

        logger.info(f"Parsed face_index: {face_index}")

        # Initialize pipeline and perform face swap
        logger.info(f"Initializing FaceSwapPipeline with video: {video_path}")
        pipeline = FaceSwapPipeline(video_path=video_path)
        logger.info(f"Calling swap_faces with source_image_path: {source_image_path}, face_index: {face_index}, clusters: {clusters}")
        swapping_artifact = pipeline.swap_faces(
            str(source_image_path),  # Positional argument
            face_index,  # Single integer or -1
            clusters,
            detection_artifact
        )

        if not swapping_artifact or not hasattr(swapping_artifact, 'final_output_video_path') or not os.path.exists(swapping_artifact.final_output_video_path):
            logger.error(f"Face swap failed. Output video not found at: {swapping_artifact.final_output_video_path if swapping_artifact else 'None'}")
            raise HTTPException(status_code=500, detail="Face swap failed. Output video not generated.")

        output_video_path = swapping_artifact.final_output_video_path
        logger.info(f"Face swapping complete. Output video at: {output_video_path}")

        return FileResponse(
            path=output_video_path,
            filename=Path(output_video_path).name,
            media_type="video/mp4"
        )

    except VideoProcessingException as e:
        logger.error(f"Error in face swap endpoint: {e.args[0]}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video Processing Error: {e.args[0]}")
    except Exception as e:
        logger.error(f"Unexpected error in face swap: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)