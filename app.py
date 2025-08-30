import os
import sys
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any
import logging
from urllib.parse import quote
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assuming these are available in your environment
from src.components.face_detection import DetectFaces
from src.components.face_swapper import SwapFaces
from src.exceptions.exception import FaceDetectionException

app = FastAPI()

# Global variable for caching processed video data (consider using a proper cache in production)
VIDEO_CACHE: Dict[str, Any] = {
    "video_path": None,
    "clusters": None,
    "detected_faces_dir": None
}

# Artifacts directory
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
logger.debug(f"Artifacts directory initialized: {ARTIFACTS_DIR}")

class FaceDetectionRequest(BaseModel):
    video_path: str

class FaceSwapperRequest(BaseModel):
    source_image_path: str
    index: int = -1

@app.post("/face-detection")
async def face_detection(request: FaceDetectionRequest):
    """
    Performs face detection on a video and caches the results.
    Returns a list of detected faces as base64-encoded images and valid cluster indices.
    """
    try:
        logger.info(f"Starting face detection for video: {request.video_path}")

        # Validate video path
        logger.debug(f"Validating video path: {request.video_path}")
        if not os.path.exists(request.video_path):
            logger.error(f"Video file not found: {request.video_path}")
            raise HTTPException(status_code=400, detail=f"Video file not found: {request.video_path}")

        # Check if the video has already been processed
        if VIDEO_CACHE["video_path"] == request.video_path and VIDEO_CACHE["clusters"] is not None:
            logger.debug(f"Using cached data for video: {request.video_path}")
            detected_faces_dir = VIDEO_CACHE["detected_faces_dir"]
            clusters = VIDEO_CACHE["clusters"]
        else:
            logger.debug(f"Processing new video: {request.video_path}")
            detector = DetectFaces(video_path=request.video_path)
            artifacts, clusters = detector.video_preprocessing()
            
            # Normalize cluster keys to strings
            normalized_clusters = {str(key): value for key, value in clusters.items()}
            logger.debug(f"Raw clusters keys: {list(clusters.keys())}, Normalized clusters keys: {list(normalized_clusters.keys())}")
            
            # Cache the results
            VIDEO_CACHE["video_path"] = request.video_path
            VIDEO_CACHE["clusters"] = normalized_clusters
            VIDEO_CACHE["detected_faces_dir"] = artifacts.detected_faces_path
            detected_faces_dir = artifacts.detected_faces_path

        if not VIDEO_CACHE["clusters"]:
            logger.warning("No faces detected in the video")
            raise FaceDetectionException("No faces detected in the video.", sys)
        
        logger.info(f"Found {len(VIDEO_CACHE['clusters'])} unique face clusters with keys: {list(VIDEO_CACHE['clusters'].keys())}")
        
        face_files = sorted([f for f in os.listdir(detected_faces_dir) if f.endswith(".jpg")])
        logger.debug(f"Found {len(face_files)} face images in {detected_faces_dir}")
        
        response_data = {
            "total_faces": len(face_files),
            "faces": [],
            "valid_indices": list(VIDEO_CACHE["clusters"].keys())
        }
        
        for i, face_file in enumerate(face_files):
            face_path = os.path.join(detected_faces_dir, face_file)
            logger.debug(f"Encoding face image: {face_path}")
            with open(face_path, "rb") as f:
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
            response_data["faces"].append({"index": i, "base64": encoded_string})
        
        logger.info(f"Face detection completed successfully. Returning {len(face_files)} faces with valid indices: {response_data['valid_indices']}")
        return response_data

    except FaceDetectionException as e:
        logger.error(f"FaceDetectionException: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File not found: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in face_detection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.post("/face-swapper")
async def face_swapper(request: FaceSwapperRequest):
    """
    Performs face swapping using the cached video data.
    Requires a successful call to /face-detection first.
    Returns a JSON response with the download link for the final video.
    """
    try:
        logger.info(f"Starting face swapping for source image: {request.source_image_path}, index: {request.index}")

        # Validate source image path
        logger.debug(f"Validating source image path: {request.source_image_path}")
        if not os.path.exists(request.source_image_path):
            logger.error(f"Source image not found: {request.source_image_path}")
            raise HTTPException(status_code=400, detail=f"Source image not found: {request.source_image_path}")

        # Ensure a video has been processed and cached
        logger.debug("Checking VIDEO_CACHE")
        if VIDEO_CACHE["video_path"] is None or VIDEO_CACHE["clusters"] is None:
            logger.error("No video data found in cache")
            raise FaceDetectionException("No video data found. Please call /face-detection first.", sys)

        video_path = VIDEO_CACHE["video_path"]
        clusters = VIDEO_CACHE["clusters"]
        
        # Log clusters for debugging
        logger.debug(f"Clusters type: {type(clusters)}, keys: {list(clusters.keys())}, length: {len(clusters)}")
        
        # Ensure clusters is a dictionary
        if not isinstance(clusters, dict):
            logger.error(f"Unexpected clusters format; expected a dictionary, got {type(clusters)}")
            raise HTTPException(status_code=500, detail="Unexpected clusters format; expected a dictionary.")
        
        # Validate index against actual cluster keys
        if request.index != -1:
            cluster_key = str(request.index)
            if cluster_key not in clusters:
                logger.error(f"Invalid index: {request.index}. Available cluster keys: {list(clusters.keys())}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid index: {request.index}. Available cluster keys: {list(clusters.keys())}"
                )

        # Perform face swapping
        logger.debug("Initializing SwapFaces")
        swapper = SwapFaces(
            index=request.index,
            video_path=video_path,
            source_face_path=request.source_image_path,
            clusters=clusters,  # Pass the normalized dictionary
            artifacts_dir=ARTIFACTS_DIR
        )
        
        logger.debug("Running swap_faces")
        output_video_path = swapper.swap_faces()
        logger.debug(f"Swap faces completed. Output video: {output_video_path}")
        
        logger.debug("Inserting audio")
        final_video_path = swapper.insert_audio(output_video_path)
        logger.info(f"Face swapping completed. Final video with audio: {final_video_path}")

        # Generate download link
        filename = os.path.basename(final_video_path)
        download_link = f"http://localhost:8000/artifacts/{quote(filename)}"
        logger.info(f"Returning download link: {download_link}")

        return {
            "message": "Face swapping completed successfully",
            "download_link": download_link,
            "filename": filename
        }

    except FaceDetectionException as e:
        logger.error(f"FaceDetectionException: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: {str(e)}")
        raise HTTPException(status_code=400, detail=f"File not found: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in face_swapper: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/artifacts/{filename}")
async def download_artifact(filename: str):
    """
    Serves files from the artifacts directory.
    """
    try:
        logger.info(f"Attempting to serve file: {filename}")
        file_path = os.path.join(ARTIFACTS_DIR, filename)
        logger.debug(f"File path: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")
        logger.info(f"Serving file: {file_path}")
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting FastAPI application on port 8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)