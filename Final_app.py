from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from src.pipeline.face_swap_video_pipeline import FaceSwapPipeline
from src.components.face_swapper import SwapFaces
from src.loggings.logger import logger
import os
import base64
import sys
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

app = FastAPI()

# AWS S3 Configuration
BUCKET_NAME = "your-face-swap-bucket"  # Replace with your S3 bucket name
S3_REGION = "us-east-1"  # Replace with your AWS region if different
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Global variable to store the latest session data
latest_session_data = {}

def get_s3_client():
    """Initialize and return S3 client."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        raise ValueError("AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION
    )

def upload_to_s3(local_file_path, bucket, object_name):
    """Upload file to S3 bucket and return the public URL (assuming bucket policy allows public read)."""
    try:
        s3_client = get_s3_client()
        s3_client.upload_file(local_file_path, bucket, object_name)
        # Assuming the bucket has public read access enabled
        s3_url = f"https://{bucket}.s3.amazonaws.com/{object_name}"
        logger.info(f"File uploaded to S3: {s3_url}")
        return s3_url
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
        raise HTTPException(status_code=500, detail="AWS credentials not configured properly.")
    except ClientError as e:
        logger.error(f"Error uploading to S3: {e}")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/upload-files/")
async def upload_files(video: UploadFile = File(...), source_face_image: UploadFile = File(...)):
    try:
        global latest_session_data
        
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        video_path = os.path.join(data_dir, video.filename)
        with open(video_path, "wb") as f:
            f.write(await video.read())
        
        # Source face image is mandatory
        source_path = os.path.join(data_dir, source_face_image.filename)
        with open(source_path, "wb") as f:
            f.write(await source_face_image.read())
        
        pipeline = FaceSwapPipeline(video_path=video_path)
        if not pipeline.detect_faces():
            latest_session_data = {
                "pipeline": pipeline,
                "video_path": video_path,
                "source_path": source_path,
                "detected_faces": []
            }
            logger.info(f"No faces detected. Session data: {latest_session_data.keys()}")
            return {
                "message": "No faces detected in the video. Please try a different video.",
                "detected_faces": [],
                "video_path": video_path,
                "source_path": source_path
            }
        
        # Extract detected faces for response
        detected_dir = pipeline.detection_artifact.detected_faces_path
        sorted_labels = sorted(pipeline.clusters.keys())
        detected_faces = []
        for idx, label in enumerate(sorted_labels):
            path = os.path.join(detected_dir, f"face_{label}.jpg")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    img_bytes = f.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                detected_faces.append({
                    "index": idx + 1,
                    "base64": f"data:image/jpeg;base64,{img_base64}",
                    "path": path
                })
        
        latest_session_data = {
            "pipeline": pipeline,
            "video_path": video_path,
            "source_path": source_path,
            "detected_faces": detected_faces,
            "sorted_labels": sorted_labels
        }
        
        logger.info(f"Session data updated: {latest_session_data.keys()}")
        logger.debug(f"Clusters in session data: {pipeline.clusters}")
        return {
            "message": f"Detected {len(detected_faces)} faces. Please select a face to swap by index (1 to {len(detected_faces)} or -1 for all) in the /swap-faces/ endpoint.",
            "detected_faces": detected_faces,
            "video_path": video_path,
            "source_path": source_path
        }
    except Exception as e:
        logger.error(f"Error in file upload endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/swap-faces/")
async def swap_faces(indices: int = Query(..., description="Index of the face to swap (e.g., 1) or -1 for all faces")):
    try:
        logger.info(f"Received swap-faces request with indices={indices}")
        logger.debug(f"Current session data: {latest_session_data}")
        
        if not latest_session_data:
            logger.error("No session data available. Please upload files first.")
            raise HTTPException(status_code=404, detail="No session data available. Please upload files first.")
        
        pipeline = latest_session_data["pipeline"]
        sorted_labels = latest_session_data["sorted_labels"]
        video_path = latest_session_data["video_path"]
        detected_faces = latest_session_data["detected_faces"]
        source_path = latest_session_data["source_path"]
        
        logger.info(f"Session data: video_path={video_path}, detected_faces_count={len(detected_faces)}, sorted_labels={sorted_labels}, source_path={source_path}")
        
        # Validate indices
        if indices == -1:
            swap_clusters = [face for label in sorted_labels for face in pipeline.clusters[label]]
            index = -1
        else:
            try:
                if indices < 1 or indices > len(detected_faces):
                    raise ValueError(f"Index out of range. Valid range: 1 to {len(detected_faces)}")
                label = sorted_labels[indices - 1]
                swap_clusters = pipeline.clusters[label]
                index = 0  # Dummy index for single face
            except ValueError as e:
                logger.warning(f"Invalid index input: {indices}. Error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid index: {str(e)}")
        
        logger.info(f"Swap clusters length: {len(swap_clusters)}, index: {index}")
        
        if not swap_clusters:
            logger.error("No valid clusters selected for face swapping.")
            raise HTTPException(status_code=400, detail="No valid clusters selected for face swapping.")
        
        logger.debug(f"Swap clusters sample: {swap_clusters[:1] if swap_clusters else 'empty'}")
        
        # Set static extracted_audio_path
        from src.entity.artifact_entity import FaceDetectionArtifact
        FaceDetectionArtifact.extracted_audio_path = pipeline.detection_artifact.extracted_audio_path
        
        # Perform swap
        sf = SwapFaces(index=index, video_path=video_path, source_face_path=source_path, clusters=swap_clusters)
        artifact = sf.video_preprocessing()
        
        if not artifact.final_output_video_path or not os.path.exists(artifact.final_output_video_path):
            logger.error(f"Face swap failed. Result video not found at: {artifact.final_output_video_path}")
            raise HTTPException(status_code=500, detail="Face swap failed. Result video not generated.")
        
        # Generate S3 object name
        object_name = f"swapped_videos/{os.path.basename(artifact.final_output_video_path)}"
        
        # Upload to S3
        s3_url = upload_to_s3(artifact.final_output_video_path, BUCKET_NAME, object_name)
        
        logger.info(f"Face swap completed and uploaded to S3. URL: {s3_url}")
        return JSONResponse(content={
            "message": "Face swap completed successfully and uploaded to S3.",
            "output_video_path": artifact.final_output_video_path,
            "s3_url": s3_url
        })
    
    except Exception as e:
        logger.error(f"Error in face swap endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)