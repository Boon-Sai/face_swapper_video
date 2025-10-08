from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from src.pipeline.face_swap_video_pipeline import FaceSwapPipeline
from src.components.face_swapper import SwapFaces
from src.loggings.logger import logger
import os
import base64
import sys

app = FastAPI()

# Global variable to store the latest session data
latest_session_data = {}

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
        selected_labels = []
        if indices == -1:
            selected_labels = sorted_labels
            index = -1
        else:
            try:
                if indices < 1 or indices > len(detected_faces):
                    raise ValueError(f"Index out of range. Valid range: 1 to {len(detected_faces)}")
                selected_labels = [sorted_labels[indices - 1]]
                index = 0  # Dummy index for single face
            except ValueError as e:
                logger.warning(f"Invalid index input: {indices}. Error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid index: {str(e)}")
        
        logger.info(f"Selected labels: {selected_labels}, index: {index}")
        
        # Compute swap_clusters
        swap_clusters = {label: pipeline.clusters[label] for label in selected_labels}
        if not swap_clusters:
            logger.error("No valid clusters selected for face swapping.")
            raise HTTPException(status_code=400, detail="No valid clusters selected for face swapping.")
        
        logger.debug(f"Swap clusters: {swap_clusters}")
        
        # Set static extracted_audio_path
        from src.entity.artifact_entity import FaceDetectionArtifact
        FaceDetectionArtifact.extracted_audio_path = pipeline.detection_artifact.extracted_audio_path
        
        # Perform swap
        sf = SwapFaces(index=index, video_path=video_path, source_face_path=source_path, clusters=swap_clusters)
        artifact = sf.video_preprocessing()
        
        if not artifact.final_output_video_path or not os.path.exists(artifact.final_output_video_path):
            logger.error(f"Face swap failed. Result video not found at: {artifact.final_output_video_path}")
            raise HTTPException(status_code=500, detail="Face swap failed. Result video not generated.")
        
        # Convert result video to base64 for response
        with open(artifact.final_output_video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")
        
        logger.info(f"Face swap completed. Output video: {artifact.final_output_video_path}")
        return JSONResponse(content={
            "message": "Face swap completed successfully.",
            "output_video_path": artifact.final_output_video_path,
            "base64": f"data:video/mp4;base64,{video_base64}"
        })
    
    except Exception as e:
        logger.error(f"Error in face swap endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True)