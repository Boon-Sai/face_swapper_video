import os
import uuid
import shutil
import base64
import time
import traceback
import asyncio
import numpy as np

from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Query,
    BackgroundTasks,
    Request,
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError

# Import pipeline and swapper (ensure no circular top-level imports in components)
from src.pipeline.face_swap_video_pipeline import FaceSwapPipeline
from src.components.face_swapper import SwapFaces

from src.loggings.logger import logger
try:
    from src.exceptions.exception import VideoProcessingException
except Exception:
    VideoProcessingException = Exception

# -------------------------
# Config & directories
# -------------------------
DATA_ROOT = "data"                     # per-session temp dirs created under data/{session_id}
DOWNLOADS_DIR = "downloads"            # final outputs moved here and served via /download/
ARTIFACTS_DIR = "artifacts"            # cleared after successful swap (per your request)
UPLOADS_DIR = "uploads"                # cleared after successful swap (per your request)

for d in (DATA_ROOT, DOWNLOADS_DIR, ARTIFACTS_DIR, UPLOADS_DIR):
    os.makedirs(d, exist_ok=True)

# Cleaner config
RETENTION_SECONDS = 60 * 60        # 1 hour
CLEAN_INTERVAL_SECONDS = 60 * 5    # every 5 minutes

executor = ThreadPoolExecutor(max_workers=4)

# In-memory sessions
sessions: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="Face Swap API",
    description="Upload a video + source image, detect faces and swap them. Produces a downloadable output and cleans artifacts/uploads.",
    version="1.1.0",
)

# -------------------------
# Exception handlers
# -------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, VideoProcessingException):
        logger.error("Video processing exception", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "video_processing_error", "detail": str(exc)})
    logger.error("Unhandled exception", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Request validation error", exc_info=True)
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": exc.errors()})


# -------------------------
# Helper functions
# -------------------------
def make_session_dir(session_id: str) -> str:
    path = os.path.join(DATA_ROOT, session_id)
    os.makedirs(path, exist_ok=True)
    return path


async def save_upload_file(upload: UploadFile, dest_path: str) -> None:
    contents = await upload.read()
    with open(dest_path, "wb") as f:
        f.write(contents)
    await upload.seek(0)


def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_video_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def sanitize_clusters(raw_items: List[Any]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict) and "embedding" in item:
            cleaned.append(item)
        elif isinstance(item, (list, tuple, np.ndarray)):
            cleaned.append({"embedding": np.array(item)})
        else:
            logger.warning(f"Skipping cluster item of unsupported type: {type(item)}")
    return cleaned


def cleanup_session_dir(session_id: str) -> None:
    meta = sessions.pop(session_id, None)
    if not meta:
        return
    temp_dir = meta.get("temp_dir")
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Removed session dir: {temp_dir} for session {session_id}")
    except Exception:
        logger.exception(f"Error removing session dir for {session_id}")


def empty_directory(dir_path: str) -> None:
    """Remove all files and subdirectories inside dir_path (but not dir_path itself)."""
    if not os.path.exists(dir_path):
        return
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Removed {path} from {dir_path}")
        except Exception:
            logger.exception(f"Failed to remove {path} while emptying {dir_path}")


def move_to_downloads(src_path: str, session_id: str) -> str:
    """Move the produced file to downloads/ and return the new absolute path."""
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    ext = os.path.splitext(src_path)[1] or ".mp4"
    dest_name = f"{session_id}{ext}"
    dest_path = os.path.join(DOWNLOADS_DIR, dest_name)
    shutil.move(src_path, dest_path)
    logger.info(f"Moved output {src_path} -> {dest_path}")
    return dest_path


def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, lambda: func(*args, **kwargs))


# -------------------------
# Background cleaner
# -------------------------
async def cleaner_loop(stop_event: asyncio.Event):
    logger.info("Session cleaner started")
    try:
        while not stop_event.is_set():
            await asyncio.sleep(CLEAN_INTERVAL_SECONDS)
            now = time.time()
            for sid, meta in list(sessions.items()):
                temp_dir = meta.get("temp_dir")
                if not temp_dir:
                    continue
                try:
                    mtime = os.path.getmtime(temp_dir)
                    if now - mtime > RETENTION_SECONDS:
                        logger.info(f"Cleaner removing stale session {sid}")
                        cleanup_session_dir(sid)
                except FileNotFoundError:
                    sessions.pop(sid, None)
                except Exception:
                    logger.exception(f"Cleaner error for session {sid}")
    finally:
        logger.info("Session cleaner stopped")


@app.on_event("startup")
async def startup_event():
    app.state.cleaner_stop = asyncio.Event()
    app.state.cleaner_task = asyncio.create_task(cleaner_loop(app.state.cleaner_stop))
    logger.info("App startup complete; cleaner scheduled")


@app.on_event("shutdown")
async def shutdown_event():
    try:
        if hasattr(app.state, "cleaner_stop"):
            app.state.cleaner_stop.set()
        if hasattr(app.state, "cleaner_task"):
            await asyncio.wait_for(app.state.cleaner_task, timeout=10)
    except asyncio.TimeoutError:
        logger.warning("Cleaner task did not stop in time; cancelling")
        app.state.cleaner_task.cancel()
    except Exception:
        logger.exception("Error stopping cleaner task")

    try:
        executor.shutdown(wait=True)
    except Exception:
        logger.exception("Error shutting down executor")

    for sid in list(sessions.keys()):
        try:
            cleanup_session_dir(sid)
        except Exception:
            pass


# -------------------------
# Endpoints
# -------------------------
@app.post("/upload-files/")
async def upload_files(video: UploadFile = File(...), source_face_image: UploadFile = File(...)):
    """
    Save uploaded video + source image, run detection (in threadpool), return:
      { "session_id": "...", "message": "...", "detected_faces": [...] }
    detected_faces entries: { index, path, base64 }
    """
    session_id = uuid.uuid4().hex
    temp_dir = make_session_dir(session_id)

    try:
        # Save uploads
        video_path = os.path.join(temp_dir, video.filename)
        source_path = os.path.join(temp_dir, source_face_image.filename)
        await save_upload_file(video, video_path)
        await save_upload_file(source_face_image, source_path)

        # Run pipeline detection in threadpool
        def create_and_detect():
            pipeline = FaceSwapPipeline(video_path=video_path)
            detected = pipeline.detect_faces()
            return pipeline, detected

        pipeline, detected = await run_blocking(create_and_detect)

        if not detected:
            sessions[session_id] = {
                "pipeline": pipeline,
                "video_path": video_path,
                "source_path": source_path,
                "detected_faces": [],
                "sorted_labels": [],
                "temp_dir": temp_dir,
            }
            logger.info(f"Session {session_id}: no faces detected")
            return JSONResponse(
                content={
                    "session_id": session_id,
                    "message": "No faces detected in the video. Please try a different video.",
                    "detected_faces": [],
                    "video_path": video_path,
                    "source_path": source_path,
                },
                status_code=200,
            )

        # Build detected_faces list
        detected_dir = pipeline.detection_artifact.detected_faces_path
        sorted_labels = sorted(pipeline.clusters.keys())
        detected_faces: List[Dict[str, Any]] = []
        for idx, label in enumerate(sorted_labels):
            face_path = os.path.join(detected_dir, f"face_{label}.jpg")
            if os.path.exists(face_path):
                try:
                    b64 = image_to_base64(face_path)
                except Exception:
                    b64 = None
                    logger.exception(f"Failed to base64-encode {face_path}")
                detected_faces.append(
                    {
                        "index": idx + 1,
                        "path": face_path,
                        "base64": f"data:image/jpeg;base64,{b64}" if b64 else None,
                    }
                )

        sessions[session_id] = {
            "pipeline": pipeline,
            "video_path": video_path,
            "source_path": source_path,
            "detected_faces": detected_faces,
            "sorted_labels": sorted_labels,
            "temp_dir": temp_dir,
        }

        logger.info(f"Session {session_id} created, detected {len(detected_faces)} faces")
        return JSONResponse(
            content={
                "session_id": session_id,
                "message": f"Detected {len(detected_faces)} faces. Use this session_id with /swap-faces/",
                "detected_faces": detected_faces,
                "video_path": video_path,
                "source_path": source_path,
            },
            status_code=200,
        )

    except Exception as e:
        logger.exception("Error in upload-files")
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/swap-faces/")
async def swap_faces(
    background_tasks: BackgroundTasks,
    session_id: str = Query(..., description="Session ID returned by /upload-files/"),
    index: int = Query(..., description="1-based index of detected face to swap, or -1 for all"),
):
    """
    Perform face swapping for a given session_id and index (-1 => all).
    After producing the final output file, move it into downloads/ and clear artifacts & uploads.
    Returns JSON with session_id and download_url.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Invalid session_id. Please upload files first.")

    meta = sessions[session_id]
    pipeline: FaceSwapPipeline = meta["pipeline"]
    sorted_labels: List[str] = meta.get("sorted_labels", [])
    video_path: str = meta["video_path"]
    source_path: str = meta["source_path"]
    temp_dir: str = meta["temp_dir"]

    # Build raw cluster list
    if index == -1:
        raw_items = []
        for label in sorted_labels:
            raw_items.extend(pipeline.clusters.get(label, []))
        chosen_index = -1
    else:
        if index < 1 or index > len(meta.get("detected_faces", [])):
            raise HTTPException(
                status_code=400,
                detail=f"Index out of range. Valid range: 1 to {len(meta.get('detected_faces', []))}",
            )
        label = sorted_labels[index - 1]
        raw_items = pipeline.clusters.get(label, [])
        chosen_index = 0

    # Sanitize clusters
    clusters = sanitize_clusters(raw_items)
    if not clusters:
        raise HTTPException(status_code=500, detail="No valid cluster embeddings found for swapping.")

    # Inject audio path if present
    try:
        from src.entity.artifact_entity import FaceDetectionArtifact
        FaceDetectionArtifact.extracted_audio_path = getattr(
            pipeline.detection_artifact, "extracted_audio_path", None
        )
    except Exception:
        logger.exception("Could not set FaceDetectionArtifact.extracted_audio_path (non-fatal)")

    # Blocking swap operation
    def do_swap():
        sf = SwapFaces(index=chosen_index, video_path=video_path, source_face_path=source_path, clusters=clusters)
        return sf.video_preprocessing()

    try:
        artifact = await run_blocking(do_swap)

        output_video_path = getattr(artifact, "final_output_video_path", None)
        if not output_video_path or not os.path.exists(output_video_path):
            raise HTTPException(status_code=500, detail="Face swap failed. Output video not generated.")

        # Move final output to downloads/
        download_path = move_to_downloads(output_video_path, session_id)
        sessions[session_id]["final_output_path"] = download_path

        # Now that output is safely moved to downloads, empty artifacts & uploads directories
        try:
            empty_directory(ARTIFACTS_DIR)
            empty_directory(UPLOADS_DIR)
        except Exception:
            logger.exception("Failed to empty artifacts/uploads (non-fatal)")

        # Cleanup session temp dir (background)
        background_tasks.add_task(cleanup_session_dir, session_id)

        # Provide a download URL endpoint to the client
        download_url = f"/download/{session_id}"

        logger.info(f"Face swap completed for session {session_id}, download at {download_url}")

        # Optionally include base64 (be careful: large). Here we do not include to keep response small.
        return JSONResponse(
            content={
                "session_id": session_id,
                "message": "Face swap completed successfully.",
                "download_url": download_url,
                "download_path": download_path,
            },
            status_code=200,
        )

    except VideoProcessingException as e:
        logger.error("Video processing error during swap", exc_info=True)
        background_tasks.add_task(cleanup_session_dir, session_id)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception("Error during face swap")
        background_tasks.add_task(cleanup_session_dir, session_id)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.get("/download/{session_id}")
async def download_file(session_id: str, background_tasks: BackgroundTasks):
    """
    Serve the moved download file for the session_id.
    Schedule deletion of the download file after serving (optional).
    """
    if session_id not in sessions:
        # But the download file may exist even if session entry was removed — check downloads folder
        filename_candidates = [f for f in os.listdir(DOWNLOADS_DIR) if f.startswith(session_id)]
        if not filename_candidates:
            raise HTTPException(status_code=404, detail="Download not found")
        download_path = os.path.join(DOWNLOADS_DIR, filename_candidates[0])
        # if file exists, serve it (no session metadata)
    else:
        download_path = sessions[session_id].get("final_output_path")
        if not download_path or not os.path.exists(download_path):
            # fallback: look for file in downloads dir
            filename_candidates = [f for f in os.listdir(DOWNLOADS_DIR) if f.startswith(session_id)]
            if not filename_candidates:
                raise HTTPException(status_code=404, detail="Download not found")
            download_path = os.path.join(DOWNLOADS_DIR, filename_candidates[0])

    if not os.path.exists(download_path):
        raise HTTPException(status_code=404, detail="Download file not found")

    # Schedule deletion of the download file after it is served (adjust delay if you want)
    def _delete_file(path: str):
        try:
            # small delay so client has time to start downloading
            time.sleep(5)
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted download file: {path}")
        except Exception:
            logger.exception(f"Failed to delete download file: {path}")

    background_tasks.add_task(_delete_file, download_path)

    # If a session exists, remove its metadata (we already moved file to downloads)
    try:
        sessions.pop(session_id, None)
    except Exception:
        pass

    return FileResponse(download_path, media_type="video/mp4", filename=os.path.basename(download_path))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "active_sessions": len(sessions)}


# -------------------------
# Run (for development only; use uvicorn in production)
# -------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=False)
