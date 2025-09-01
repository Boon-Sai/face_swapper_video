import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from insightface.utils import face_align

from src.loggings.logger import logger
from src.components.face_detection import DetectFaces
from src.components.face_swapper import SwapFaces
from src.entity.artifact_entity import FaceDetectionArtifact

class FaceSwapPipeline:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.detection_artifact = None
        self.clusters = None
        self.face_detector = None

    def detect_faces(self):
        """Detects faces in the video and clusters them."""
        logger.info("Starting face detection and clustering...")
        self.face_detector = DetectFaces(video_path=self.video_path)
        self.detection_artifact, self.clusters = self.face_detector.video_preprocessing()

        if not self.clusters:
            logger.warning("No valid clusters found in the video.")
            return False
        
        logger.info(f"Found {len(self.clusters)} unique persons.")
        return True

    def swap_faces(self, source_face_path: str, index: int, clusters: dict, detection_artifact: FaceDetectionArtifact):
        """Swaps faces in the video based on user selection."""
        logger.info(f"Preparing to swap faces for index {index}...")

        if index == -1:
            swap_clusters = [face for cluster_faces in clusters.values() for face in cluster_faces]
        else:
            swap_clusters = clusters.get(index, [])

        if not swap_clusters:
            logger.warning(f"No faces found for index {index}. Nothing to swap.")
            return

        FaceDetectionArtifact.extracted_audio_path = detection_artifact.extracted_audio_path

        sf = SwapFaces(index=index, video_path=self.video_path, source_face_path=source_face_path, clusters=swap_clusters)
        swapping_artifact = sf.video_preprocessing()
        
        return swapping_artifact

# if __name__ == "__main__":
#     # This main function is for testing the pipeline class directly if needed.
#     # The main application entry point will be in src/main.py
#     video_path = "path/to/your/video.mp4" # Example path
#     pipeline = FaceSwapPipeline(video_path=video_path)
#     pipeline.run_full_pipeline()
