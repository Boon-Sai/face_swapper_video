import os
import sys
import json
from collections import Counter

from src.components.face_detection import DetectFaces
from src.loggings.logger import logger
from src.exceptions.exception import FaceDetectionException


class FaceSwapPipeline:
    def __init__(self, video_path: str, image_path: str):
        """
        Initialize FaceSwapPipeline with paths.
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            self.video_path = video_path
            self.image_path = image_path
            self.output_folder = os.path.join("artifacts", "detected_faces")

            os.makedirs(self.output_folder, exist_ok=True)

            logger.info(f"Pipeline initialized with video: {video_path}, image: {image_path}")

        except Exception as e:
            logger.error(f"Error initializing FaceSwapPipeline: {str(e)}")
            raise FaceDetectionException(str(e), sys)

    def run(self):
        """
        Run the entire pipeline: detect faces, store results, and summarize.
        """
        try:
            logger.info("Pipeline started...")

            # Step 1: Detect faces
            detector = DetectFaces(self.video_path, self.image_path, self.output_folder)
            faces_info = detector.process()

            # Step 2: Store metadata in JSON
            metadata_path = os.path.join(self.output_folder, "faces_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(faces_info, f, indent=4)

            logger.info(f"Pipeline finished successfully. Metadata stored at {metadata_path}")
            return faces_info

        except FaceDetectionException as fde:
            logger.error(f"Face detection failed: {str(fde)}")
            raise fde

        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {str(e)}")
            raise FaceDetectionException(str(e), sys)
