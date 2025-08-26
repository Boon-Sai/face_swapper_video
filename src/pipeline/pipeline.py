import os
import sys
import json
from collections import Counter

from src.components.face_detection import DetectFaces
from src.components.face_swapper import SwapFaces
from src.exceptions.exception import FaceDetectionException
from src.loggings.logger import logger

class FaceSwapPipeline:
    def __init__(self, video_path: str, image_path: str):
        self.video_path = video_path
        self.image_path = image_path

    def get_target_cluster_id(self, json_path: str, user_cluster_id: int = None) -> int:
        """
        Determines the cluster_id to swap. If user provides an ID, uses it; otherwise, finds the most common cluster_id (ignoring noise).
        """
        try:
            logger.info("Determining target cluster ID for face swapping...")
            with open(json_path, 'r') as f:
                face_data = json.load(f)

            cluster_ids = []
            for frame_path in face_data:
                for face_info in face_data[frame_path]:
                    if 'cluster_id' in face_info and face_info['cluster_id'] != -1:
                        cluster_ids.append(face_info['cluster_id'])

            if not cluster_ids:
                raise Exception("No suitable face clusters found for swapping. Could not identify a main person.")

            if user_cluster_id is not None and user_cluster_id in cluster_ids:
                logger.info(f"Using user-specified cluster ID: {user_cluster_id}")
                return user_cluster_id

            most_common_cluster = Counter(cluster_ids).most_common(1)[0][0]
            logger.info(f"Identified most prominent person with cluster ID: {most_common_cluster} (default selection)")
            return most_common_cluster

        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def run(self, user_cluster_id: int = None):
        try:
            logger.info("Starting Face Swap Pipeline...")

            # Step 1: Detect faces, cluster them, and extract audio
            face_detector = DetectFaces(video_path=self.video_path)
            detection_artifact = face_detector.initiate_face_detection()

            if detection_artifact.json_information is None or not os.path.exists(detection_artifact.json_information):
                logger.info("Face detection did not produce a valid JSON output. Aborting pipeline.")
                return None

            # Step 2: Determine which person to swap (allow user input)
            target_cluster_id = self.get_target_cluster_id(detection_artifact.json_information, user_cluster_id)

            # Step 3: Swap the faces (pass original video_path for FPS retrieval)
            face_swapper = SwapFaces(
                source_image_path=self.image_path,
                detection_artifact=detection_artifact,
                cluster_id_to_swap=target_cluster_id
            )
            swapping_artifact = face_swapper.initiate_face_swapping(original_video_path=self.video_path)

            logger.info("Face Swap Pipeline completed successfully!")
            logger.info(f"Final video saved at: {swapping_artifact.final_output_video_path}")

            return swapping_artifact.final_output_video_path

        except Exception as e:
            logger.error(f"An error occurred during the pipeline execution: {e}")
            raise FaceDetectionException(str(e), sys) from e