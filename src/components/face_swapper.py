import os
import sys
import ffmpeg
import cv2
import numpy as np
from pathlib import Path
from IPython.display import clear_output
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from src.exceptions.exception import FaceDetectionException
from src.loggings.logger import logger
from src.entity.config_entity import (
    FaceSwappingConfig,
    ConfigEntity,
    FaceDetectionConfig
)
from src.entity.artifact_entity import (
    FaceSwappingArtifact,
    FaceDetectionArtifact
)


class SwapFaces:
    def __init__(self, index: int, video_path: str, source_face_path: str, clusters: list):
        """
        Initialize SwapFaces operation
        Args:
            index (int): -1 means all faces, else swap only selected cluster
            video_path (str): Path to input video
            source_face_path (str): Path to source face image
            clusters (list): List of clustered embeddings
        """
        try:
            logger.log("Initializing Face Swap Operation...")

            # Face Swapper model
            self.swap_face_config = FaceSwappingConfig(config=ConfigEntity)
            self.face_swap_model = get_model(
                self.swap_face_config.face_swapper_model,
                download=False,
                download_zip=False
            )
            logger.log("Face swap model initialized successfully")

            # Face Detection model
            self.detection_config = FaceDetectionConfig(config=ConfigEntity())
            self.face_detection_model = FaceAnalysis(
                name=self.detection_config.face_detection_model
            )
            self.face_detection_model.prepare(
                ctx_id=self.detection_config.ctx_id,
                det_size=self.detection_config.det_size
            )
            logger.log("Face detection model initialized successfully")

            # Instance attributes
            self.video_path = video_path
            self.index = index
            self.source_face_path = source_face_path
            self.clusters = clusters

            # Load source face
            source_img = cv2.imread(self.source_face_path)
            if source_img is None:
                raise FaceDetectionException("Source face image not found!", sys)
            self.source_face = self.face_detection_model.get(source_img)[0]

        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def swap_faces(self) -> str:
        """
        Perform face swapping on video
        Returns:
            str: path to swapped video
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_video_path = f"swapped_{Path(self.video_path).stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            processed_frames = 0
            frame_count = 0
            sampling_interval = 10

            # Precompute target embeddings if specific cluster chosen
            target_embeddings = None
            if self.index != -1:
                target_embeddings = [f['embedding'] for f in self.clusters]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sampling_interval != 0:
                    frame_count += 1
                    continue

                detected_faces = self.face_detection_model.get(frame)

                if detected_faces:
                    for face in detected_faces:
                        if self.index == -1:
                            # Swap all detected faces
                            frame = self.face_swap_model.get(frame, face, self.source_face, paste_back=True)
                        else:
                            # Swap only chosen cluster faces
                            new_embedding = face['embedding']
                            sims = [
                                np.dot(new_embedding, t) /
                                (np.linalg.norm(new_embedding) * np.linalg.norm(t))
                                for t in target_embeddings
                            ]
                            if max(sims) > 0.65:
                                frame = self.face_swap_model.get(frame, face, self.source_face, paste_back=True)

                out.write(frame)
                processed_frames += 1

                if processed_frames % 10 == 0:
                    clear_output(wait=True)
                    progress_percent = (processed_frames / total_frames) * 100
                    logger.log(f"Processed {processed_frames}/{total_frames} frames ({progress_percent:.1f}%)")

                frame_count += 1

            cap.release()
            out.release()
            clear_output(wait=True)

            logger.info("Processing Complete!")
            logger.info(f"Input video: {Path(self.video_path).name}")
            logger.info(f"Output video: {output_video_path}")
            logger.info(f"Face swapped: {'ALL' if self.index == -1 else f'person {self.index}'}")
            logger.info(f"Total frames processed: {processed_frames}")

            return output_video_path

        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def insert_audio(self, output_video_path: str) -> str:
        """
        Add original audio back to swapped video
        Returns:
            str: final output video path
        """
        try:
            input_video = ffmpeg.input(output_video_path)
            input_audio = ffmpeg.input(FaceDetectionArtifact.extracted_audio_path)

            output_filename = os.path.splitext(output_video_path)[0] + "_with_audio.mp4"
            (
                ffmpeg
                .output(input_video.video, input_audio.audio, output_filename, vcodec='copy', acodec='aac')
                .run(overwrite_output=True)
            )

            logger.info(f"Video with new audio created successfully at: {output_filename}")
            return output_filename

        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def video_preprocessing(self):
        """
        Future step: Apply preprocessing before face swap
        Example: stabilization, resizing, cropping etc.
        """
        try:
            logger.log("Video preprocessing step invoked... (placeholder)")
            # output_video_path = self.swap_faces()
            return True
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e
