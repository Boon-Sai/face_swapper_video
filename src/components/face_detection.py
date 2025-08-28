import os
import sys
import json
import cv2
import numpy as np
import ffmpeg
from pathlib import Path
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
from insightface.utils import face_align
from IPython.display import clear_output

from src.exceptions.exception import FaceDetectionException
from src.loggings.logger import logger

from src.entity.config_entity import (
    ConfigEntity,
    FaceDetectionConfig
)

from src.entity.artifact_entity import (
    FaceDetectionArtifact
)


class DetectFaces:
    def __init__(self, video_path: str):
        try:
            logger.info("Initializing DetectFaces component...")

            self.detection_config = FaceDetectionConfig(config=ConfigEntity())
            self.face_detection_model = FaceAnalysis(
                name=self.detection_config.face_detection_model
            )
            self.face_detection_model.prepare(
                ctx_id=self.detection_config.ctx_id,
                det_size=self.detection_config.det_size
            )

            self.video_path = video_path
            self.video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Paths inside artifacts/face_detection
            self.audio_path = os.path.join(
                self.detection_config.face_detection_folder_path, "extracted_audio.mp3"
            )
            self.detected_faces_dir = os.path.join(
                self.detection_config.face_detection_folder_path, "detected_faces"
            )
            os.makedirs(self.detected_faces_dir, exist_ok=True)
            self.clusters_json = os.path.join(
                self.detection_config.face_detection_folder_path, "clusters.json"
            )

            logger.info("DetectFaces initialized successfully.")

        except Exception as e:
            logger.error("Error initializing DetectFaces", exc_info=True)
            raise FaceDetectionException(str(e), sys) from e

    def extract_audio(self) -> str | None:
        try:
            logger.info("Starting audio extraction...")

            if not Path(self.video_path).exists():
                logger.warning("Input video not found.")
                return None

            probe = ffmpeg.probe(self.video_path)
            has_audio = any(
                stream["codec_type"] == "audio" for stream in probe.get("streams", [])
            )
            if not has_audio:
                logger.warning("No audio stream found. Skipping extraction.")
                return None

            (
                ffmpeg.input(self.video_path)
                .output(self.audio_path, acodec="mp3")
                .run(overwrite_output=True)
            )

            logger.info(f"Audio extracted successfully: {self.audio_path}")
            return self.audio_path

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else "No stderr output"
            logger.error(f"FFmpeg error during audio extraction: {error_msg}")
            raise FaceDetectionException(f"Error extracting audio: {error_msg}", sys) from e

        except Exception as e:
            logger.error("Unexpected error during audio extraction", exc_info=True)
            raise FaceDetectionException(str(e), sys) from e

    def detecting_and_clustering_faces(self) -> dict | None:
        try:
            logger.info("Starting face detection and clustering...")
            clusters = {}
            cap = cv2.VideoCapture(self.video_path)
            all_faces, all_embeddings = [], []
            frame_count, sampling_interval = 0, 10

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sampling_interval != 0:
                    frame_count += 1
                    continue

                detected_faces = self.face_detection_model.get(frame)
                for face in detected_faces:
                    face["frame_idx"] = frame_count
                    all_faces.append(face)
                    all_embeddings.append(face["embedding"])

                frame_count += 1
                if frame_count % 50 == 0:
                    clear_output(wait=True)
                    logger.info(
                        f"Processed {frame_count} frames, collected {len(all_faces)} faces..."
                    )

            cap.release()

            if len(all_embeddings) == 0:
                logger.warning("No faces detected in the video.")
                return None

            all_embeddings = np.array(all_embeddings)
            if all_embeddings.ndim == 1:
                all_embeddings = all_embeddings.reshape(1, -1)

            logger.info("Clustering embeddings using DBSCAN...")
            clustering = DBSCAN(eps=0.6, min_samples=3, metric="cosine").fit(
                all_embeddings
            )
            labels = clustering.labels_

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"Clustering completed: {num_clusters} unique persons found.")

            for idx, label in enumerate(labels):
                if label == -1:
                    continue
                clusters.setdefault(label, []).append(all_faces[idx])

            logger.info("Face detection & clustering finished successfully.")
            return clusters

        except Exception as e:
            logger.error("Error during face detection and clustering", exc_info=True)
            raise FaceDetectionException(str(e), sys) from e

    def get_best_faces(self, clusters: dict) -> dict:
        """
        Select best representative face from each cluster
        but do NOT save here. Just return the aligned faces.
        """
        try:
            logger.info("Selecting best representative faces from clusters...")
            best_faces = {}

            if not clusters:
                logger.warning("No clusters found to extract best faces.")
                return best_faces

            for label, faces in clusters.items():
                best_face = max(
                    faces,
                    key=lambda f: (f["bbox"][2] - f["bbox"][0])
                    * (f["bbox"][3] - f["bbox"][1]),
                )
                frame_idx = best_face["frame_idx"]

                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    logger.warning(f"Could not read frame {frame_idx} for cluster {label}")
                    continue

                aligned_face = face_align.norm_crop(frame, landmark=best_face["kps"])
                best_faces[label] = aligned_face

            logger.info("Best representative faces selected successfully.")
            return best_faces

        except Exception as e:
            logger.error("Error while selecting best faces", exc_info=True)
            raise FaceDetectionException(str(e), sys) from e

    def video_preprocessing(self) -> FaceDetectionArtifact:
        try:
            logger.info("Starting full video preprocessing pipeline...")

            # Step 1: Extract audio
            audio_path = self.extract_audio()

            # Step 2: Detect & cluster faces
            clusters = self.detecting_and_clustering_faces()
            if clusters is None:
                raise FaceDetectionException("No faces detected in video.", sys)

            # Step 3: Select best faces
            best_faces = self.get_best_faces(clusters)

            # Step 4: Save best faces
            detected_faces_paths = []
            for idx, face_img in best_faces.items():
                save_path = os.path.join(self.detected_faces_dir, f"face_{idx}.jpg")
                cv2.imwrite(save_path, face_img)
                detected_faces_paths.append(save_path)

            # Step 5: Save clusters.json
            cluster_dict = {
                str(label): [f"face_{label}.jpg" for _ in faces]
                for label, faces in clusters.items()
            }
            with open(self.clusters_json, "w") as f:
                json.dump(cluster_dict, f, indent=4)

            logger.info("Video preprocessing completed successfully.")

            return FaceDetectionArtifact(
                detected_faces_path=self.detected_faces_dir,
                extracted_audio_path=audio_path,
            )

        except Exception as e:
            logger.error("Error during full preprocessing pipeline", exc_info=True)
            raise FaceDetectionException(str(e), sys) from e
