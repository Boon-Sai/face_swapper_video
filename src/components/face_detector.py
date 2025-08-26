import os
import sys
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
import moviepy.editor as mp

from src.exceptions.exception import FaceDetectionException
from src.loggings.logger import logger
from src.entity.config_entity import FaceDetectionConfig, ConfigEntity
from src.entity.artifact_entity import FaceDetectionArtifact

class DetectFaces:
    def __init__(self, video_path: str):
        try:
            logger.info("Started Detecting Faces...")
            self.detection_config = FaceDetectionConfig(config=ConfigEntity())
            self.face_detector_model = FaceAnalysis(name=self.detection_config.face_detection_model)
            self.face_detector_model.prepare(ctx_id=self.detection_config.ctx_id, det_size=self.detection_config.det_size)
            self.video_path = video_path
            self.video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.output_dir = os.path.join(self.detection_config.detected_faces, self.video_name)
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Face Detection configurations invoked successfully with model path: {self.detection_config.face_detection_model}")
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def extract_audio(self) -> str:
        try:
            logger.info("Extracting audio from video...")
            audio_path = os.path.join(self.detection_config.face_detection_folder_path, f"{self.video_name}_audio.mp3")
            ffmpeg.input(self.video_path).output(audio_path, acodec='mp3').run(overwrite_output=True)
            logger.info(f"Audio extracted and saved to: {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            raise FaceDetectionException(f"Error extracting audio: {e.stderr.decode()}", sys) from e
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def video_preprocess(self) -> list:
        try:
            logger.info("Processing video and extracting frames...")
            cap = cv2.VideoCapture(self.video_path)
            frame_paths = []
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(self.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_count += 1
            cap.release()
            logger.info(f"Extracted {frame_count} frames from the video.")
            return frame_paths
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def detect_and_cluster_faces(self, frame_paths: list) -> str:
        try:
            logger.info("Detecting and clustering faces from frames...")
            all_faces_data = {}
            all_embeddings = []
            face_indices = []

            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                faces = self.face_detector_model.get(frame)
                all_faces_data[frame_path] = []
                if faces:
                    for face in faces:
                        all_faces_data[frame_path].append({
                            'bbox': face.bbox.tolist(),
                            'kps': face.kps.tolist(),
                            'det_score': float(face.det_score),
                            'embedding': face.embedding.tolist()
                        })
                        all_embeddings.append(face.embedding)
                        face_indices.append((frame_path, len(all_faces_data[frame_path]) - 1))

            if not all_embeddings:
                logger.info("No faces detected in the video.")
                json_path = os.path.join(self.detection_config.face_detection_folder_path, f"{self.video_name}_face_data.json")
                with open(json_path, 'w') as f:
                    json.dump(all_faces_data, f, indent=4)
                return json_path


            all_embeddings = np.array(all_embeddings)
            
            logger.info("Clustering detected faces...")
            dbscan = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
            labels = dbscan.fit_predict(all_embeddings)

            for i, label in enumerate(labels):
                frame_path, face_index = face_indices[i]
                all_faces_data[frame_path][face_index]['cluster_id'] = int(label)

            json_path = os.path.join(self.detection_config.face_detection_folder_path, f"{self.video_name}_face_data.json")
            with open(json_path, 'w') as f:
                json.dump(all_faces_data, f, indent=4)
            
            logger.info(f"Face detection and clustering data saved to: {json_path}")
            return json_path
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e
            
    def initiate_face_detection(self) -> FaceDetectionArtifact:
        try:
            logger.info("Initiating face detection pipeline...")
            audio_path = self.extract_audio()
            frame_paths = self.video_preprocess()
            json_path = self.detect_and_cluster_faces(frame_paths)
            
            artifact = FaceDetectionArtifact(
                detected_faces_path=self.output_dir,
                json_information=json_path,
                extracted_audio_path=audio_path
            )
            logger.info(f"Face detection pipeline completed successfully. Artifacts: {artifact}")
            return artifact
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e