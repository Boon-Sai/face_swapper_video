import os
import sys
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN
import ffmpeg
from insightface.utils import face_align
from collections import Counter

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
            self.frames_dir = os.path.join(self.detection_config.face_detection_folder_path, "frames", self.video_name)
            os.makedirs(self.frames_dir, exist_ok=True)
            logger.info(f"Face Detection configurations invoked successfully")
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def extract_audio(self) -> str:
        try:
            logger.info("Checking for audio in video...")
            probe = ffmpeg.probe(self.video_path)
            has_audio = any(stream['codec_type'] == 'audio' for stream in probe.get('streams', []))
            if not has_audio:
                logger.info("No audio stream found in the video. Skipping audio extraction.")
                return None

            logger.info("Extracting audio from video...")
            output_filename = os.path.join(self.detection_config.face_detection_folder_path, f"{self.video_name}_audio.mp3")
            ffmpeg.input(self.video_path).output(output_filename, acodec='mp3').run()
            logger.info(f"Audio extracted and saved to: {output_filename}")
            return output_filename
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else "No stderr output available."
            logger.error(f"Error extracting audio: {error_msg}")
            raise FaceDetectionException(f"Error extracting audio: {error_msg}", sys) from e
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def video_preprocess(self, sampling_interval: int = 10) -> list:
        try:
            logger.info(f"Processing video and extracting frames (sampling every {sampling_interval} frames)...")
            cap = cv2.VideoCapture(self.video_path)
            frame_paths = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames to improve efficiency
                if frame_count % sampling_interval == 0:
                    frame_path = os.path.join(self.frames_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frame_paths)} frames from the video (total frames: {frame_count}).")
            return frame_paths, frame_count
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def detect_and_cluster_faces(self, frame_paths: list, total_frames: int) -> str:
        try:
            logger.info("Detecting and clustering faces from frames...")
            all_faces_data = {}
            all_embeddings = []
            face_indices = []
            cluster_best_representative = {}

            for i, frame_path in enumerate(frame_paths):
                frame = cv2.imread(frame_path)
                faces = self.face_detector_model.get(frame)
                all_faces_data[frame_path] = []
                
                if faces:
                    for face in faces:
                        # Align and normalize face
                        aligned_face = face_align.norm_crop(frame, landmark=face.kps)
                        
                        all_faces_data[frame_path].append({
                            'bbox': face.bbox.tolist(),
                            'kps': face.kps.tolist(),
                            'det_score': float(face.det_score),
                            'embedding': face.embedding.tolist(),
                            'frame_idx': i * 10  # Approximate original frame index
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
            dbscan = DBSCAN(eps=0.6, min_samples=3, metric="cosine")
            labels = dbscan.fit_predict(all_embeddings)

            for i, label in enumerate(labels):
                frame_path, face_index = face_indices[i]
                all_faces_data[frame_path][face_index]['cluster_id'] = int(label)
                
                if int(label) != -1:
                    det_score = all_faces_data[frame_path][face_index]['det_score']
                    bbox = all_faces_data[frame_path][face_index]['bbox']
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    
                    # Better representative selection (size + detection score)
                    best = cluster_best_representative.get(int(label))
                    if best is None or (bbox_area * det_score) > (best[3] * best[2]):
                        cluster_best_representative[int(label)] = (frame_path, face_index, det_score, bbox_area)

            json_path = os.path.join(self.detection_config.face_detection_folder_path, f"{self.video_name}_face_data.json")
            with open(json_path, 'w') as f:
                json.dump(all_faces_data, f, indent=4)
            
            logger.info(f"Face detection and clustering data saved to: {json_path}")

            # Save unique representative face crops per cluster
            try:
                unique_faces_dir = os.path.join(self.detection_config.face_detection_folder_path, "unique_faces", self.video_name)
                os.makedirs(unique_faces_dir, exist_ok=True)
                
                for cluster_id, (rep_frame_path, rep_face_index, _, _) in cluster_best_representative.items():
                    frame_img = cv2.imread(rep_frame_path)
                    if frame_img is None:
                        continue
                    
                    bbox = all_faces_data[rep_frame_path][rep_face_index]['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame_img.shape[1], x2); y2 = min(frame_img.shape[0], y2)
                    
                    crop = frame_img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    out_path = os.path.join(unique_faces_dir, f"cluster_{cluster_id}.jpg")
                    cv2.imwrite(out_path, crop)
                
                logger.info(f"Saved unique face representatives to: {unique_faces_dir}")
            except Exception as crop_err:
                logger.warning(f"Failed to save unique face crops: {crop_err}")
            
            return json_path, cluster_best_representative
            
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def display_cluster_faces(self, cluster_best_representative: dict, all_faces_data: dict):
        """Display cluster faces for user selection (similar to reference)"""
        try:
            import matplotlib.pyplot as plt
            from IPython.display import display, clear_output
            
            cluster_reps = []
            for cluster_id, (frame_path, face_index, _, _) in cluster_best_representative.items():
                cluster_reps.append((cluster_id, frame_path, face_index))
            
            num_faces = len(cluster_reps)
            if num_faces == 0:
                logger.info("No clusters found to display")
                return
            
            # Create visualization (simplified version of reference)
            cols = min(5, num_faces)
            rows = (num_faces + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
            
            if num_faces == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(rows * cols):
                row = i // cols
                col = i % cols
                
                if i < num_faces:
                    cluster_id, frame_path, face_index = cluster_reps[i]
                    frame_img = cv2.imread(frame_path)
                    
                    if frame_img is not None:
                        bbox = all_faces_data[frame_path][face_index]['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(frame_img.shape[1], x2); y2 = min(frame_img.shape[0], y2)
                        
                        crop = frame_img[y1:y2, x1:x2]
                        if crop.size > 0:
                            axes[row, col].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    
                    axes[row, col].set_title(f"Cluster {cluster_id}", fontsize=12)
                    axes[row, col].axis('off')
                else:
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.detection_config.face_detection_folder_path, f"{self.video_name}_clusters.png"))
            plt.close()
            
            logger.info(f"Cluster visualization saved")
            
        except Exception as e:
            logger.warning(f"Could not create cluster visualization: {e}")

    def initiate_face_detection(self) -> FaceDetectionArtifact:
        try:
            logger.info("Initiating face detection pipeline...")
            audio_path = self.extract_audio()
            frame_paths, total_frames = self.video_preprocess()
            json_path, cluster_reps = self.detect_and_cluster_faces(frame_paths, total_frames)
            
            # Load data to display clusters
            with open(json_path, 'r') as f:
                all_faces_data = json.load(f)
            self.display_cluster_faces(cluster_reps, all_faces_data)
            
            artifact = FaceDetectionArtifact(
                detected_faces_path=self.frames_dir,
                json_information=json_path,
                extracted_audio_path=audio_path
            )
            logger.info(f"Face detection pipeline completed successfully.")
            return artifact
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e