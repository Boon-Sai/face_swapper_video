import os
import sys
import json
import cv2
import insightface
import numpy as np
from insightface.app.common import Face

from src.exceptions.exception import FaceDetectionException
from src.loggings.logger import logger
from src.entity.config_entity import FaceSwappingConfig, ConfigEntity
from src.entity.artifact_entity import FaceDetectionArtifact, FaceSwappingArtifact

class SwapFaces:
    def __init__(self, source_image_path: str, detection_artifact: FaceDetectionArtifact, cluster_id_to_swap: int):
        try:
            logger.info("Started Face Swapping...")
            self.swapping_config = FaceSwappingConfig(config=ConfigEntity())
            self.swapper_model = insightface.model_zoo.get_model(self.swapping_config.face_swapper_model)
            self.face_analyzer = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            
            self.source_image_path = source_image_path
            self.detection_artifact = detection_artifact
            self.cluster_id_to_swap = cluster_id_to_swap
            
            self.video_name = os.path.basename(os.path.dirname(self.detection_artifact.detected_faces_path))
            self.output_video_dir = os.path.join(self.swapping_config.face_swapped_video_with_audio, self.video_name)
            os.makedirs(self.output_video_dir, exist_ok=True)

            logger.info(f"Face Swapping configurations invoked successfully with model path: {self.swapping_config.face_swapper_model}")

        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def get_source_face(self):
        try:
            logger.info(f"Reading source face from {self.source_image_path}")
            source_img = cv2.imread(self.source_image_path)
            source_faces = self.face_analyzer.get(source_img)
            if not source_faces:
                raise Exception("No face found in the source image.")
            logger.info(f"Found {len(source_faces)} face(s) in the source image. Using the first one.")
            return source_faces[0]
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def swap_faces(self, source_face) -> str:
        try:
            logger.info("Swapping faces in video frames...")
            with open(self.detection_artifact.json_information, 'r') as f:
                face_data = json.load(f)

            frame_paths = sorted(face_data.keys())
            
            output_frames_dir = os.path.join(self.output_video_dir, "swapped_frames")
            os.makedirs(output_frames_dir, exist_ok=True)

            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_path}")
                    continue
                
                faces_in_frame = face_data.get(frame_path, [])
                
                for face_info in faces_in_frame:
                    if face_info.get('cluster_id') == self.cluster_id_to_swap:
                        target_face = Face(bbox=np.array(face_info['bbox']), kps=np.array(face_info['kps']), det_score=face_info['det_score'])
                        target_face.embedding = np.array(face_info['embedding'])
                        frame = self.swapper_model.get(frame, target_face, source_face, paste_back=True)

                output_frame_path = os.path.join(output_frames_dir, os.path.basename(frame_path))
                cv2.imwrite(output_frame_path, frame)
            
            logger.info("Face swapping on frames completed.")
            return output_frames_dir
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def create_video_from_frames(self, frames_dir: str) -> str:
        try:
            logger.info("Creating video from swapped frames...")
            
            original_video_path = os.path.join(self.swapping_config.data_folder_path, f"{self.video_name}.mp4")
            if not os.path.exists(original_video_path):
                 original_video_path = os.path.join(self.swapping_config.data_folder_path, f"{self.video_name}.webm")
            if not os.path.exists(original_video_path):
                for ext in ['.avi', '.mov', '.mkv']:
                    original_video_path = os.path.join(self.swapping_config.data_folder_path, f"{self.video_name}{ext}")
                    if os.path.exists(original_video_path):
                        break
            
            cap = cv2.VideoCapture(original_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            output_video_path = os.path.join(self.output_video_dir, f"{self.video_name}_swapped.mp4")
            
            frame_files = sorted(os.listdir(frames_dir))
            if not frame_files:
                raise Exception("No frames found to create video.")

            first_frame_path = os.path.join(frames_dir, frame_files[0])
            first_frame = cv2.imread(first_frame_path)
            height, width, layers = first_frame.shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            for frame_file in frame_files:
                frame_path = os.path.join(frames_dir, frame_file)
                frame = cv2.imread(frame_path)
                video_writer.write(frame)
            
            video_writer.release()
            logger.info(f"Video created at: {output_video_path}")
            return output_video_path
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def add_audio_to_video(self, video_path: str) -> str:
        try:
            logger.info("Adding audio to the swapped video...")
            
            input_video = ffmpeg.input(video_path)
            input_audio = ffmpeg.input(self.detection_artifact.extracted_audio_path)

            output_filename = os.path.join(self.swapping_config.face_swapped_video_with_audio, f"{self.video_name}_final.mp4")

            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_filename).run(overwrite_output=True)

            logger.info(f"Video with new audio created successfully at: {output_filename}")
            return output_filename
        except ffmpeg.Error as e:
            raise FaceDetectionException(f"Error adding audio to video: {e.stderr.decode()}", sys) from e
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e

    def initiate_face_swapping(self) -> FaceSwappingArtifact:
        try:
            logger.info("Initiating face swapping pipeline...")
            source_face = self.get_source_face()
            swapped_frames_dir = self.swap_faces(source_face)
            swapped_video_path = self.create_video_from_frames(swapped_frames_dir)
            final_video_path = self.add_audio_to_video(swapped_video_path)
            
            artifact = FaceSwappingArtifact(final_output_video_path=final_video_path)
            logger.info(f"Face swapping pipeline completed. Artifact: {artifact}")
            return artifact
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e
