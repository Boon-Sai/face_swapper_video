import os
from datetime import datetime
from src.constants import *

class ConfigEntity:
    def __init__(self, timestamp = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%y__%H_%M_%S")

        self.face_detection_model = DETECTION_MODEL
        self.face_swapping_model = SWAPPING_MODEL
        self.clustering_model = CLUSTERING_MODEL
        self.audio_model = AUDIO_MODEL
        self.ctx_id = CTX_ID
        self.det_size = DET_SIZE
        
        self.logs_dir_path = LOG_DIR_PATH
        self.artifacts_dir_path = ARTIFACT_DIR_PATH
        self.data_dir_path = DATA_DIR_PATH

class FaceDetectionConfig:
    def __init__(self, config: ConfigEntity):
        self.data_dir_path = config.data_dir_path
        self.artifact_dir_path = config.artifacts_dir_path
        self.face_detection_model = config.face_detection_model
        self.ctx_id = config.ctx_id
        self.det_size = config.det_size
        
        # face detection main folder
        self.face_detection_folder_path = os.path.join(
            self.artifact_dir_path,"face_datection"
        )

        os.makedirs(self.face_detection_folder_path, exist_ok=True)

        self.detected_faces = os.path.join(
            self.face_detection_folder_path,"detected_faces" 
        )

        os.makedirs(self.detected_faces, exist_ok=True)

class FaceSwappingConfig:
    def __init__(self, config: ConfigEntity):
        self.data_folder_path = config.data_dir_path
        self.artifact_folder_path = config.artifacts_dir_path
        self.face_swapper_model = config.face_swapping_model

        self.face_swapped_video_with_audio = os.path.join(
            self.artifact_folder_path, "face_swapped_video_with_audio"
        )
        os.makedirs(self.face_swapped_video_with_audio, exist_ok=True)



        
