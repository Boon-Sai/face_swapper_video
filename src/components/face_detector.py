import os
import sys
import json
import cv2
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from IPython.display import clear_output


from src.exceptions.exception import FaceDetectionException
from src.loggings.logger import logger

from src.entity.config_entity import (
    FaceDetectionConfig,
    ConfigEntity
)

from src.entity.artifact_entity import FaceDetectionArtifact



class DetectFaces:
    def __init__(self):
        try: 
            logger.Logger.info("started Detecting Faces..")
            
            self.detect_faces = FaceDetectionConfig(config=ConfigEntity())
            self.face_detector_model = FaceAnalysis(name = self.detect_faces.face_detecion_model)
            self.face_detector_model.prepare(ctx_id=self.detect_faces.ctx_id, det_size=self.detect_faces.det_size)
            logger.logger.info(f"Face Detection configurations invoked successfully with model path: {self.detect_faces.face_detection_model}")
                    
        except Exception as e:
            raise FaceDetectionException(str(e), sys) from e
    
    def extract_audio(self):
        ...

    def video_preprocess(self):
        ...

    def clustering_frames(self):
        ...

    def detect_faces(self):
        ...