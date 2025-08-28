import os
import sys
import json
from collections import Counter

from src.components.face_detection import DetectFaces
from src.loggings.logger import logger
from src.exceptions.exception import FaceDetectionException

class FaceSwapPipeline:
    def __init__(self, video_path):
        self.video_path = video_path
    
    def detecting_faces_pipeline(self):
        df = DetectFaces(video_path=self.video_path)
        df.video_preprocessing()

# Removed the redundant if __name__ == "__main__": block to avoid instantiation errors.
# If standalone testing is needed, provide a video_path argument or run via main.py.