import os
import json
import sys
import insightface
import numpy as np
import ffmpeg
from insightface.app.common import Face
from insightface.app import FaceAnalysis

from src.exceptions.exception import FaceDetectionException
from src.loggings.logger import logger
from src.entity.config_entity import(
    FaceSwappingConfig,
    ConfigEntity
)
from src.entity.artifact_entity import (
    FaceSwappingArtifact,
    FaceDetectionArtifact
)


class SwapFaces:
    def __init__(self):
        ...
    
    def swap_faces(self):
        ...
    
    def re_arrange_video_with_swapped_face(self):
        ...
    
    def insert_audio(self):
        ...