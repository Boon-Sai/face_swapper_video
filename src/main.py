import sys

from src.loggings.logger import logger
from src.pipeline.pipeline import FaceSwapPipeline

def main():
    video_path = "/home/prem/Boonsai/face_swap_video/data/input_2.mp4"
    pipeline = FaceSwapPipeline(video_path=video_path)
    pipeline.detecting_faces_pipeline()  # Invoke the pipeline method to perform processing

if __name__ == "__main__":
    main()