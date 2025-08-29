# src/main.py

import sys

from src.loggings.logger import logger
from src.pipeline.pipeline import FaceSwapPipeline

def main():
    video_path = "/home/prem/Boonsai/face_swap_video/data/input_2.mp4"

    
    pipeline = FaceSwapPipeline(video_path=video_path, source_face_path=source_face_path, index=index)
    detection_artifact, clusters = pipeline.detecting_faces_pipeline()
    swapping_artifact = pipeline.swapping_faces_pipeline(detection_artifact, clusters)
    source_face_path = input("Enter the path to your source image: ").strip().strip('"')

    try:
        index = int(input("Enter the index of the person to swap (0 to n-1, or -1 for all): "))
    except ValueError:
        logger.error("Invalid index input. Must be an integer.")
        sys.exit(1)
    pipeline.run_full_pipeline()  # Invoke the full pipeline to perform detection and swapping

if __name__ == "__main__":
    main()