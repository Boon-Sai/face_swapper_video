import sys
from pathlib import Path

from src.loggings.logger import logger
from src.pipeline.face_swap_video_pipeline import FaceSwapPipeline

def main():
    """Main function to run the face swapping pipeline."""
    video_path_str = input("Enter the path to your video file: ").strip().strip('"')
    video_path = Path(video_path_str)

    if not video_path.exists() or not video_path.is_file():
        logger.error(f"Video file not found or is not a file: {video_path_str}")
        sys.exit(1)

    try:
        pipeline = FaceSwapPipeline(video_path=str(video_path))
        pipeline.run_full_pipeline()
    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
