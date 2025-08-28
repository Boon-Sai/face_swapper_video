import sys
import argparse
from src.loggings.logger import logger
from src.pipeline.pipeline import FaceSwapPipeline

def main():
    
    parser = argparse.ArgumentParser(description="Run the Face swap pipeline.")
    parser.add_argument("--video_path", type=str, required=True, help='Path to the input video.')
    parser.add_argument("--source_image_path", required=True, type=str, help="Path to the source image file for face swapping.")

    args = parser.parse_args()

    pipeline = FaceSwapPipeline(video_path=args.video_path, image_path=args.source_image_path)
    # final_video_path = pipeline.run(user_cluster_id=args.cluster_id)

    # if final_video_path:
    #     logger.info(f"Pipeline executed successfully, output: {final_video_path}")
    
if __name__ == "__main__":
    main()