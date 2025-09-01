import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.utils import face_align

from src.loggings.logger import logger
from src.pipeline.face_swap_video_pipeline import FaceSwapPipeline

def display_and_select_faces(pipeline: FaceSwapPipeline):
    """Displays unique faces and prompts user for selection."""
    if not pipeline.clusters:
        logger.warning("No clusters to display.")
        return None, None

    cluster_reps = []
    for label, faces in sorted(pipeline.clusters.items()):
        best_face = max(
            faces,
            key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])
        )
        cluster_reps.append((label, best_face))

    num_faces = len(cluster_reps)
    if num_faces > 0:
        cols = min(5, num_faces)
        rows = (num_faces + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

        if num_faces == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_faces):
            row = i // cols
            col = i % cols
            label, face = cluster_reps[i]
            frame_idx = face['frame_idx']
            cap = cv2.VideoCapture(pipeline.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                aligned_face = face_align.norm_crop(frame, landmark=face['kps'])
                axes[row, col].imshow(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"Index {label}", fontsize=12)
            axes[row, col].axis('off')

        for i in range(num_faces, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

        print("--- Unique Persons Found ---")
        for label, face in cluster_reps:
            bbox = face['bbox'].astype(int)
            print(f"Index {label}: Representative face at [x1:{bbox[0]}, y1:{bbox[1]}, x2:{bbox[2]}, y2:{bbox[3]}]")
        print("Index -1: Swap ALL persons")

    source_face_path = input("Enter the path to your source image: ").strip().strip('"')

    if not Path(source_face_path).exists():
        logger.error(f"Source image '{source_face_path}' not found. Exiting.")
        return None, None

    source_img = cv2.imread(source_face_path)
    
    if pipeline.face_detector is None:
        logger.error("Face detector not initialized. Run detect_faces first.")
        return None, None
        
    source_faces = pipeline.face_detector.face_detection_model.get(source_img)
    if len(source_faces) == 0:
        print("No face found in the source image. Exiting.")
        return None, None

    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    plt.title("Source Face", fontsize=14)
    plt.axis('off')
    plt.show()

    try:
        index = int(input(f"Enter the index of the person to swap (0 to {len(pipeline.clusters)-1}, or -1 for all): "))
        if index < -1 or (index >= len(pipeline.clusters) and index != -1):
            print("Invalid index. Exiting.")
            return None, None
    except ValueError:
        logger.error("Invalid index input. Must be an integer.")
        return None, None
        
    return source_face_path, index

def main():
    """Main function to run the face swapping pipeline."""
    video_path_str = input("Enter the path to your video file: ").strip().strip('"')
    video_path = Path(video_path_str)

    if not video_path.exists() or not video_path.is_file():
        logger.error(f"Video file not found or is not a file: {video_path_str}")
        sys.exit(1)

    try:
        pipeline = FaceSwapPipeline(video_path=str(video_path))
        if pipeline.detect_faces():
            source_face_path, index = display_and_select_faces(pipeline)
            if source_face_path and index is not None:
                pipeline.swap_faces(source_face_path, index, pipeline.clusters, pipeline.detection_artifact)

    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        sys.exit(1)
