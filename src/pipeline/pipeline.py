import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from insightface.utils import face_align

from src.loggings.logger import logger
from src.components.face_detection import DetectFaces
from src.components.face_swapper import SwapFaces
from src.entity.artifact_entity import FaceDetectionArtifact

def main():
    video_path = "/home/prem/Boonsai/face_swap_video/data/input_2.mp4"

    # Initialize face detection and perform preprocessing
    df = DetectFaces(video_path=video_path)
    detection_artifact, clusters = df.video_preprocessing()

    # Handle case where no clusters are found
    if not clusters:
        logger.warning("No valid clusters found in the video. Exiting.")
        sys.exit(0)

    # Select representative (best) face for each cluster for display
    cluster_reps = []
    for label, faces in sorted(clusters.items()):
        best_face = max(
            faces,
            key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1])
        )
        cluster_reps.append((label, best_face))

    # Display unique faces if clusters exist
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
            cap = cv2.VideoCapture(video_path)
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

    # Obtain source image path from user
    source_face_path = input("Enter the path to your source image: ").strip().strip('"')

    # Validate source image existence
    if not Path(source_face_path).exists():
        logger.error(f"Source image '{source_face_path}' not found. Exiting.")
        sys.exit(1)

    # Load and validate source face using existing detection model
    source_img = cv2.imread(source_face_path)
    source_faces = df.face_detection_model.get(source_img)
    if len(source_faces) == 0:
        print("No face found in the source image. Exiting.")
        sys.exit(1)

    # Display source face
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    plt.title("Source Face", fontsize=14)
    plt.axis('off')
    plt.show()

    # Obtain chosen index from user
    try:
        index = int(input(f"Enter the index of the person to swap (0 to {len(clusters)-1}, or -1 for all): "))
        if index < -1 or (index >= len(clusters) and index != -1):
            print("Invalid index. Exiting.")
            sys.exit(1)
    except ValueError:
        logger.error("Invalid index input. Must be an integer.")
        sys.exit(1)

    # Prepare clusters for swapping based on index
    if index == -1:
        swap_clusters = []
    else:
        swap_clusters = clusters.get(index, [])
        if not swap_clusters:
            logger.warning(f"No faces found for index {index}. Exiting.")
            sys.exit(0)

    # Set extracted audio path for use in swapping
    FaceDetectionArtifact.extracted_audio_path = detection_artifact.extracted_audio_path

    # Initialize and perform face swapping
    sf = SwapFaces(index=index, video_path=video_path, source_face_path=source_face_path, clusters=swap_clusters)
    swapping_artifact = sf.video_preprocessing()

if __name__ == "__main__":
    main()