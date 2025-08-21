
---

````markdown
# Face Swapping with InsightFace

This project performs **automatic face detection, clustering, and face swapping in videos** using [InsightFace](https://github.com/deepinsight/insightface).  
It allows you to detect all unique persons in a video, choose one (or all), and swap their face with a face from a source image.

---

## Features

- Detects all unique faces in a video
- Clusters faces by identity using DBSCAN
- Displays representative faces for each unique person
- Allows swapping faces of:
  - A specific person (selected by index)
  - All persons in the video
- Generates an output video with swapped faces

---

## Requirements

- Python 3.8 or higher
- GPU recommended for faster processing (but works on CPU)

---

## Installation

1. Clone or download this project.
2. Install required packages:

```bash
pip install insightface opencv-python matplotlib onnxruntime scikit-learn
````

---

## Usage

1. Run the script in Jupyter Notebook or any Python environment:

```bash
python face_swap.py
```

2. Enter the path to your input video when prompted:

```
Enter the path to your input video: path/to/video.mp4
```

3. The program will scan and cluster faces, then display unique representative faces with indices.

4. Enter the path to your source image when prompted:

```
Enter the path to your source image: path/to/source.jpg
```

5. Choose which person to swap:

   * Enter a number (e.g., `0`, `1`, `2`) to select a specific person.
   * Enter `-1` to swap **all persons** in the video.

6. Wait for processing. The output video will be saved in the same directory as:

```
swapped_inputvideo.mp4
```

---

## Example Workflow

1. Input video is scanned.
2. Script displays unique persons found:

```
Index 0: Representative face at [x1:..., y1:..., x2:..., y2:...]
Index 1: Representative face at [x1:..., y1:..., x2:..., y2:...]
Index -1: Swap ALL persons
```

3. You provide a source image.
4. You choose which index to swap.
5. Script generates the swapped output video.

---

## Notes

* Ensure your source image contains a clear, front-facing face.
* Larger or longer videos will take more processing time.
* If no faces are detected, check video resolution or face visibility.
* Similarity threshold for matching faces is set to `0.65`. Adjust if necessary.

---

## License

This project is for educational and research purposes only. Please respect privacy and consent when using face-swapping technologies.

```

```
