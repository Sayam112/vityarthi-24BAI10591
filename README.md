# Computer Vision Image Analysis Toolkit

## 1. Project Overview

Computer Vision Image Analysis Toolkit is a command-line Python project that demonstrates practical computer vision techniques covered in the course.

The project accepts an image from the command line and provides five functional modules:

1. Image Enhancement and Low-Level Processing
2. Feature Extraction
3. Image Segmentation
4. Pattern Analysis
5. Motion Analysis

The project is intentionally designed to run completely from a terminal without requiring a GUI-based setup.

## 2. Course Concepts Covered

| Course Area | Implemented Concepts |
|---|---|
| Digital Image Formation & Low-Level Processing | Grayscale conversion, Gaussian filtering, Median filtering, Histogram Equalization, Fourier Transform, Canny |
| Feature Extraction | Canny, Harris Corners, SIFT, HOG |
| Image Segmentation | Thresholding, K-Means |
| Pattern Analysis | K-Means clustering, PCA |
| Motion Analysis | Frame Difference, Optical Flow |
| Multi-Camera/Geometry | Discussed as future extension; not claimed as implemented |

## 3. Functional Modules

### Module 1 - Image Enhancement
- Grayscale conversion
- Gaussian filtering
- Median filtering
- Histogram equalization
- Fourier magnitude spectrum
- Canny edge detection

### Module 2 - Feature Extraction
- Harris corner detection
- SIFT keypoint extraction
- HOG feature extraction

### Module 3 - Image Segmentation
- Binary threshold segmentation
- K-Means color segmentation

### Module 4 - Pattern Analysis
- K-Means clustering
- PCA dimensionality reduction

### Module 5 - Motion Analysis
- Frame difference
- Dense optical flow

## 4. System Architecture

```text
                +----------------------+
                |      User / CLI      |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |       main.py        |
                |   Input & Menu Flow  |
                +----------+-----------+
                           |
        +------------------+------------------+
        |         |          |        |       |
        v         v          v        v       v
   Processing  Features  Segmentation Pattern Motion
        |         |          |        |       |
        +---------+----------+--------+-------+
                           |
                           v
                +----------------------+
                |   output/ directory  |
                | Images + CSV results  |
                +----------------------+
```

## 5. Workflow

```text
Start
  |
  v
Check input image
  |
  v
Select module
  |
  v
Select operation
  |
  v
Process image/data
  |
  v
Validate result
  |
  v
Save result in output/
  |
  v
Display result path
  |
  v
Exit
```

## 6. Requirements

- Python 3.9 or later
- OpenCV
- NumPy
- scikit-image
- scikit-learn

No GUI framework is required.

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/computer-vision-image-analysis.git
cd computer-vision-image-analysis
```

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 8. Running the Project

Run the application:

```bash
python main.py
```

The program will display a command-line menu.

You can also directly process an image:

```bash
python main.py --image path/to/image.jpg
```

If no image is supplied, the program can generate a small synthetic sample image for demonstration.

## 9. Running Tests

Run:

```bash
python test_project.py
```

A successful test run prints:

```text
All tests passed successfully.
```

## 10. Output

All generated results are stored in:

```text
output/
```

Examples:

```text
output/grayscale.png
output/gaussian.png
output/histogram_equalized.png
output/fourier_spectrum.png
output/canny_edges.png
output/harris_corners.png
output/sift_features.png
output/hog_visualization.png
output/kmeans_segmentation.png
output/threshold_segmentation.png
output/pca_result.csv
output/frame_difference.png
output/optical_flow.png
```

## 11. Non-Functional Requirements

### Performance
The application should process normal-sized images using standard OpenCV operations without unnecessary repeated computation.

### Usability
The command-line menu uses numbered choices and clear prompts.

### Reliability
The program validates file paths and handles invalid image input without terminating unexpectedly.

### Maintainability
Each major computer vision area is implemented in a separate Python module.

### Resource Efficiency
The project uses common Python computer vision libraries and does not require dedicated GPU hardware.

### Portability
The project can be executed from a terminal on Windows, Linux, or macOS with Python and the listed dependencies installed.

## 12. Testing Strategy

Testing covers:
- Image loading
- Image shape validation
- Filtering output
- Histogram processing
- Fourier transform
- Feature extraction
- Segmentation
- PCA
- Motion processing

## 13. Limitations

The current implementation focuses on fundamental computer vision algorithms suitable for a course project. Advanced topics such as full stereo 3-D reconstruction, auto-calibration, DLT-based reconstruction, RANSAC homography estimation, MRF segmentation, KLT tracking and dynamic stereo are not claimed as implemented features.

## 14. Future Enhancements

- Stereo depth estimation
- Camera calibration
- Homography and image rectification
- DLT 3-D reconstruction
- RANSAC-based geometric estimation
- Real-time webcam motion analysis
- KLT feature tracking
- Object detection using a trained model

## 15. Conclusion

The project provides a modular command-line toolkit for demonstrating fundamental image processing, feature extraction, segmentation, pattern analysis and motion analysis techniques. Its modular structure makes it suitable for experimentation and future extension.
