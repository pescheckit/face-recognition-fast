# Face Recognition Fast

A **developer-friendly**, multi-method face recognition system featuring:
- **OpenCV** (Haar Cascade),
- **MTCNN**,
- **face_recognition** (dlib-based), and
- **MobileNet** (TensorFlow/Keras).

Compare **speed** and **memory usage** of these approaches on your own images, or just pick a single method to run.

## Table of Contents

- [Features](#features)  
- [Setup & Installation](#setup--installation)  
- [Usage](#usage)  
  - [Single Method](#single-method)  
  - [Benchmark Mode](#benchmark-mode)  
  - [Example Output](#example-output)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)

---

## Features

1. **Multiple Face Detection & Embedding Methods**  
   - *OpenCV* for detection, with face\_recognition embeddings  
   - *MTCNN* for detection (TensorFlow-based)  
   - *face\_recognition* for both detection + embedding  
   - *MobileNet* as an alternate embedding model (TensorFlow/Keras)

2. **Benchmark Mode**  
   - Runs *all four* methods in a single pass and reports:
     - Execution time
     - Memory usage difference
     - L2 distance
     - Cosine similarity

3. **Local (Lazy) Imports**  
   - TensorFlow only loads if you actually use MTCNN or MobileNet.  
   - This keeps overhead and logs minimal for non-TF methods.

4. **Easy CLI**  
   - Just provide two image paths (reference & test images) plus a method (or `--benchmark`).

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pescheckit/face-recognition-fast.git
cd face-recognition-fast
```

### 2. Create a Virtual Environment

To keep things clean and self-contained, create a Python virtual environment. For example:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> On Windows, use:
> ```
> .venv\Scripts\activate
> ```

### 3. Install Dependencies

Install all required libraries from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**:  
> - `face_recognition` depends on **dlib**. If you encounter build issues, see the [face_recognition docs](https://github.com/ageitgey/face_recognition#installation).  
> - `mtcnn` and `MobileNet` embedding require **TensorFlow**. Make sure you have a compatible CPU (and optional GPU drivers if you plan on using CUDA).  

---

## Usage

From the **face-recognition-fast** folder:

```bash
python main.py --help
```

will show you the available arguments. You must pass **two images**:  
1. A reference or ID image (`id_image_path`)  
2. A second image to compare (`webcam_image_path`)  

### Single Method

Choose exactly **one** of the four methods:

- `opencv`
- `mtcnn`
- `facerecognition`
- `mobilenet`

**Example**:

```bash
python main.py images/id1.png images/front1.png --method opencv
```

This will:
1. Detect faces in both images using OpenCV’s Haar Cascade.  
2. Generate embeddings using the `face_recognition` library.  
3. Print out the L2 distance and cosine similarity.

### Benchmark Mode

Compare **all four methods** at once:

```bash
python main.py images/id1.png images/front1.png --benchmark
```

You’ll see a summary of each method’s:
- **Status** (OK if detection & embedding succeeded)  
- **Time** in seconds (wall-clock time)  
- **MemΔ** in bytes (approximate memory usage difference during that run)  
- **Distance** (L2 norm of embeddings)  
- **Similarity** (cosine similarity * 100, as a percentage)

#### Example Output

Below is a sample of the **benchmark results**:

```
===== BENCHMARK RESULTS =====
Method:          opencv | Status: OK | Time: 1.0121s | MemΔ: 196816896 bytes | Distance: 0.4240 | Similarity: 96.23%
Method:           mtcnn | Status: OK | Time: 1.9144s | MemΔ: 563507200 bytes | Distance: 0.4421 | Similarity: 95.76%
Method: facerecognition | Status: OK | Time: 0.7560s | MemΔ: 14987264  bytes | Distance: 0.4142 | Similarity: 96.65%
Method:       mobilenet | Status: OK | Time: 1.5593s | MemΔ: 37396480  bytes | Distance: 21.5536 | Similarity: 62.89%
```

- **`opencv`** uses Haar Cascade detection + face_recognition embedding  
- **`mtcnn`** uses MTCNN detection + face_recognition embedding (requires TF)  
- **`facerecognition`** uses face\_recognition for both detection & embedding  
- **`mobilenet`** uses face\_recognition detection + MobileNet embedding  

---

## Project Structure

```
face-recognition-fast/
├── main.py
├── src/
│   ├── face_detection.py
│   ├── face_embedding.py
│   ├── compare_faces.py
│   └── utils.py
├── images/
│   ├── id1.png
│   └── front1.png
├── requirements.txt
└── README.md
```

- **main.py**  
  The primary script. Parses arguments, chooses the method, coordinates detection + embedding + comparison, and prints or benchmarks.

- **src/face_detection.py**  
  Contains detection functions for OpenCV, MTCNN, and face\_recognition. Each function imports its library only when called (lazy import).

- **src/face_embedding.py**  
  Provides embedding methods. Supports face\_recognition (128-d embeddings) or MobileNet (TensorFlow-based, ~1280-d embedding).

- **src/compare_faces.py**  
  Contains comparison logic (L2 distance, cosine similarity).

- **src/utils.py**  
  Optional utilities (e.g., garbage collection).

- **requirements.txt**  
  Lists dependencies like `opencv-python`, `face-recognition`, `mtcnn`, `tensorflow`, `numpy`, `scipy`, `psutil`.

---

## Contributing

1. **Fork** the project on GitHub.  
2. Create a **new branch**:  
   ```bash
   git checkout -b my-feature
   ```
3. Make your changes and **commit**:  
   ```bash
   git commit -am "Add new feature"
   ```
4. **Push** to your branch:  
   ```bash
   git push origin my-feature
   ```
5. Open a **Pull Request** on GitHub and describe your changes.

---

## License

This project is distributed under the [MIT License](LICENSE).
