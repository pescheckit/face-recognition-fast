# Face Recognition Fast

A **developer-friendly**, multi-method face recognition system featuring:
- **OpenCV** (Haar Cascade),
- **MTCNN**,
- **face_recognition** (dlib-based),
- **MobileNet** (TensorFlow/Keras),
- **ArcFace** (InsightFace framework),
- **DeepFace** (multi-backend wrapper).

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
   - *MobileNet* (MTCNN detection + MobileNetV2 embedding)  
   - *ArcFace* (RetinaFace detection + ArcFace embedding)  
   - *DeepFace* (multi-backend with Facenet, VGG-Face, ArcFace, etc.)

2. **Benchmark Mode**  
   - Runs *all six* methods in a single pass and reports:
     - Execution time
     - Memory usage difference (MemΔ)
     - L2 distance
     - Cosine similarity

3. **Local (Lazy) Imports**  
   - TensorFlow only loads if you actually use MTCNN, MobileNet, or DeepFace.  
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

To keep things clean and self-contained, create a Python virtual environment:

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

---

## Usage

From the **face-recognition-fast** folder:

```bash
python main.py --help
```

### Single Method

Choose exactly **one** of the six methods:

- `opencv`
- `mtcnn`
- `facerecognition`
- `mobilenet`
- `arcface`
- `deepface`

**Example**:

```bash
python main.py images/id1.png images/front1.png --method arcface
```

### Benchmark Mode

Compare **all six methods** at once:

```bash
python main.py images/id1.png images/front1.png --benchmark
```

#### Example Output

```
===== BENCHMARK RESULTS =====
Method:          opencv | Status: OK | Time: 0.9981s | MemΔ: 189902848 bytes | Distance: 0.4240 | Similarity: 96.23%
Method:           mtcnn | Status: OK | Time: 2.0303s | MemΔ: 600047616 bytes | Distance: 0.4421 | Similarity: 95.76%
Method: facerecognition | Status: OK | Time: 0.7643s | MemΔ: 15212544 bytes | Distance: 0.4290 | Similarity: 96.18%
Method:       mobilenet | Status: OK | Time: 1.4072s | MemΔ: 38670336 bytes | Distance: 18.9774 | Similarity: 64.26%
Method:         arcface | Status: OK | Time: 2.2136s | MemΔ: 400027648 bytes | Distance: 2.0415 | Similarity: 95.43%
Method:        deepface | Status: OK | Time: 1.7379s | MemΔ: 36253696 bytes | Distance: 3.5570 | Similarity: 58.36%
```

> **What MemΔ Means:**  
> - Higher `MemΔ` means the method consumes **more RAM** during execution.
> - **TensorFlow-based models (MTCNN, MobileNet, ArcFace, DeepFace)** typically use **more memory** than lightweight methods like OpenCV.

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
