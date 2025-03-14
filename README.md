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
+-----------------+----------+-----------+--------------+----------+------------+--------------+
| Method          | Status   |   Time(s) |   Memory(MB) |   CPU(%) |   Distance | Similarity   |
+=================+==========+===========+==============+==========+============+==============+
| opencv          | OK       |    1.0056 |       212.3  |    27.09 |     0.424  | 96.23%       |
+-----------------+----------+-----------+--------------+----------+------------+--------------+
| mtcnn           | OK       |    2.7672 |       562.93 |    11.34 |     0.4499 | 95.60%       |
+-----------------+----------+-----------+--------------+----------+------------+--------------+
| facerecognition | OK       |    0.7948 |        20.86 |    20.12 |     0.429  | 96.18%       |
+-----------------+----------+-----------+--------------+----------+------------+--------------+
| mobilenet       | OK       |    1.8644 |        50.74 |     8.11 |    18.7707 | 65.85%       |
+-----------------+----------+-----------+--------------+----------+------------+--------------+
| arcface         | OK       |    1.7279 |       688.3  |    18.55 |     2.1713 | 94.79%       |
+-----------------+----------+-----------+--------------+----------+------------+--------------+
| deepface        | OK       |    2.3077 |        30.41 |    13.59 |  2986.94   | 93.25%       |
+-----------------+----------+-----------+--------------+----------+------------+--------------+
```

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
