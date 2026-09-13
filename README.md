# 🧘 Pilates AI Pose Coach: Real-Time Posture Alignment & Kinematic Feedback

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Engine](https://img.shields.io/badge/ONNX_Runtime-1.16%2B-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Model](https://img.shields.io/badge/Pose_Engine-RTMPose--m_(SimCC)-FF6F00?style=for-the-badge&logo=openaccess&logoColor=white)](https://github.com/open-mmlab/mmpose)
[![Detector](https://img.shields.io/badge/Detector-RTMDet--m-blueviolet?style=for-the-badge)](https://github.com/open-mmlab/mmdetection)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Hệ thống AI thị giác máy tính hỗ trợ nhận diện khung xương, phân tích động học chuyển động và căn chỉnh tư thế tập Pilates theo thời gian thực.**

[Tính Năng Nổi Bật](#-tính-năng-nổi-bật) •
[Kiến Trúc Mô Hình](#-kiến-trúc-mô-hình--công-nghệ) •
[Cài Đặt & Chạy](#-hướng-dẫn-cài-đặt--chạy-thực-nghiệm) •
[Cấu Trúc Thư Mục](#-cấu-trúc-dự-án) •
[Lộ Trình](#-kế-hoạch-phát-triển-roadmap)

</div>

---

## 📌 Giới thiệu dự án (Overview)

Tập luyện **Pilates** và **Home Workout** đòi hỏi sự chuẩn xác cao về mặt **căn chỉnh tư thế (alignment)** và **độ ổn định vùng cơ lõi (core stability)**. Khác với tập gym thông thường (chủ yếu đếm rep tạ), việc tập sai tư thế trong Pilates (như võng thắt lưng, lệch hông, góc gối vượt quá tầm vận động an toàn) có thể gây chấn thương trực tiếp lên đĩa đệm cột sống và các khớp.

Dự án này xây dựng một trợ lý AI thông minh ứng dụng Thị giác máy tính (Computer Vision) chạy trực tiếp trên luồng camera/webcam thông thường hoặc thiết bị di động:
- 🎯 **Ước lượng tư thế người (Human Pose Estimation)** với độ chính xác cao ngay cả ở các tư thế nằm sàn (*Supine, Prone, Side-lying*).
- 📐 **Đo lường góc động học (Kinematics Calculation)**: Tính toán góc mở khớp gối, khớp háng, độ đối xứng cơ thể theo thời gian thực.
- ⚡ **Tối ưu hóa thời gian thực (Edge-ready)**: Hoạt động mượt mà trên CPU thông thường thông qua ONNX Runtime mà không cần GPU rời đắt đỏ.

---

## 🚀 Tính năng nổi bật (Key Features)

- ⚡ **Hiệu năng cao (>90 FPS trên CPU, >150 FPS trên GPU)**: Tối ưu hoá toàn diện bằng ONNX Runtime FP16/FP32.
- 🎯 **Độ chính xác Sub-pixel (SimCC Paradigm)**: Triệt tiêu hiện tượng rung giật khung xương (*Zero-jitter*) khi người tập giữ các tư thế tĩnh (*Isometric Hold* như *Plank, Hundred, Bird Dog*).
- 🧘 **Tối ưu cho tư thế nằm sàn & tự che khuất (Floor Poses & Self-Occlusion)**: Nhận diện chính xác vị trí các khớp chân và tay khi nằm nghiêng đá chân hoặc uốn cong người.
- 📐 **Đo góc động học tự động**: Tính góc khớp theo giải thuật vector dot product tức thời.
- 📱 **Sẵn sàng triển khai thiết bị biên (On-device Edge AI)**: Trọng số mô hình nhẹ (~13.8 MB), dễ dàng tích hợp vào ứng dụng di động Android/iOS.

---

## 🧠 Kiến trúc mô hình & Công nghệ (Model Zoo & Specs)

Hệ thống triển khai pipeline Deep Learning 2 giai đoạn (Top-down Architecture) chạy trên ONNX Runtime:

| Thành phần | Mô hình sử dụng | Kiến trúc kỹ thuật | Vai trò |
| :--- | :--- | :--- | :--- |
| **Human Detector** | **RTMDet-m** / **RTMDet-nano** | CSPNeXt Backbone, Anchor-free | Phát hiện và cắt vùng cơ thể người tập từ khung hình camera. |
| **2D Pose Estimator** | **RTMPose-m** | **SimCC Head** (1D Coordinate Classification) | Trích xuất 17 điểm mốc cơ thể chuẩn COCO (độ chính xác ~75.3% AP). |
| **Inference Engine** | **ONNX Runtime** | ONNX Engine (Hỗ trợ cả CPU & CUDA) | Thực thi suy luận siêu tốc, tự động tải pre-trained weights từ cache. |
| **Kinematics Engine** | Vector Kinematics | Dot Product & Trigonometric Vectorization | Tính toán góc mở khớp gối, góc hông và phát hiện sai lệch form. |

```mermaid
flowchart LR
    A[📹 Video / Webcam Input] --> B[🔍 RTMDet-m<br>Human Detector]
    B --> C[🧘 RTMPose-m<br>SimCC Keypoint Estimator]
    C --> D[📐 Kinematics Engine<br>Joint Angles & Alignment]
    D --> E[🖥️ Real-Time HUD & Visual Feedback]
```

---

## 🛠️ Hướng dẫn cài đặt & Chạy thực nghiệm (Quick Start)

### 1. Clone repository về máy

```bash
git clone https://github.com/dothang13/pilates-ai-pose-coach.git
cd pilates-ai-pose-coach
```

### 2. Cài đặt môi trường (Python 3.9 - 3.11)

Khuyến khích khởi tạo môi trường ảo Python:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường (trên Windows):
.\venv\Scripts\activate

# Cài đặt các thư viện cần thiết:
pip install -r requirements.txt
```

> **Mẹo (Tùy chọn):** Nếu máy bạn có card đồ họa rời NVIDIA và muốn chạy GPU CUDA:
> ```bash
> pip install onnxruntime-gpu
> ```

### 3. Khởi chạy kiểm thử Real-time

```bash
# 1. Chạy mặc định với Webcam máy tính (Model balanced RTMPose-m trên CPU):
python test_rtmpose_m.py

# 2. Chạy với video bài tập có sẵn:
python test_rtmpose_m.py --source data/raw/pilates_sample.mp4

# 3. Chạy chế độ siêu nhẹ (RTMDet-nano + RTMPose-t, >120 FPS):
python test_rtmpose_m.py --mode lightweight

# 4. Chạy trên GPU CUDA (nếu có NVIDIA GPU):
python test_rtmpose_m.py --device cuda
```

Trong lần chạy đầu tiên, thư viện `rtmlib` sẽ tự động tải các file trọng số ONNX đã huấn luyện (~13MB) về bộ nhớ đệm.

---

## 📂 Cấu trúc dự án (Project Structure)

```text
pilates-ai-pose-coach/
│
├── .gitignore               # Cấu hình loại trừ file rác (venv, pycache, onnx weights)
├── LICENSE                  # Giấy phép mã nguồn mở MIT
├── README.md                # Tài liệu hướng dẫn & giới thiệu tổng thể dự án
├── requirements.txt         # Danh sách thư viện phụ thuộc (rtmlib, onnxruntime,...)
├── test_rtmpose_m.py        # Mã nguồn chạy Pose Estimation & trích xuất góc khớp
│
└── data/                    # Thư mục quản lý dữ liệu
    ├── raw/                 # Nơi chứa video bài tập thô (.gitkeep)
    └── processed/           # Nơi lưu trữ tọa độ keypoints JSON/NPY (.gitkeep)
```

---

## 🗓️ Kế hoạch phát triển (Roadmap)

- [x] **Giai đoạn 1**: Xây dựng Baseline 2D Pose Estimation với RTMPose-m qua ONNX Runtime.
- [x] **Giai đoạn 2**: Tích hợp module tính toán góc động học các khớp chính (Gối, Hông).
- [ ] **Giai đoạn 3**: Mở rộng nhận diện bộ 26 keypoints (WholeBody) để bắt kỹ thuật chuyển động mu bàn chân (Pointed/Flexed toes).
- [ ] **Giai đoạn 4**: Xây dựng thuật toán nhận diện pha bài tập và đếm Rep tự động (Dynamic Time Warping / Phase Segmentation).
- [ ] **Giai đoạn 5**: Tích hợp cảnh báo giọng nói (Text-to-Speech) khi phát hiện tư thế sai.
- [ ] **Giai đoạn 6**: Đóng gói hoàn chỉnh thành ứng dụng di động (On-device Edge AI Mobile App).

---

## 👥 Đóng góp & Bản quyền (License)

Dự án được phát hành dưới giấy phép mã nguồn mở [MIT License](LICENSE).
Mọi đóng góp, báo cáo lỗi (Issue) hoặc đề xuất tính năng (Pull Request) đều rất được hoan nghênh!
