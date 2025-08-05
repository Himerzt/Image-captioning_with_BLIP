#  Image Captioning với BLIP và COCO 2017

##  Giới thiệu

Dự án này nhằm giải quyết bài toán **Image Captioning** – tự động tạo mô tả bằng ngôn ngữ tự nhiên cho một bức ảnh. Đây là một trong những ứng dụng điển hình kết hợp giữa **thị giác máy tính (Computer Vision)** và **xử lý ngôn ngữ tự nhiên (Natural Language Processing)**.

Ví dụ:  
> Ảnh đầu vào là một người đang cưỡi ngựa → Mô hình sẽ sinh ra:  
> **"A man riding a horse in the field."**

---

##  Mô hình sử dụng: `BLIP (Bootstrapped Language-Image Pretraining)`

- Được phát triển bởi Salesforce.
- Học song song cả từ ảnh và văn bản.
- Trong dự án này, BLIP được sử dụng ở chế độ **Image Captioning đã fine-tuned**.

---

##  Dữ liệu: `COCO 2017 Captions`

- Bộ dữ liệu phổ biến trong thị giác máy tính: **Common Objects in Context**.
- Gồm hơn **118,000 ảnh huấn luyện**, mỗi ảnh có khoảng **5 mô tả văn bản**.
- Được dùng để đánh giá độ chính xác khi mô hình sinh caption.

---

##  Pipeline 

1. Chuẩn bị dữ liệu COCO (ảnh + mô tả).
2. Tải mô hình BLIP từ HuggingFace.
3. Dự đoán caption cho ảnh đầu vào.
4. (Tuỳ chọn) So sánh caption dự đoán với caption thật bằng cosine similarity.
5. Hiển thị ảnh + caption bằng `matplotlib`.

---


##  Yêu cầu môi trường

```bash
torch>=1.13
transformers
sentence-transformers
Pillow
matplotlib
tqdm
```



##  Hướng dẫn sử dụng model trên Kaggle

Bạn có thể chạy mô hình trực tiếp trên **Kaggle Notebook** theo các bước sau:

---

### 1️ Mở file `model-result.py` trên Kaggle

- Vào link: [model-result.py trên Kaggle](https://www.kaggle.com/code/himernors/model-result)
- Hoặc clone notebook từ repo này và upload lên Kaggle Notebook.

---

### 2️ Thêm mô hình đã huấn luyện vào Notebook

- Nhấn vào biểu tượng **Add Input** (ở sidebar bên phải).
- Chọn mục **Models**.
- Tìm kiếm theo link:  
  👉 [`himernors/caption_blip_model_final`](https://www.kaggle.com/models/himernors/caption_blip_model_final)
- Thêm model này vào notebook.

---

### 3️ Tải lên hình ảnh bạn muốn tạo caption

- Nhấn vào **Add Input** → **Upload Data**.
- Tải ảnh của bạn lên (ví dụ như `bear.jpg`).
- Ảnh sẽ nằm trong đường dẫn `/kaggle/input/<tên-folder>/tên-ảnh.jpg`.

---

### 4️⃣ Cập nhật đường dẫn ảnh trong notebook

Ở **cell cuối cùng**, bạn chỉnh lại dòng:

```python
img_path = "/kaggle/input/bearwithknife/bear.jpg"  # Đổi thành đường dẫn ảnh của bạn

```

### Lưu ý nhớ chạy các cell phía trước để chuẩn bị dữ liệu model
