# 📰 Fake News Detection using Machine Learning & NLP

Dự án ứng dụng **Xử lý ngôn ngữ tự nhiên (NLP)** và **Học máy (Machine Learning)** nhằm phát hiện và phân loại tin tức giả mạo (**Fake News**) và tin tức thật (**Real News**), tích hợp giao diện Web tương tác thời gian thực bằng **Flask**.

---

## 📌 Mục lục
- [1. Giới thiệu tổng quan](#1-giới-thiệu-tổng-quan)
- [2. Cấu trúc thư mục dự án](#2-cấu-trúc-thư-mục-dự-án)
- [3. Bộ dữ liệu (Dataset)](#3-bộ-dữ-liệu-dataset)
- [4. Quy trình xử lý NLP Pipeline & Công thức TF-IDF](#4-quy-trình-xử-lý-nlp-pipeline--công-thức-tf-idf)
- [5. Cơ sở lý thuyết & Công thức toán học của các mô hình](#5-cơ-sở-lý-thuyết--công-thức-toán-học-của-các-mô-hình)
  - [5.1. Passive-Aggressive Classifier (Mô hình chính)](#51-passive-aggressive-classifier-mô-hình-chính)
  - [5.2. Multinomial Naive Bayes (Mô hình đối chứng)](#52-multinomial-naive-bayes-mô-hình-đối-chứng)
  - [5.3. Công thức tính độ tin cậy (Confidence Score)](#53-công-thức-tính-độ-tin-cậy-confidence-score)
- [6. Các chỉ số đo lường & Đánh giá hiệu năng](#6-các-chỉ-số-đo-lường--đánh-giá-hiệu-năng)
- [7. Ứng dụng Web (Flask App)](#7-ứng-dụng-web-flask-app)
- [8. Hướng dẫn cài đặt & Thực thi](#8-hướng-dẫn-cài-đặt--thực-thi)
- [9. Yêu cầu môi trường & Thư viện](#9-yêu-cầu-môi-trường--thư-viện)

---

## 1. Giới thiệu tổng quan

Tin giả (Fake News) lan truyền với tốc độ nhanh chóng trên không gian mạng và gây ra nhiều hệ lụy cho xã hội. Mục tiêu cốt lõi của dự án là xây dựng một hệ thống phân loại văn bản tự động có độ tin cậy cao, kết hợp giữa mô hình học trực tuyến tốc độ cao (**Passive-Aggressive Classifier**) và mô hình xác suất thống kê cơ sở (**Multinomial Naive Bayes**).

Hệ thống bao gồm chu trình khép kín:
$$\text{Raw Text} \xrightarrow{\text{NLP Preprocessing}} \text{Tokens / Lemmatized Words} \xrightarrow{\text{TF-IDF Matrix}} \text{Classifier (PAC / MNB)} \xrightarrow{\text{Inference}} \text{Real / Fake + Confidence}$$

---

## 2. Cấu trúc thư mục dự án

```text
nlp/
├── Fake_News_Detector-checkpoint.ipynb     # Jupyter Notebook huấn luyện & đối sánh mô hình
├── Fake_News_Detector-checkpoint.ipynb.bak # Bản sao lưu an toàn của Notebook
├── train.csv                                 # Tập dữ liệu huấn luyện tin tức (~98MB)
├── model.pkl                                 # Trọng số mô hình Passive-Aggressive đã lưu
├── vector.pkl                                # Bộ từ điển và ma trận trọng số TfidfVectorizer
├── webapp-checkpoint.py                      # Ứng dụng Web Flask hoàn chỉnh (Port 8080)
├── app.py                                    # Mã nguồn Flask khởi tạo ban đầu
├── index.html                                # Giao diện Trang chủ (Landing page)
├── prediction.html                           # Giao diện Trang kiểm tra tin tức
├── static/                                   # Tài nguyên tĩnh
│   ├── hero_img.svg                          # Minh họa vector giao diện
│   ├── icons8-facebook-ios-16-filled-32.png  # Favicon icon 32px
│   └── icons8-facebook-ios-16-filled-96.png  # Favicon icon 96px
└── README.md                                 # Tài liệu chi tiết kỹ thuật của dự án
```

---

## 3. Bộ dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu **Fake News Dataset** chuẩn gồm hàng chục nghìn bài báo với các trường dữ liệu:
- `id`: Định danh duy nhất của mỗi bài viết.
- `title`: Tiêu đề bài viết.
- `author`: Tác giả hoặc cơ quan phát hành bài báo.
- `text`: Toàn văn nội dung bài báo (đặc trưng văn bản đầu vào chính).
- `label`: Nhãn phân loại nhị phân:
  - $y = 1$: **Fake News** (Tin giả mạo).
  - $y = 0$: **Real News** (Tin tức xác thực).

---

## 4. Quy trình xử lý NLP Pipeline & Công thức TF-IDF

### 4.1. Tiền xử lý văn bản (NLP Preprocessing)
Mỗi văn bản thô $d$ trải qua các bước chuẩn hóa sau:
1. **Lọc ký tự nhiễu**: Sử dụng biểu thức chính quy (Regex) `re.sub(r'[^a-zA-Z\s]', '', text)` để chỉ giữ lại các ký tự chữ cái và khoảng trắng.
2. **Chuyển chữ thường (Lowercasing)**: Đồng nhất từ vựng, tránh phân biệt chữ hoa/thường (ví dụ: `Breaking` $\rightarrow$ `breaking`).
3. **Tách từ (Tokenization)**: Phân rã chuỗi văn bản thành danh sách các từ đơn (tokens) nhờ bộ tách từ `nltk.word_tokenize`.
4. **Loại bỏ từ dừng (Stopwords Removal)**: Loại bỏ các từ chức năng phổ biến trong tiếng Anh không mang giá trị phân loại (`the`, `is`, `at`, `which`,... từ danh sách `stopwords.words('english')`).
5. **Đưa từ về nguyên thể (Lemmatization)**: Sử dụng `WordNetLemmatizer` ánh xạ các dạng biến thể của từ về từ điển gốc (lemma), ví dụ: `running` $\rightarrow$ `run`, `better` $\rightarrow$ `good`.

---

### 4.2. Công thức toán học của TF-IDF Vectorization

Mô hình chuyển đổi văn bản thành vector số học bằng phương pháp **TF-IDF (Term Frequency - Inverse Document Frequency)**:

#### a) Tần suất xuất hiện của từ (Term Frequency - TF)
Đo lường mức độ xuất hiện thường xuyên của từ $t$ trong văn bản $d$:
$$\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
*Trong đó:*
- $f_{t, d}$: Số lần từ khóa $t$ xuất hiện trong văn bản $d$.
- $\sum_{t' \in d} f_{t', d}$: Tổng số lượng từ trong văn bản $d$.

#### b) Nghịch đảo tần số văn bản (Inverse Document Frequency - IDF)
Đánh giá mức độ quan trọng/hiếm của từ $t$ trên toàn bộ tập ngữ liệu $D$ gồm $N$ tài liệu. Để tránh lỗi chia cho $0$ và làm mượt giá trị (Smooth IDF theo chuẩn Scikit-learn):
$$\text{IDF}(t, D) = \ln\left( \frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$
*Trong đó:*
- $N = |D|$: Tổng số văn bản trong tập ngữ liệu huấn luyện.
- $\text{DF}(t) = |\{ d \in D : t \in d \}|$: Số lượng văn bản có chứa từ $t$.
- Nếu từ $t$ xuất hiện ở hầu hết các tài liệu $\implies \text{IDF}$ thấp (từ phổ biến). Ngược lại, từ hiếm sẽ có $\text{IDF}$ cao.

#### c) Trọng số TF-IDF tổng hợp
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

#### d) Chuẩn hóa vector theo chuẩn Euclidean ($L_2$ Normalization)
Để loại bỏ sự chênh lệch do độ dài văn bản dài/ngắn khác nhau, vector đặc trưng $\mathbf{v} = [v_1, v_2, \dots, v_M]$ được chuẩn hóa:
$$\mathbf{x} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{j=1}^{M} v_j^2}} \quad \implies \quad \|\mathbf{x}\|_2 = 1$$

---

## 5. Cơ sở lý thuyết & Công thức toán học của các mô hình

---

### 5.1. Passive-Aggressive Classifier (Mô hình chính)

Mô hình **Passive-Aggressive (PA)** (Crammer et al., 2006) là thuật toán phân loại dựa trên biên (margin-based) thuộc nhóm **Học trực tuyến (Online Learning)**, đặc biệt tối ưu cho dữ liệu văn bản chiều lớn và ma trận thưa.

#### a) Hàm quyết định phân loại (Decision Rule)
Cho vector đặc trưng $\mathbf{x} \in \mathbb{R}^M$ và vector trọng số mô hình $\mathbf{w} \in \mathbb{R}^M$, nhãn nhị phân $y \in \{-1, +1\}$ (hoặc ánh xạ $\{0, 1\}$):
$$\hat{y} = \text{sign}(\mathbf{w}^T \mathbf{x})$$

#### b) Hàm mất mát bản lề (Hinge Loss)
Tại bước lặp $t$, với mẫu huấn luyện $(\mathbf{x}_t, y_t)$, tổn thất được định nghĩa bởi hàm Hinge Loss:
$$\ell_t = \ell(\mathbf{w}_t; (\mathbf{x}_t, y_t)) = \max\Big(0, \, 1 - y_t (\mathbf{w}_t^T \mathbf{x}_t)\Big)$$

- **Trường hợp 1 ($\ell_t = 0 \iff y_t(\mathbf{w}_t^T \mathbf{x}_t) \ge 1$):**
  Mô hình phân loại hoàn toàn chính xác với khoảng cách lề (margin) an toàn tối thiểu là 1. Thuật toán giữ nguyên trọng số:
  $$\mathbf{w}_{t+1} = \mathbf{w}_t \quad \textbf{(Passive - Thụ động)}$$
- **Trường hợp 2 ($\ell_t > 0 \iff y_t(\mathbf{w}_t^T \mathbf{x}_t) < 1$):**
  Mô hình phân loại sai hoặc biên độ tin cậy chưa đạt chuẩn. Thuật toán ngay lập tức cập nhật trọng số để triệt tiêu tổn thất:
  $$\mathbf{w}_{t+1} \neq \mathbf{w}_t \quad \textbf{(Aggressive - Quyết liệt)}$$

#### c) Bài toán tối ưu hóa có ràng buộc (Optimization Problem)
Mục tiêu là tìm $\mathbf{w}_{t+1}$ vừa gần $\mathbf{w}_t$ nhất (để không quên tri thức đã học trước đó), vừa thỏa mãn phân loại đúng mẫu hiện tại với lề ít nhất là 1:
$$\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \frac{1}{2} \|\mathbf{w} - \mathbf{w}_t\|_2^2 \quad \text{sao cho} \quad \ell(\mathbf{w}; (\mathbf{x}_t, y_t)) = 0 \iff y_t(\mathbf{w}^T \mathbf{x}_t) \ge 1$$

Sử dụng hàm nhân tử Lagrange:
$$\mathcal{L}(\mathbf{w}, \tau) = \frac{1}{2}\|\mathbf{w} - \mathbf{w}_t\|_2^2 + \tau \Big(1 - y_t(\mathbf{w}^T \mathbf{x}_t)\Big)$$
Lấy đạo hàm riêng theo $\mathbf{w}$ và giải điều kiện KKT (Karush-Kuhn-Tucker):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \mathbf{w}_t - \tau y_t \mathbf{x}_t = 0 \implies \mathbf{w} = \mathbf{w}_t + \tau y_t \mathbf{x}_t$$

Thay ngược vào phương trình biên $y_t(\mathbf{w}^T \mathbf{x}_t) = 1$, ta thu được nghiệm giải tích dạng đóng (Closed-form solution):

#### d) Quy tắc cập nhật trọng số (Weight Update Rule)
$$\mathbf{w}_{t+1} = \mathbf{w}_t + \tau_t y_t \mathbf{x}_t$$

Trong đó, bước học thích ứng $\tau_t$ (Adaptive Step Size) cho các biến thể được xác định như sau:
1. **Thuật toán PA chuẩn (Hard-margin):**
   $$\tau_t = \frac{\ell_t}{\|\mathbf{x}_t\|_2^2}$$
2. **Thuật toán PA-I (Soft-margin với hệ số kiểm soát $C$):**
   $$\tau_t = \min\left( C, \, \frac{\ell_t}{\|\mathbf{x}_t\|_2^2} \right)$$
3. **Thuật toán PA-II (Mất mát toàn phương):**
   $$\tau_t = \frac{\ell_t}{\|\mathbf{x}_t\|_2^2 + \frac{1}{2C}}$$

*(Trong Scikit-learn `PassiveAggressiveClassifier`, mô hình mặc định triển khai PA-I với hàm mất mát `hinge` và tham số điều hòa $C=1.0$).*

---

### 5.2. Multinomial Naive Bayes (Mô hình đối chứng)

**Multinomial Naive Bayes (MNB)** là thuật toán phân loại xác suất dựa trên định lý Bayes, được thiết kế chuyên biệt cho dữ liệu văn bản biểu diễn theo phân phối đa thức (Multinomial Distribution) của các từ.

#### a) Định lý Bayes tổng quát
Xác suất hậu nghiệm (Posterior Probability) của văn bản $\mathbf{x} = (x_1, x_2, \dots, x_M)$ thuộc về nhãn $y \in \{0, 1\}$:
$$P(y \mid \mathbf{x}) = \frac{P(y) \, P(\mathbf{x} \mid y)}{P(\mathbf{x})}$$

Vì $P(\mathbf{x})$ là hằng số chuẩn hóa với mọi lớp $y$, nên ta có quan hệ đồng dạng:
$$P(y \mid \mathbf{x}) \propto P(y) \, P(\mathbf{x} \mid y)$$

#### b) Giả định độc lập có điều kiện (Conditional Independence Assumption)
Giả sử sự xuất hiện của các từ là độc lập với nhau khi biết trước nhãn văn bản $y$:
$$P(\mathbf{x} \mid y) = P(x_1, x_2, \dots, x_M \mid y) = \prod_{i=1}^{M} P(w_i \mid y)^{x_i}$$
*Trong đó $x_i$ là trọng số TF-IDF của từ $w_i$ trong văn bản $\mathbf{x}$.*

#### c) Quy tắc quyết định MAP (Maximum A Posteriori)
Chuyển sang không gian Log-likelihood để tránh hiện tượng tràn số dưới (arithmetic underflow):
$$\hat{y} = \arg\max_{y \in \{0, 1\}} \left[ \ln P(y) + \sum_{i=1}^{M} x_i \ln P(w_i \mid y) \right]$$

#### d) Ước lượng tham số & Kỹ thuật làm mịn Laplace ($\alpha$-Smoothing)
- **Xác suất tiên nghiệm của nhãn (Prior Probability):**
  $$\hat{P}(y) = \frac{N_y}{N}$$
  *(Với $N_y$ là số lượng tài liệu thuộc nhãn $y$, $N$ là tổng số tài liệu trong tập huấn luyện).*

- **Xác suất có điều kiện của từ vựng $P(w_i \mid y)$:**
  Nếu một từ trong tập kiểm tra chưa từng xuất hiện ở nhãn $y$ trong tập huấn luyện, $P(w_i \mid y) = 0$, dẫn đến toàn bộ tích xác suất bị triệt tiêu bằng 0 (**Zero Probability Trap**). Để khắc phục, ta áp dụng công thức làm mịn Laplace:
  $$\hat{P}(w_i \mid y) = \frac{N_{y, i} + \alpha}{N_y + \alpha |V|}$$
  *Trong đó:*
  - $N_{y, i} = \sum_{d \in y} x_{d, i}$: Tổng trọng số TF-IDF của từ khóa $w_i$ trong tất cả tài liệu thuộc nhãn $y$.
  - $N_y = \sum_{i=1}^{|V|} N_{y, i}$: Tổng trọng số TF-IDF của tất cả các từ trong nhãn $y$.
  - $|V|$: Tổng số lượng từ vựng độc nhất trong không gian từ điển (Vocabulary Size).
  - $\alpha$: Tham số làm mịn (Laplace smoothing parameter). Thường chọn $\alpha = 1.0$ (hoặc Lidstone smoothing nếu $0 < \alpha < 1$).

---

### 5.3. Công thức tính độ tin cậy (Confidence Score)

Trong ứng dụng Web (`webapp-checkpoint.py`), mô hình tính toán giá trị khoảng cách hình học tới siêu phẳng phân tách thông qua hàm `decision_function`:
$$z = \mathbf{w}^T \mathbf{x} + b$$

Khoảng cách biên tuyệt đối $|z|$ biểu thị mức độ tự tin của quyết định phân loại. Giá trị này được ánh xạ về miền xác suất $[0\%, 100\%]$ bằng hàm Logistic Sigmoid:
$$\text{Confidence (\%)} = \sigma(|z|) \times 100\% = \frac{1}{1 + e^{-|z|}} \times 100\%$$
- Nếu $|z| \approx 0$ (nằm sát ranh giới quyết định giữa Real và Fake) $\implies \text{Confidence} \approx 50\%$ (không chắc chắn).
- Nếu $|z| \gg 0$ (nằm rất sâu về phía miền tin tức Fake hoặc Real) $\implies \text{Confidence} \rightarrow 100\%$.

---

## 6. Các chỉ số đo lường & Đánh giá hiệu năng

### 6.1. Ma trận nhầm lẫn (Confusion Matrix)

| Nhãn thực tế \ Nhãn dự đoán | Dự đoán: Real ($0$) | Dự đoán: Fake ($1$) |
| :---: | :---: | :---: |
| **Thực tế: Real ($0$)** | **True Negative (TN)** | **False Positive (FP)** |
| **Thực tế: Fake ($1$)** | **False Negative (FN)** | **True Positive (TP)** |

- **True Positive (TP)**: Tin giả dự đoán chính xác là Fake.
- **True Negative (TN)**: Tin thật dự đoán chính xác là Real.
- **False Positive (FP)**: Báo động nhầm (Tin thật bị gán nhãn là Fake).
- **False Negative (FN)**: Bỏ lọt (Tin giả bị phán đoán nhầm là Real - *rất nguy hiểm*).

---

### 6.2. Công thức các chỉ số đánh giá

#### a) Độ chính xác tổng thể (Accuracy)
Tỷ lệ các mẫu được dự đoán đúng trên toàn bộ tập kiểm tra:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

#### b) Độ chuẩn xác (Precision)
Tỷ lệ các tin được dự đoán là Fake mà thực sự là tin Fake:
$$\text{Precision} = \frac{TP}{TP + FP}$$

#### c) Độ phủ / Độ nhạy (Recall / Sensitivity)
Tỷ lệ các tin Fake thực tế được mô hình phát hiện thành công:
$$\text{Recall} = \frac{TP}{TP + FN}$$

#### d) Điểm F1 (F1-Score)
Trung bình điều hòa giữa Precision và Recall, phản ánh sự cân bằng của mô hình khi tập dữ liệu có sự chênh lệch:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

---

### 6.3. Bảng so sánh thực nghiệm 2 mô hình (trên tập Test 20%)

| Tiêu chí | Passive-Aggressive Classifier (PAC) | Multinomial Naive Bayes (MNB) | Nhận xét phân tích |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **~93.5% - 95.8%** | **~88.0% - 91.2%** | PAC vượt trội ~4 - 5% nhờ cơ chế tối ưu biên lề lớn (margin maximization). |
| **Weighted Precision** | **~94%** | **~90%** | PAC kiểm soát tốt tỷ lệ báo động giả (False Positive thấp). |
| **Weighted Recall** | **~94%** | **~88%** | PAC nhận diện đầy đủ hơn các dạng tin tức giả mạo tinh vi. |
| **Weighted F1-Score** | **~94%** | **~89%** | Điểm F1 của PAC ổn định và đạt sự cân bằng tối ưu hơn. |
| **Tốc độ huấn luyện** | Rất nhanh (Online SGD) | Rất nhanh (Closed-form) | Cả hai mô hình đều có tốc độ tính toán tính bằng giây. |

---

## 7. Ứng dụng Web (Flask App)

Ứng dụng web được xây dựng nhằm phục vụ người dùng cuối với giao diện trực quan:
- **Trang chủ (`index.html`)**: Giới thiệu công nghệ NLP và giải pháp xác minh tính xác thực của tin tức, thiết kế theo phong cách hiện đại với Bootstrap 5 & Tailwind CSS.
- **Trang phân tích (`prediction.html`)**:
  - Cho phép người dùng nhập tiêu đề hoặc đoạn văn bản bài báo bất kỳ.
  - Chu trình tiền xử lý tức thời: làm sạch, tokenize, lemmatize và chuyển hóa sang TF-IDF vector.
  - Trả về kết quả phân định:
    - `Looking Fake News ⚠ 📰` (nếu nhãn $1$).
    - `Looking Real News 📰` (nếu nhãn $0$).
  - Hiển thị trực tiếp **Độ tin cậy (%)** ước lượng từ hàm biên quyết định.

---

## 8. Hướng dẫn cài đặt & Thực thi

### Bước 1: Điều hướng vào thư mục dự án
```bash
cd d:\nlp
```

### Bước 2: Cài đặt các thư viện cần thiết
```bash
pip install flask pandas numpy scikit-learn seaborn matplotlib nltk
```

### Bước 3: Tải dữ liệu từ vựng NLTK (WordNet, Stopwords, Punkt)
Khởi chạy lệnh Python nhanh để tải tài nguyên NLP:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Bước 4: Khởi chạy Web Server Flask
```bash
python webapp-checkpoint.py
```
Sau khi server chạy, truy cập vào trình duyệt web tại đường dẫn:
```text
http://127.0.0.1:8080/
```

### Bước 5: Huấn luyện lại và kiểm tra mô hình trong Notebook
Mở notebook bằng Jupyter:
```bash
jupyter notebook Fake_News_Detector-PA-checkpoint.ipynb
```
Chạy tuần tự các ô lệnh để:
1. Tiền xử lý dữ liệu và tạo ma trận TF-IDF.
2. Huấn luyện đồng thời mô hình **Passive-Aggressive** và **Multinomial Naive Bayes**.
3. Hiển thị ma trận nhầm lẫn (`Confusion Matrix`) và biểu đồ cột đối sánh 4 chỉ số.
4. Tự động xuất các tệp pickle `model.pkl` và `vector.pkl`.

---

## 9. Yêu cầu môi trường & Thư viện

- **Hệ điều hành**: Windows 10/11, Linux, macOS.
- **Phiên bản Python**: Khuyến nghị Python 3.8 – 3.11.
- **Các gói thư viện chính**:
  - `scikit-learn`: Xây dựng vectorizer TF-IDF, bộ phân loại PAC và MultinomialNB, các hàm đánh giá metrics.
  - `nltk`: Thư viện xử lý văn bản tự nhiên (tách từ, loại stopwords, lemmatization).
  - `flask`: Framework backend phục vụ giao diện Web.
  - `pandas` & `numpy`: Thao tác dữ liệu ma trận và cấu trúc bảng.
  - `matplotlib` & `seaborn`: Trực quan hóa ma trận nhầm lẫn và biểu đồ đánh giá.
  - `pickle`: Tuần tự hóa (serialization) và nạp mô hình đã huấn luyện.

---

## 📄 Tài liệu tham khảo
1. **Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., & Singer, Y. (2006)**. *Online passive-aggressive algorithms*. Journal of Machine Learning Research, 7, 551-585.
2. **Manning, C. D., Raghavan, P., & Schütze, H. (2008)**. *Introduction to Information Retrieval*. Cambridge University Press.
3. **Scikit-Learn Documentation**: *Passive Aggressive Algorithms* & *Naive Bayes Models*.
