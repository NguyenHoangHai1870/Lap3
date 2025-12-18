# Lab 3 – Report

BÁO CÁO THỰC HÀNH LAB 3

Môn học: NLP



Cấu trúc thư mục dự án (Directory Structure)

├── data/                                # Thư mục chứa dữ liệu (Dataset)
│  
├── notebook/                            # Thư mục chứa Jupyter Notebooks (Mã nguồn chính)
│   ├── Lab3_part1.pdf                              
│   ├── Lab3_part2.pdf  
│
├──  lap3.md                            # File báo cáo chi tiết này
│   
├─  Lap3                            # Mã nguồn Python (Modules/Classes tái sử dụng) 
├── .gitignore                           # File cấu hình bỏ qua file rác (tmp, __pycache__)






1. Giải thích các bước thực hiện

Bài thực hành được triển khai qua hai phần chính:

1.1. Sử dụng mô hình embedding pre-trained (GloVe)

(Lap3_part1)

Các bước:

Bước 1 – Cài đặt thư viện

Sử dụng các thư viện:

gensim để tải mô hình GloVe

numpy cho xử lý vector

scikit-learn cho PCA và t-SNE

matplotlib cho trực quan hóa

Bước 2 – Tải mô hình GloVe pre-trained

Bạn tải mô hình glove-wiki-gigaword-100 (100 chiều, 400.000 từ).

Bước 3 – Lấy mẫu từ và trích xuất vector

Lấy 300 từ ngẫu nhiên trong từ điển GloVe

Lấy vector embedding tương ứng (100 chiều)

Bước 4 – Giảm chiều

Hai kỹ thuật giảm chiều được áp dụng:

PCA → 2D

t-SNE → 3D

Bước 5 – Tính cosine similarity và tìm từ gần nghĩa

Cài đặt hàm cosine_similarity() và hàm find_most_similar() để:

Tính độ tương đồng

Tìm top-K từ gần nhất

1.2. Xây dựng lớp WordEmbedder và Bonus Task

(Lap3_part2)

Bước 1 – Thiết kế lớp WordEmbedder

Lớp gồm các hàm:

get_vector(word)

get_similarity(word1, word2)

get_most_similar(word)

embed_document(document) → embedding trung bình từ

Sử dụng mô hình: glove-wiki-gigaword-50

Bước 2 – Bonus Task: huấn luyện Word2Vec thủ công

Có 2 kiểu huấn luyện:

Huấn luyện Word2Vec bằng thư viện gensim trên UD English

Huấn luyện Word2Vec bằng PySpark trên tập C4 (30.000 mẫu)
Bao gồm:

Load dữ liệu .json.gz

Làm sạch văn bản

Tokenize

Train Word2Vec (100 chiều)

Tìm từ đồng nghĩa

Trực quan hóa embedding với PCA

2. Hướng dẫn chạy code
2.1. Chạy notebook Lap3_part1

Mở Google Colab

Tải notebook lên

Chạy lần lượt các cell:

Cell cài thư viện

Tải mô hình GloVe

Chạy PCA và t-SNE

Gọi hàm find_most_similar()

Lưu ý: Colab có thể yêu cầu Restart runtime sau khi cài lại numpy.

2.2. Chạy notebook Lap3_part2

Upload requirements.txt (nếu có)

Chạy cell cài đặt thư viện

Chạy phần WordEmbedder

Test các hàm trong lớp

Với Bonus Task:

Upload file dữ liệu (UD_*.tar.gz hoặc c4-train...json.gz)

Chạy phần đọc dữ liệu → train Word2Vec

Chạy phần visualize PCA

Kết quả hiển thị trực tiếp trong notebook

3. Phân tích kết quả
3.1. Nhận xét về độ tương đồng và các từ đồng nghĩa từ model pre-trained
Từ “king” (GloVe 100D)

Top từ gần nhất (ví dụ từ file) gồm:

“prince”, “queen”, “monarch”, “throne”, “kingdom”

→ Đây là nhóm từ cùng trường nghĩa “hoàng gia – quyền lực”.
→ Vector embedding của GloVe bắt rất tốt quan hệ ngữ nghĩa.

Từ “computer”

“computers”, “software”, “technology”, “desktop”, “hardware”
→ Nhóm hoàn toàn chính xác, thuộc lĩnh vực công nghệ.

Từ “beautiful”

“lovely”, “gorgeous”, “wonderful”, “elegant”, “pretty”
→ Đều là từ miêu tả sắc thái tích cực → rất đúng kỳ vọng.

Kết luận:
Mô hình GloVe pre-trained học được pattern ngữ nghĩa tự nhiên và ổn định.

3.2. Phân tích biểu đồ trực quan hóa
PCA (2D)

(Kết quả từ Lap3_part1.pdf)

Các điểm phân bố rộng, không phân cụm rõ.

PCA giữ phương sai toàn cục nhưng không giữ quan hệ phi tuyến, dẫn đến:

Các từ gần nghĩa đôi khi vẫn cách xa nhau

Các cụm không rõ rệt

→ PCA phù hợp để xem cấu trúc tổng thể nhưng không phản ánh tốt cụm ngữ nghĩa.

t-SNE (3D)

t-SNE cho thấy các cụm rõ hơn:

Ví dụ nhóm từ “king–queen–throne–monarch” gần nhau

Các nhóm từ miêu tả sắc thái (“beautiful–lovely–gorgeous”) cũng gần nhau

t-SNE bảo toàn ngữ nghĩa cục bộ, vì vậy các từ gần nghĩa cluster tốt.

Điểm thú vị:

Một số từ có thể nằm gần nhau do xuất hiện cùng ngữ cảnh, không phải đồng nghĩa
→ ví dụ từ về thể thao - quốc gia - nhân vật chính trị thường gần nhau.

3.3. So sánh model pre-trained và model tự huấn luyện
1. Pre-trained (GloVe)

Huấn luyện trên dữ liệu lớn (Wikipedia + Gigaword)

Vector có chất lượng cao, đa dạng ngữ nghĩa

Tìm đồng nghĩa rất chính xác

2. Model Word2Vec tự huấn luyện (Bonus Task)
Huấn luyện bằng gensim trên UD English

Bộ dữ liệu bé, khoảng vài chục nghìn câu
→ Kết quả tìm từ đồng nghĩa cho “computer” cho ra:

“wheel”, “campaign”, “cruise”, …
→ Không liên quan → chất lượng kém.

Huấn luyện bằng PySpark Word2Vec trên C4 (30.000 câu)

Lượng dữ liệu lớn hơn → kết quả tốt hơn:

“desktop”, “laptop”, “computers”, “usb”
→ Tương đối đúng nhưng vẫn kém hơn GloVe.

Kết luận:

Pre-trained GloVe vượt trội về chất lượng.

Model tự train chỉ tốt khi dữ liệu đủ lớn (hàng trăm triệu token trở lên).

GloVe học tốt quan hệ ngữ nghĩa toàn cục hơn Word2Vec trên tập nhỏ.

4. Khó khăn và giải pháp
1. Xung đột phiên bản thư viện

Trong cả Lap3_part1 và Lap3_part2 đều xuất hiện lỗi khi Colab cài numpy, pandas.
Giải pháp:

Restart runtime sau khi cài

Cố định phiên bản thư viện trong requirements.txt

2. t-SNE chạy lâu, dễ lỗi

Do dữ liệu 300 vector × 100 chiều
Giải pháp:

Giảm số từ

Dùng PCA trước rồi mới t-SNE (PCA→50→TSNE)

3. Training Word2Vec với C4 khá nặng

PySpark yêu cầu bộ nhớ lớn
Giải pháp:

Giảm lượng dữ liệu (lấy 30.000 dòng như bài hiện tại)

Chạy Spark ở server/VM nếu cần nhiều tài nguyên hơn

4. Một số từ không nằm trong từ điển (OOV)

Giải pháp:

Với pre-trained: bỏ qua hoặc thay bằng vector 0

Với Word2Vec: tăng min_count=1 khi huấn luyện (trade-off noise)

5. Trích dẫn tài liệu tham khảo

Jeffrey Pennington, Richard Socher, Christopher D. Manning (2014).
GloVe: Global Vectors for Word Representation.

Mikolov et al. (2013).
Efficient Estimation of Word Representations in Vector Space (Word2Vec).


Documentation của gensim: https://radimrehurek.com/gensim/

