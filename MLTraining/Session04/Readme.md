## Coder: Nguyễn Huy Hoàng
## Sử dụng mô hình RNN để phân loại bài viết

Tensorflow version 2.0

1. Thiết lập khối lượng train ``` max_step training = 5000 ``` (sau epoch thứ 5 thì độ chính xác không thay đổi nhiều)
1. Các file encoded đã được tạo sẵn trong ``` datasets/w2v ```
1. Nếu muốn encode lại thì tải dữ liệu về từ http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz sau đó uncomment dòng 326,327,328.
1. Mô hình tổng quát của các layer và kết quả được vẽ ở file Model.png
