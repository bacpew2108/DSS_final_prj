# Báo Cáo Chi Tiết Phần Xử Lý Dữ Liệu

## 1. Mục Tiêu

Mục tiêu của giai đoạn xử lý dữ liệu là biến bộ dữ liệu laptop thô thành một bộ dữ liệu sạch, đồng nhất và đủ thông tin để phục vụ phân tích, trực quan hóa và xếp hạng bằng các phương pháp như AHP - TOPSIS.

Quá trình xử lý hiện được triển khai trong [modules/data_processing.py](modules/data_processing.py#L1) và được gọi trực tiếp khi chạy pipeline tạo file `laptops_dataset_cleaned.csv`.

## 2. Tổng Quan Pipeline

Pipeline xử lý dữ liệu gồm các bước chính sau:

1. Đọc dữ liệu gốc từ `data/laptops_dataset_raw.csv`.
2. Chuẩn hóa tên cột về dạng không dấu, chữ thường, dùng dấu gạch dưới.
3. Tách và chuẩn hóa các trường kỹ thuật như CPU, GPU, RAM, storage, weight, display resolution.
4. Gán điểm benchmark cho CPU và GPU từ NanoReview.
5. Lấp các giá trị thiếu theo quy tắc có kiểm soát.
6. Xuất ra file `data/laptops_dataset_cleaned.csv`.

Phần gọi pipeline nằm ở [modules/data_processing.py](modules/data_processing.py#L662).

## 3. Chuẩn Hóa Cấu Trúc Dữ Liệu

### 3.1 Chuẩn hóa tên cột

Hàm `normalize_col_name()` chuyển tên cột về dạng chuẩn bằng cách:

- loại bỏ khoảng trắng đầu cuối,
- chuyển về chữ thường,
- bỏ dấu tiếng Việt,
- thay mọi ký tự không phải chữ/số bằng dấu `_`.

Mục đích là giúp các cột có thể được truy cập nhất quán trong toàn bộ pipeline.

### 3.2 Đảm bảo các cột quan trọng luôn tồn tại

Nếu cột `processor` hoặc `video_graphics` không có trong dữ liệu sau khi chuẩn hóa, pipeline sẽ tạo cột rỗng tương ứng để các bước sau không bị lỗi.

## 4. Trích Xuất Và Chuẩn Hóa Thông Tin Kỹ Thuật

### 4.1 CPU và GPU benchmark

Pipeline lấy bảng benchmark từ NanoReview để gán điểm:

- `cpu_point`
- `gpu_point`

CPU được so khớp bằng fuzzy matching trên tên bộ xử lý. GPU được so khớp bằng nhiều tầng logic hơn: substring match, token overlap, rồi mới tới fuzzy matching.

Kết quả là dữ liệu không chỉ sạch về mặt hình thức mà còn có thêm tiêu chí định lượng để so sánh laptop.

### 4.2 RAM và storage

`compute_ram_capacity()` trích dung lượng RAM từ chuỗi mô tả RAM.

`compute_storage_capacity()` trích dung lượng lưu trữ từ cột `hard_drive`, hỗ trợ các dạng như:

- `512GB`
- `1 TB`
- `2 x 512GB`
- `256GB SSD + 1TB HDD`

### 4.3 Cân nặng

Hàm `parse_weight_kg()` cố gắng nhận diện và chuẩn hóa cân nặng về đơn vị kilogram bằng các luật sau:

- nhận diện trực tiếp các mẫu như `2.3 kg`, `2.3kg`;
- nhận diện đơn vị gram như `2300 g` và đổi sang kg;
- nếu không có đơn vị rõ ràng, dùng giá trị số hợp lý đầu tiên trong khoảng cân nặng thực tế.

Sau đó `compute_weight()` chuẩn hóa toàn bộ cột `weight` theo chuỗi xử lý:

1. parse trực tiếp từng ô;
2. điền theo trung vị của cùng `product_name`;
3. điền theo trung vị của cùng `brand`;
4. điền bằng trung vị toàn cục nếu vẫn còn thiếu.

Chi tiết triển khai nằm ở [modules/data_processing.py](modules/data_processing.py#L335).

### 4.4 Màn hình

Hai thuộc tính được tách từ mô tả màn hình:

- `display_resolution`
- `display_refresh_rate`

Các hàm `extract_display_resolution()` và `extract_display_refresh_rate()` dùng regex để trích ra độ phân giải và tần số quét từ chuỗi mô tả display.

## 5. Chiến Lược Xử Lý Giá Trị Thiếu

Đây là phần quan trọng nhất của pipeline vì bộ dữ liệu có nhiều cột mô tả còn trống.

### 5.1 Lấp theo suy luận ngữ nghĩa

Một số trường được suy luận trực tiếp từ thông tin liên quan:

- `brand` được suy ra từ `product_name` khi bị trống.
- `video_graphics` được suy ra từ `processor` hoặc tên máy trong những trường hợp đặc biệt.
- `video_graphics_memory` được suy ra từ chính chuỗi GPU.
- `display_resolution` và `display_refresh_rate` được suy ra từ chuỗi mô tả màn hình.

### 5.2 Lấp theo nhóm cùng model hoặc cùng hãng

Với các cột dạng mô tả như:

- `operating_system`
- `keyboard`
- `battery`
- `webcam`
- `connections`
- `dimensions`
- `display`

pipeline ưu tiên điền theo:

1. mode của cùng `product_name`;
2. mode của cùng `brand`;
3. giá trị dự phòng `Unknown`.

Chiến lược này giúp giữ lại dữ liệu thay vì xóa dòng, đồng thời hạn chế việc điền bừa theo một hằng số chung.

### 5.3 Quy tắc cho các trường đặc biệt

- `colors` nếu thiếu sẽ được điền là `Black`.
- `price` được chuyển sang VND và nếu không đọc được thì về `0`.
- `weight` không bị xóa dòng nữa mà được lấp theo chuỗi trung vị như đã mô tả ở trên.

## 6. Lý Do Chọn Cách Làm Này

### 6.1 Không xóa hàng tràn lan

Một số cột như `video_graphics_memory` hoặc `display_refresh_rate` thiếu khá nhiều. Nếu xóa hàng, bộ dữ liệu sẽ mất nhiều mẫu quan trọng và có thể làm lệch phân phối theo thương hiệu hoặc phân khúc máy.

### 6.2 Ưu tiên suy luận từ ngữ cảnh gần nhất

Ví dụ:

- nếu `product_name` đã cho biết rõ model, cùng model thường có cấu hình rất giống nhau;
- nếu `display` đã chứa thông tin độ phân giải hoặc refresh rate, không cần để trống;
- nếu `video_graphics` là GPU tích hợp hay dòng phổ biến, có thể suy ra loại bộ nhớ hợp lý.

### 6.3 Bảo toàn dữ liệu cho phân tích TOPSIS

Bộ dữ liệu sau khi làm sạch được dùng cho phân tích quyết định. Vì vậy, các cột đầu vào cần:

- không trống,
- có dạng nhất quán,
- đủ để so sánh giữa các máy.

## 7. Kết Quả Sau Xử Lý

Sau khi chạy pipeline, file `data/laptops_dataset_cleaned.csv` được tạo lại với các cột quan trọng đã được lấp đầy. Các trường mô tả hiện không còn ô trống trong nhóm cột đã nêu, giúp dữ liệu sẵn sàng cho bước phân tích tiếp theo.

## 8. Hạn Chế Và Lưu Ý

1. Một số giá trị được gán `Unknown` khi không đủ ngữ cảnh để suy luận an toàn.
2. Dữ liệu benchmark lấy từ NanoReview phụ thuộc vào kết nối mạng và cấu trúc trang web bên ngoài.
3. Một số chuỗi mô tả sản phẩm rất không đồng nhất nên regex chỉ xử lý tốt ở mức phần lớn trường hợp, không thể tuyệt đối chính xác.

## 9. Kết Luận

Phần xử lý dữ liệu hiện tại tập trung vào ba mục tiêu chính:

- chuẩn hóa dữ liệu thô,
- lấp khuyết có kiểm soát,
- tạo thêm các thuộc tính định lượng phục vụ phân tích.

Nhờ đó, bộ dữ liệu cuối cùng phù hợp hơn để dùng trong hệ hỗ trợ ra quyết định laptop và các bước trực quan hóa, xếp hạng về sau.