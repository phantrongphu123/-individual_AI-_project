# 8-Puzzle AI Solver (Visualizer)

Dự án này là một chương trình Python sử dụng thư viện Pygame để giải và trực quan hóa các bước giải của bài toán 8-puzzle cổ điển bằng nhiều thuật toán trí tuệ nhân tạo khác nhau.

## Tính năng

*   **Giao diện đồ họa (GUI):** Hiển thị lưới 8-puzzle, bảng điều khiển với các nút thuật toán và khu vực hiển thị kết quả/thông báo theo layout 3 cột.
*   **Nhập liệu thủ công:** Người dùng có thể click vào các ô trên lưới để tự tạo trạng thái bắt đầu cho bài toán.
*   **Tạo trạng thái ngẫu nhiên:**
    *   Tự động tạo trạng thái ban đầu ngẫu nhiên và *có thể giải được* cho một số thuật toán (Genetic Algorithm).
    *   Trực quan hóa quá trình tạo bảng ngẫu nhiên khi nhấn nút "Backtrack".
*   **Kiểm tra tính giải được:** Đảm bảo các trạng thái do người dùng nhập hoặc tự động tạo ra đều có thể giải được trước khi chạy thuật toán.
*   **Triển khai nhiều thuật toán AI:** Xem chi tiết ở phần "Thuật toán đã triển khai".
*   **Trực quan hóa:**
    *   Hiển thị từng bước di chuyển của các ô số trong đường đi lời giải (tốc độ chậm).
    *   Hiển thị quá trình khám phá trạng thái của thuật toán Backtracking/CSP (tốc độ chậm).
    *   Hiển thị quá trình tạo bảng ngẫu nhiên của nút Backtrack (tốc độ nhanh).
*   **Benchmark:** So sánh thời gian chạy của các thuật toán tìm đường đi (pathfinding) trên cùng một trạng thái bắt đầu, chỉ hiển thị tên và thời gian.
*   **Hiển thị kết quả:** Cung cấp thông báo trạng thái, kết quả benchmark, và chi tiết các bước giải (nếu có).

## Yêu cầu

*   Python 3.x
*   Thư viện Pygame (`pip install pygame`)
*   **(Quan trọng)** Một font chữ hệ thống hỗ trợ tiếng Việt đã được cài đặt (ví dụ: Arial, Tahoma, Times New Roman...). Code đang ưu tiên tìm 'arial'. Nếu không có, bạn có thể cần sửa biến `SYSTEM_FONT_NAME` trong code hoặc đặt một file `.ttf` hỗ trợ tiếng Việt vào cùng thư mục và sửa code để load trực tiếp file đó.

## Cài đặt

```bash
pip install pygame