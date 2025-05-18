# Báo cáo Đồ án: Giải quyết Bài toán 8-Puzzle bằng các Thuật toán Trí tuệ Nhân tạo

# 1. Giới thiệu
linkgithub : https://github.com/phantrongphu123/-individual_AI-_project
## 1.1. Bài toán 8-Puzzle
8-Puzzle là một trò chơi xếp số kinh điển gồm một lưới 3x3 chứa 8 ô được đánh số từ 1 đến 8 và một ô trống. Mục tiêu là di chuyển các ô số để đạt được trạng thái đích (thường là các số được sắp xếp theo thứ tự). Đây là một bài toán tiêu biểu trong lĩnh vực Trí tuệ Nhân tạo, thường được sử dụng để nghiên cứu và đánh giá hiệu suất của các thuật toán tìm kiếm.

## 1.2. Mục tiêu đồ án
Phát triển một ứng dụng giải bài toán 8-Puzzle thông qua giao diện trực quan.

Triển khai nhiều thuật toán từ cơ bản đến nâng cao trong lĩnh vực AI.

Cung cấp công cụ để nghiên cứu, so sánh và đánh giá các thuật toán tìm kiếm.

Mô phỏng trực quan quá trình giải quyết bài toán.

## 2. Cấu trúc thư mục dự án
Dự án được thiết kế theo mô hình MVC để tách biệt rõ giữa logic, giao diện và dữ liệu:
```plaintext
EightPuzzleMVC/
└── EightPuzzleMVC/
    ├── main.py
    ├── main_2.py
    ├── README.md
    ├── app/
    │   ├── __init__.py
    │   ├── controllers/
    │   │   ├── __init__.py
    │   │   └── game_controller.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── puzzle_game.py
    │   │   ├── puzzle_state.py
    │   │   └── solvers/
    │   │       ├── __init__.py
    │   │       ├── local_search_solvers.py
    │   │       ├── rl_agents.py
    │   │       ├── search_solvers.py
    │   │       └── solver_base.py
    │   └── views/
    │       ├── __init__.py
    │       └── pygame_view.py
    └── data/
        ├── q_table_8puzzle.json
        └── sarsa_q_table_8puzzle.json
```
## 3. Các tính năng chính
Giao diện đồ họa (GUI) sử dụng Pygame.

Hỗ trợ nhiều thuật toán AI:

Tìm kiếm không thông tin: BFS, DFS, UCS, IDS

Tìm kiếm có thông tin: Greedy, A*, IDA*

Tìm kiếm cục bộ: Hill Climbing, Beam Search, Simulated Annealing, Genetic Algorithm

CSP: Backtracking, Min-Conflicts

Học tăng cường: Q-Learning, SARSA, Policy Gradient, DQN

Thuật toán cho môi trường phức tạp: AND-OR Search, Belief State Search

Tự động giải puzzle với bất kỳ thuật toán nào đã triển khai.

Mô phỏng trực quan quá trình tìm kiếm.

Lưu/Tải Q-Table cho thuật toán học tăng cường.

## 4. Công nghệ sử dụng
Ngôn ngữ lập trình: Python 3.x

Thư viện GUI: Pygame

AI/ML: NumPy, TensorFlow hoặc PyTorch (tùy thuật toán DQN)

Thư viện tiện ích: json, random, math, collections, ...

## 5. Cài đặt và Chạy chương trình
## 5.1. Yêu cầu
Python 3.7 trở lên

pip (Python package installer)

## 5.2. Cài đặt thư viện
pip install pygame numpy
# pip install tensorflow    # Hoặc PyTorch nếu dùng DQN
Tham khảo requirements.txt nếu có để biết đầy đủ danh sách.

## 5.3. Chạy ứng dụng

cd EightPuzzleMVC/EightPuzzleMVC/
python main.py
# hoặc python main_2.py
6. Mô hình hóa bài toán 8-Puzzle
Trạng thái: Mảng 2D hoặc tuple/list biểu diễn vị trí các ô.

Hành động: Di chuyển ô trống (lên, xuống, trái, phải).

Hàm chuyển tiếp: Thực hiện hành động để tạo trạng thái mới.

Trạng thái ban đầu: Nhập từ người dùng hoặc sinh ngẫu nhiên (hợp lệ).

Trạng thái đích: Thường là dãy số 1 đến 8.

Chi phí: Mỗi hành động có chi phí 1.

## 7. Các thuật toán Trí tuệ Nhân tạo được triển khai
## 7.1. Tìm kiếm không thông tin (Uninformed Search)
BFS: Tìm lời giải ngắn nhất.

DFS: Nhanh nhưng không đảm bảo tối ưu.

UCS: Tìm đường đi có chi phí thấp nhất.

IDS: Kết hợp ưu điểm BFS & DFS.

## 7.2. Tìm kiếm có thông tin (Informed Search)
Sử dụng heuristic để dẫn đường:

Greedy Search, A*, IDA*

Heuristic phổ biến:

Số ô sai vị trí

Tổng khoảng cách Manhattan

## 7.3. Tìm kiếm cục bộ (Local Search)
Hill Climbing: Dễ mắc kẹt ở cực trị cục bộ.

Simulated Annealing: Cho phép "lùi bước" có tính toán.

Genetic Algorithm: Dựa trên tiến hóa tự nhiên.

## 7.4. Bài toán thỏa mãn ràng buộc (CSP)
Mô hình hoá các ô là biến, giá trị là số.

Backtracking + Min-Conflicts.

## 7.5. Học tăng cường (Reinforcement Learning)
Q-Learning, SARSA, Policy Gradient, DQN

Lưu/đọc Q-Table từ file JSON.

7.6. Thuật toán môi trường phức tạp
AND-OR Search, Belief State Search

Dành cho các môi trường không xác định hoặc phi tuyến tính.

## 8. Tổng quan Giao diện Người dùng (GUI)
Giao diện trực quan với Pygame.

Cho phép nhập trạng thái đầu, chọn thuật toán và theo dõi trực tiếp quá trình giải.

Hiển thị số bước đi, thời gian chạy và các trạng thái trung gian.

## 9. Hướng phát triển tương lai
Tối ưu hiệu năng thuật toán.

Hỗ trợ nhiều kích thước puzzle (15-puzzle, 24-puzzle).

Thêm chế độ học tự động và huấn luyện online.

Cải thiện GUI: thêm màu sắc, hiệu ứng, thống kê.
 
## 10. Tác giả
Họ tên: Phan Trọng Phú

MSSV: 23133056
