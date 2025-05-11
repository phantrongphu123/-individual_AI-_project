# 8-Puzzle Solver (MVC - Pygame)

Dự án này là một ứng dụng giải game 8-Puzzle được xây dựng bằng Python với thư viện Pygame cho giao diện người dùng, tuân theo mô hình kiến trúc Model-View-Controller (MVC). Ứng dụng cho phép người dùng tạo bảng puzzle ngẫu nhiên, tự nhập bảng, và giải bằng nhiều thuật toán tìm kiếm và trí tuệ nhân tạo khác nhau.

## Mục lục

1.  [Cấu trúc Dự án](#cấu-trúc-dự-án)
2.  [Mô hình MVC](#mô-hình-mvc)
3.  [Các Thuật toán Được Triển khai](#các-thuật-toán-được-triển-khai)
    *   [Uninformed Search](#uninformed-search)
    *   [Informed Search](#informed-search)
    *   [Local Search](#local-search)
    *   [Constraint Satisfaction Problems (CSP)](#constraint-satisfaction-problems-csp)
    *   [Reinforcement Learning (RL)](#reinforcement-learning-rl)
    *   [Complex Environments](#complex-environments)
4.  [Cách Chạy Ứng dụng](#cách-chạy-ứng-dụng)
5.  [Hướng dẫn Sử dụng Giao diện](#hướng-dẫn-sử-dụng-giao-diện)
6.  [Hướng Phát triển Tiếp theo](#hướng-phát-triển-tiếp-theo)

## Cấu trúc Dự án
Use code with caution.
Markdown
EightPuzzleMVC/
├── app/
│ ├── init.py
│ ├── controllers/
│ │ ├── init.py
│ │ └── game_controller.py # Điều khiển luồng ứng dụng
│ ├── models/
│ │ ├── init.py
│ │ ├── puzzle_state.py # Biểu diễn trạng thái của puzzle
│ │ ├── puzzle_game.py # Quản lý logic game (tạo, reset)
│ │ └── solvers/ # Chứa các thuật toán giải
│ │ ├── init.py
│ │ ├── solver_base.py # Lớp cơ sở cho các solver
│ │ ├── search_solvers.py # BFS, DFS, UCS, IDS, Greedy, A*, IDA*
│ │ ├── local_search_solvers.py # Hill Climbing, SA, GA, Beam Search
│ │ ├── csp_solvers.py # Backtracking, Min-Conflicts
│ │ └── rl_agents.py # Q-Learning, SARSA (và khung cho DQN, PG)
│ │ └── complex_env_solvers.py # Khung cho And-Or, Belief State
│ ├── views/
│ │ ├── init.py
│ │ └── pygame_view.py # Giao diện người dùng Pygame
├── data/ # (Tùy chọn) Có thể chứa file Q-table đã lưu
│ └── q_table_8puzzle.json
├── main.py # Điểm vào chính của ứng dụng
└── README.md
## Mô hình MVC

*   **Model (`app/models/`)**:
    *   `PuzzleState`: Đại diện cho một cấu hình cụ thể của bảng 8-puzzle. Bao gồm các phương thức để kiểm tra trạng thái đích, sinh các nước đi hợp lệ, tính heuristic (Manhattan distance), kiểm tra tính giải được.
    *   `PuzzleGame`: Quản lý trạng thái hiện tại của game, trạng thái đích mặc định, và các hành động cấp cao như tạo bảng ngẫu nhiên có thể giải được, reset bảng.
    *   `solvers/`: Mỗi thuật toán giải được triển khai như một lớp riêng kế thừa từ `SolverBase`. Các solver nhận một `PuzzleState` ban đầu và cố gắng tìm một `solution_path` (danh sách các `PuzzleState`) đến đích. Chúng cũng thu thập các `metrics` (thời gian, số trạng thái đã khám phá, v.v.).
*   **View (`app/views/pygame_view.py`)**:
    *   `PygameView`: Sử dụng thư viện Pygame để hiển thị bảng puzzle, các nút điều khiển (Randomize, Reset, Input Board, chọn thuật toán, Solve), và khu vực hiển thị thông báo/kết quả.
    *   Nó không chứa logic game hay giải thuật toán mà chỉ nhận dữ liệu từ Controller để hiển thị và gửi các hành động của người dùng đến Controller.
*   **Controller (`app/controllers/game_controller.py`)**:
    *   `GameController`: Đóng vai trò trung gian.
        *   Nhận yêu cầu từ `PygameView` (ví dụ: người dùng nhấn nút "Solve BFS").
        *   Tương tác với `PuzzleGame` để lấy trạng thái hiện tại hoặc tạo bảng mới.
        *   Tạo instance của `Solver` tương ứng (ví dụ: `BFSSolver`) và truyền vào trạng thái puzzle hiện tại.
        *   **Chạy solver trong một luồng riêng (`threading.Thread`)** để giao diện không bị treo.
        *   Sử dụng `queue.Queue` để nhận kết quả (metrics, solution path) từ luồng solver.
        *   Yêu cầu `PygameView` cập nhật giao diện (hiển thị bảng, kết quả, animation đường đi) khi có kết quả từ solver hoặc khi trạng thái game thay đổi.

## Các Thuật toán Được Triển khai

Mỗi thuật toán được triển khai như một lớp kế thừa từ `SolverBase` và có phương thức `solve()`.

### Uninformed Search
*(Trong `app/models/solvers/search_solvers.py`)*
Các thuật toán này tìm kiếm giải pháp mà không sử dụng thông tin về khoảng cách hay chi phí ước tính đến đích.

*   **BFS (Breadth-First Search - `BFSSolver`)**:
    *   **Hoạt động:** Duyệt các trạng thái theo từng mức (level by level) sử dụng một hàng đợi (queue).
    *   **Xử lý:** Tìm ra đường đi ngắn nhất (ít bước di chuyển nhất). Sử dụng một `set` các `board_tuple` đã thăm để tránh chu trình và lặp lại trạng thái.
*   **DFS (Depth-First Search - `DFSSolver`)**:
    *   **Hoạt động:** Ưu tiên đi sâu nhất có thể vào một nhánh tìm kiếm sử dụng một ngăn xếp (stack).
    *   **Xử lý:** Có tham số `max_depth` để tránh các nhánh quá sâu hoặc vô hạn. Sử dụng `set` các `board_tuple` đã thăm để tránh chu trình. Không đảm bảo đường đi ngắn nhất.
*   **UCS (Uniform Cost Search - `UCSSolver`)**:
    *   **Hoạt động:** Mở rộng nút có chi phí đường đi (`g(n)`) nhỏ nhất từ trạng thái bắt đầu. Sử dụng hàng đợi ưu tiên (`heapq`).
    *   **Xử lý:** Vì chi phí mỗi nước đi trong 8-puzzle là 1, UCS hoạt động tương tự BFS và tìm ra đường đi ngắn nhất. Lưu trữ `g_costs` tốt nhất đến mỗi trạng thái đã thăm.
*   **IDS (Iterative Deepening Search - `IDSSolver`)**:
    *   **Hoạt động:** Kết hợp ưu điểm của BFS (hoàn chỉnh, tối ưu theo độ sâu) và DFS (ít tốn bộ nhớ). Thực hiện một loạt các tìm kiếm giới hạn độ sâu (DLS) với giới hạn độ sâu tăng dần.
    *   **Xử lý:** Bên trong mỗi DLS, nó hoạt động như DFS. `visited` set được quản lý riêng cho mỗi lần lặp DLS để đảm bảo tìm được giải pháp nông nhất.

### Informed Search
*(Trong `app/models/solvers/search_solvers.py`)*
Các thuật toán này sử dụng hàm heuristic để ước tính khoảng cách đến đích, giúp hướng dẫn tìm kiếm hiệu quả hơn. Heuristic mặc định sử dụng là Manhattan Distance.

*   **Greedy Search (`GreedySolver`)**:
    *   **Hoạt động:** Luôn mở rộng nút có vẻ gần đích nhất theo hàm heuristic `h(n)`. Sử dụng hàng đợi ưu tiên chỉ dựa trên `h(n)`.
    *   **Xử lý:** Nhanh nhưng không hoàn chỉnh và không đảm bảo tìm ra giải pháp tối ưu. Có thể bị kẹt ở các lựa chọn "tham lam" sai lầm.
*   **A\* (A-Star - `AStarSolver`)**:
    *   **Hoạt động:** Mở rộng nút có giá trị `f(n) = g(n) + h(n)` nhỏ nhất, trong đó `g(n)` là chi phí thực tế từ đầu và `h(n)` là chi phí ước tính đến đích. Sử dụng hàng đợi ưu tiên.
    *   **Xử lý:** Hoàn chỉnh và tối ưu (tìm đường đi ngắn nhất) nếu heuristic là "admissible" (ví dụ: Manhattan distance). Lưu trữ `g_costs` tốt nhất.
*   **IDA\* (Iterative Deepening A-Star - `IDAStarSolver`)**:
    *   **Hoạt động:** Kết hợp A\* với tìm kiếm sâu dần. Thay vì giới hạn độ sâu, nó giới hạn tổng chi phí `f(n)` (threshold). Ngưỡng này tăng dần.
    *   **Xử lý:** Bên trong mỗi lần lặp, nó thực hiện một tìm kiếm sâu giới hạn bởi ngưỡng f-cost. Hoàn chỉnh và tối ưu như A\* nhưng tiết kiệm bộ nhớ hơn đáng kể.

### Local Search
*(Trong `app/models/solvers/local_search_solvers.py`)*
Các thuật toán này bắt đầu với một trạng thái (có thể ngẫu nhiên) và cố gắng cải thiện nó dần dần bằng các thay đổi nhỏ. Hàm đánh giá thường là heuristic (ví dụ: Manhattan distance, mục tiêu là giảm thiểu nó).

*   **Hill Climbing (`HillClimbingSolver` - Simple, Steepest-Ascent, Stochastic)**:
    *   **Hoạt động:** Di chuyển đến một trạng thái lân cận nếu nó "tốt hơn" (heuristic thấp hơn) trạng thái hiện tại.
        *   *Simple:* Chọn hàng xóm tốt hơn đầu tiên.
        *   *Steepest-Ascent:* Chọn hàng xóm tốt nhất.
        *   *Stochastic:* Chọn ngẫu nhiên một trong số các hàng xóm tốt hơn.
    *   **Xử lý:** Dễ bị kẹt ở cực tiểu địa phương. Trả về chuỗi các trạng thái đã đi qua.
*   **Simulated Annealing (`SimulatedAnnealingSolver`)**:
    *   **Hoạt động:** Tương tự Hill Climbing nhưng cho phép di chuyển đến trạng thái "xấu hơn" với một xác suất nhất định. Xác suất này giảm dần theo "nhiệt độ" (temperature) của thuật toán.
    *   **Xử lý:** Có khả năng thoát khỏi cực tiểu địa phương tốt hơn Hill Climbing.
*   **Beam Search (`BeamSearchSolver`)**:
    *   **Hoạt động:** Giữ lại một tập hợp `k` (beam width) các trạng thái tốt nhất tại mỗi bước. Từ `k` trạng thái này, sinh ra tất cả các trạng thái con và chọn ra `k` trạng thái con tốt nhất mới.
    *   **Xử lý:** Không hoàn chỉnh, không tối ưu, nhưng có thể hiệu quả hơn BFS về bộ nhớ nếu `k` nhỏ.
*   **Genetic Algorithm (GA - `GeneticAlgorithmSolver`)**:
    *   **Hoạt động:** Mô phỏng quá trình tiến hóa tự nhiên.
        *   *Cá thể:* Một trạng thái 8-puzzle (bảng 2D).
        *   *Quần thể:* Một tập hợp các cá thể.
        *   *Hàm thích nghi (Fitness):* Dựa trên heuristic (ví dụ: `1 / (1 + manhattan_distance)`).
        *   *Toán tử:* Lựa chọn (Tournament), Lai ghép (Cycle Crossover - CX1), Đột biến (Swap hai ô).
    *   **Xử lý:** Tạo quần thể ban đầu gồm các trạng thái giải được. Qua nhiều thế hệ, các cá thể tốt hơn sẽ được chọn và lai ghép/đột biến để tạo ra các giải pháp tiềm năng mới. Trả về trạng thái tốt nhất tìm được.

### Constraint Satisfaction Problems (CSP)
*(Trong `app/models/solvers/csp_solvers.py`)*
Mô hình hóa 8-Puzzle như một CSP (ví dụ: mỗi ô là một biến, giá trị là số từ 0-8).

*   **Backtracking (`BacktrackingCSPSolver`)**:
    *   **Hoạt động:** Gán giá trị cho các biến một cách tuần tự. Nếu một phép gán vi phạm ràng buộc, quay lui và thử giá trị khác.
    *   **Xử lý (hiện tại cho 8-puzzle):** Phiên bản này gán màu (tương đương số) cho các ô trống. Hàm `_is_consistent` kiểm tra các ràng buộc cục bộ (ví dụ: không quá 2 ô cùng màu kề nhau trừ khi là endpoint - điều này cần điều chỉnh cho 8-puzzle, ràng buộc là mỗi số chỉ xuất hiện 1 lần). Hàm `is_fully_solved()` của `Puzzle` được gọi ở cuối để kiểm tra trạng thái đích.
*   **Backtracking with Forward Checking (`ForwardCheckingBacktrackingSolver`)**:
    *   **(Khung sườn ý tưởng):** Cải tiến của Backtracking. Khi một biến được gán giá trị, các giá trị không tương thích sẽ bị loại bỏ khỏi miền của các biến chưa được gán có ràng buộc với nó. Cần quản lý và hoàn tác miền giá trị.
*   **Min-Conflicts (`MinConflictsSolver`)**:
    *   **(Triển khai trước đó):** Bắt đầu với một gán giá trị hoàn chỉnh ngẫu nhiên. Lặp đi lặp lại việc chọn một biến xung đột và gán cho nó giá trị làm giảm thiểu số xung đột.
    *   **Xử lý (cho 8-puzzle):** Hàm `_count_conflicts_for_cell` cần được định nghĩa lại cho 8-puzzle (ví dụ: xung đột là khi một số xuất hiện nhiều hơn một lần, hoặc số ô trống không phải là 1). Hàm `_is_flowfree_solution_valid` sẽ là `puzzle_state.is_goal()`.

### Reinforcement Learning (RL)
*(Trong `app/models/solvers/rl_agents.py`)*
Agent học cách giải puzzle thông qua tương tác và phần thưởng.

*   **Q-Learning (`QLearningAgent`)**:
    *   **Hoạt động:** Học một hàm giá trị hành động Q(s,a) ước tính phần thưởng kỳ vọng khi thực hiện hành động `a` tại trạng thái `s`.
    *   **Xử lý:** Sử dụng Q-table để lưu trữ Q-values. Trạng thái `s` là `board_tuple`. Hành động `a` được biểu diễn bằng `board_tuple` của trạng thái kế tiếp. Cập nhật Q-table bằng công thức Bellman. Sử dụng chính sách epsilon-greedy để chọn hành động. Sau khi huấn luyện, trích xuất chính sách bằng cách luôn chọn hành động có Q-value cao nhất. Có chức năng lưu/tải Q-table.
*   **SARSA (`SarsaAgent`)**:
    *   **Hoạt động:** Tương tự Q-Learning nhưng là on-policy. Cập nhật Q-value dựa trên hành động (A') thực sự được chọn ở trạng thái kế tiếp (S') theo chính sách hiện tại.
    *   **Xử lý:** Logic cập nhật khác Q-Learning một chút: `Q(S,A) <- Q(S,A) + alpha * [R + gamma * Q(S',A') - Q(S,A)]`.
*   **Deep Q Network (DQN - `DeepQNetworkAgent`)**:
    *   **(Khung sườn ý tưởng):** Sử dụng mạng nơ-ron sâu để xấp xỉ hàm Q(s,a) khi không gian trạng thái quá lớn cho Q-table. Yêu cầu thư viện học sâu.
*   **Policy-gradient (`PolicyGradientAgent` - ví dụ REINFORCE)**:
    *   **(Khung sườn ý tưởng):** Học trực tiếp một hàm chính sách (policy) `pi(a|s)` được tham số hóa bởi mạng nơ-ron. Cập nhật dựa trên gradient của phần thưởng kỳ vọng. Yêu cầu thư viện học sâu.

### Complex Environments
*(Trong `app/models/solvers/complex_env_solvers.py` - Yêu cầu các biến thể của 8-Puzzle)*

*   **And-Or Search Tree (`AndOrSearchSolver`)**:
    *   **(Khung sườn ý tưởng):** Cho các bài toán có hành động không xác định (ví dụ: 8-puzzle với nước đi ngẫu nhiên từ môi trường). Tìm kiếm một kế hoạch có điều kiện.
*   **Belief State Search (`BeliefStateSearchSolver`)**:
    *   **(Khung sườn ý tưởng):** Cho các bài toán không quan sát (ví dụ: 8-puzzle "mù"). Agent duy trì một tập hợp các trạng thái có thể (belief state) và tìm kiếm trên không gian các belief state này.
*   **Searching with Partially Observation**:
    *   **(Khung sườn ý tưởng):** Agent nhận được quan sát không đầy đủ, cập nhật belief state. Thường dùng POMDP.

## Cách Chạy Ứng dụng

1.  Đảm bảo bạn đã cài đặt Python và Pygame (`pip install pygame`).
    *   Đối với CP-SAT solver (nếu bạn chọn triển khai lại), cần: `pip install ortools`.
    *   Đối với các thuật toán RL nâng cao (DQN, Policy Gradient), cần thư viện học sâu như TensorFlow hoặc PyTorch.
2.  Mở terminal hoặc command prompt.
3.  Di chuyển đến thư mục gốc của dự án: `cd path/to/EightPuzzleMVC`
4.  Chạy file `main.py`: `python main.py`
    *   Mặc định sẽ khởi chạy giao diện Pygame.
    *   (Tùy chọn) Có thể thêm đối số dòng lệnh để chạy ở chế độ CLI hoặc benchmark nếu bạn đã triển khai.

## Hướng dẫn Sử dụng Giao diện (Pygame)

*   **Randomize Board:** Tạo một bảng 8-puzzle ngẫu nhiên mới có thể giải được.
*   **Reset Board:** Đưa bảng về trạng thái ngẫu nhiên được tạo gần nhất.
*   **Input Board:** (Nếu đã triển khai đầy đủ) Cho phép người dùng tự nhập cấu hình bảng.
    *   **Confirm/Cancel:** Xác nhận hoặc hủy bỏ việc nhập bảng.
*   **< Algo / Algo >:** Chọn thuật toán giải từ danh sách.
*   **Solve Selected:** Bắt đầu giải puzzle bằng thuật toán đã chọn.
*   **Khu vực hiển thị lưới:** Hiển thị trạng thái hiện tại của bảng puzzle. Các ô số ở đúng vị trí có thể được tô màu xanh, sai vị trí màu đỏ.
*   **Khu vực thông báo/kết quả:** Hiển thị các thông báo từ ứng dụng, trạng thái giải, và các metrics của thuật toán.

## Hướng Phát triển Tiếp theo

*   Hoàn thiện việc triển khai tất cả các thuật toán đã liệt kê.
*   Cải thiện hàm heuristic cho các thuật toán Informed Search.
*   Tối ưu hóa hiệu năng của các thuật toán.
*   Thêm chức năng lưu/tải Q-table cho các agent RL khác (SARSA).
*   Xây dựng các biến thể 8-puzzle cho nhóm Complex Environments.
*   Cải thiện giao diện người dùng Pygame:
    *   Dropdown thực sự để chọn thuật toán/heuristic.
    *   Trực quan hóa chi tiết hơn quá trình giải của từng thuật toán.
    *   Thanh tiến trình cho các thuật toán chạy lâu.
    *   Cho phép người dùng tùy chỉnh các tham số của thuật toán.
*   Hoàn thiện hệ thống Benchmark và thêm khả năng vẽ biểu đồ so sánh.

---
