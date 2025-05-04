# --- 23133056_PhanTrongPhu_tuan13_
# --- Phiên bản: Visualize tạo bảng ngẫu nhiên + Backtrack ---

import pygame
import time
import heapq
import random
import math
from collections import deque
import sys
import traceback
import textwrap

# --- Khởi tạo Pygame và Hằng số ---
pygame.init()

GRID_SIZE = 450         # Kích thước lưới (cột 1)
TILE_SIZE = GRID_SIZE // 3
PANEL_PADDING = 15      # Khoảng cách giữa các cột
BUTTON_PANEL_WIDTH = 200 # Chiều rộng cố định cho panel nút (cột 2)
RESULTS_AREA_WIDTH = 250 # Chiều rộng cố định cho panel kết quả (cột 3)

WIDTH = GRID_SIZE + BUTTON_PANEL_WIDTH + RESULTS_AREA_WIDTH + PANEL_PADDING * 3 # Tổng chiều rộng (thêm padding phải)
HEIGHT = max(GRID_SIZE, 600) + PANEL_PADDING * 2 # Chiều cao cửa sổ (tăng padding dưới để có chỗ note)

GRID_X = PANEL_PADDING
GRID_Y = PANEL_PADDING
BUTTON_PANEL_X = GRID_X + GRID_SIZE + PANEL_PADDING
BUTTON_PANEL_Y = GRID_Y # Nút bắt đầu cùng Y với lưới
RESULTS_AREA_X = BUTTON_PANEL_X + BUTTON_PANEL_WIDTH + PANEL_PADDING
RESULTS_AREA_Y = GRID_Y # Khu vực kết quả bắt đầu cùng Y với lưới
RESULTS_AREA_HEIGHT = HEIGHT - PANEL_PADDING * 2 - 30 # Giảm chiều cao results area để chừa chỗ cho note

# Fonts
FONT = pygame.font.Font(None, 50) # Font ô số
BUTTON_FONT = pygame.font.Font(None, 20) # Font nút
MSG_FONT = pygame.font.Font(None, 20)    # Font thông báo
RESULTS_FONT = pygame.font.Font(None, 17) # Font khu vực kết quả
NOTE_FONT = pygame.font.Font(None, 18) # Font cho ghi chú

# Màu sắc
WHITE = (255, 255, 255); BLACK = (0, 0, 0); GREY = (200, 200, 200)
DARK_GREY = (100, 100, 100); LIGHT_GREY = (220, 220, 220)
RED = (200, 0, 0); GREEN = (0, 200, 0); BLUE = (70, 130, 180)
DARK_BLUE = (0, 0, 139); ORANGE = (255, 165, 0); PURPLE = (128, 0, 128)
DARK_RED = (150, 50, 50); DARK_GREEN = (0, 100, 0)
NOTE_COLOR = (80, 80, 80) # Màu cho ghi chú

# Trạng thái đích và Giới hạn
GOAL_STATE = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
GOAL_STATE_TUPLE = tuple(map(tuple, GOAL_STATE))
DEFAULT_MAX_DEPTH = 15; MAX_BFS_STATES = 500000; MAX_DFS_POPS = 750000
MAX_GREEDY_EXPANSIONS = 500000; MAX_HILL_CLIMBING_STEPS = 5000
MAX_SA_ITERATIONS = 30000; MAX_BEAM_STEPS = 1000; MAX_GA_GENERATIONS = 300
MAX_IDA_STAR_NODES = 2000000

# --- Helper Functions ---
def state_to_tuple(state): return tuple(map(tuple, state))
def is_valid_state(state):
    try:
        flat = sum(state, [])
        return len(flat) == 9 and sorted(flat) == list(range(9))
    except (TypeError, ValueError):
        return False
def get_inversions(flat_state):
    count = 0; size = len(flat_state)
    for i in range(size):
        for j in range(i + 1, size):
            # Chỉ đếm nghịch thế cho các số khác 0
            if flat_state[i] != 0 and flat_state[j] != 0 and flat_state[i] > flat_state[j]:
                count += 1
    return count
def is_solvable(state, goal_state=GOAL_STATE):
    if not is_valid_state(state): return False
    # Đối với lưới 3x3, tính chẵn lẻ của số nghịch thế là đủ
    return get_inversions(sum(state, [])) % 2 == get_inversions(sum(goal_state, [])) % 2
def generate_random_solvable_state(goal_state=GOAL_STATE):
    attempts = 0; max_attempts = 1000
    target_inversions_parity = get_inversions(sum(goal_state, [])) % 2
    while attempts < max_attempts:
        flat = list(range(9)); random.shuffle(flat);
        current_inversions_parity = get_inversions(flat) % 2
        if current_inversions_parity == target_inversions_parity:
             state = [flat[i:i+3] for i in range(0, 9, 3)]
             # Kiểm tra lại tính hợp lệ phòng trường hợp logic thay đổi
             if is_valid_state(state) and is_solvable(state, goal_state):
                 return state
        attempts += 1
    print(f"Lỗi: Không thể tạo trạng thái giải được sau {max_attempts} lần thử.")
    # Trả về trạng thái đích nếu không tạo được trạng thái ngẫu nhiên
    return [list(row) for row in goal_state]

def get_solution_moves(path):
    if not path or len(path) < 2: return ["Không có bước đi trong đường dẫn."]
    moves = []; temp_puzzle = Puzzle(path[0]) # Dùng trạng thái đầu để tạo puzzle tạm
    for i in range(len(path) - 1):
        s1, s2 = path[i], path[i+1]
        try:
            r1, c1 = temp_puzzle.get_blank_position(s1) # Vị trí trống ở trạng thái trước
            r2, c2 = temp_puzzle.get_blank_position(s2) # Vị trí trống ở trạng thái sau
            moved_tile_value = s1[r2][c2] # Ô ở vị trí mới của ô trống trong trạng thái cũ
            move = f"Ô {moved_tile_value} di chuyen "
            if r2 < r1: move += "Xuong"
            elif r2 > r1: move += "Len"
            elif c2 < c1: move += "Phai"
            elif c2 > c1: move += "Trai"
            else: move += "Loi?"
            moves.append(move)
        except (ValueError, IndexError):
            moves.append("Loi xac dinh buoc di.")
            continue
    return moves

# --- Lớp Puzzle ---
class Puzzle:
    def __init__(self, start, goal=GOAL_STATE):
        if not is_valid_state(start):
            raise ValueError("Trạng thái bắt đầu không hợp lệ khi khởi tạo Puzzle.")
        self.start = [list(row) for row in start]
        self.goal = [list(row) for row in goal]
        self.goal_tuple = state_to_tuple(self.goal)
        self.rows = 3; self.cols = 3 # Cố định 3x3
        self.max_depth_limit = DEFAULT_MAX_DEPTH
        self._goal_pos_cache = self._build_goal_pos_cache()

    def _build_goal_pos_cache(self):
        cache = {};
        for r in range(self.rows):
            for c in range(self.cols):
                if self.goal[r][c] != 0: cache[self.goal[r][c]] = (r, c)
        return cache

    def get_blank_position(self, state):
        for r in range(self.rows):
            for c in range(self.cols):
                if state[r][c] == 0: return r, c
        raise ValueError("Trạng thái không hợp lệ: Không tìm thấy ô trống (0).")

    def is_goal(self, state): return state_to_tuple(state) == self.goal_tuple

    def get_neighbors(self, state):
        neighbors = [];
        try: r, c = self.get_blank_position(state)
        except ValueError: return neighbors
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                new_state = [row[:] for row in state]
                new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
                neighbors.append(new_state)
        return neighbors

    def heuristic(self, state): # Manhattan distance
        distance = 0
        for r in range(self.rows):
            for c in range(self.cols):
                val = state[r][c]
                if val != 0:
                    # Kiểm tra xem giá trị có trong cache không trước khi truy cập
                    if val in self._goal_pos_cache:
                        goal_r, goal_c = self._goal_pos_cache[val]
                        distance += abs(r - goal_r) + abs(c - goal_c)
                    else:
                         # Nếu giá trị không có trong cache (ví dụ state chứa số > 8)
                         # Trả về giá trị vô cùng lớn để tránh lỗi và không ưu tiên trạng thái này
                         # print(f"Cảnh báo: Giá trị {val} không có trong goal cache.")
                         return float('inf')
        return distance

    # --- Thuật toán tìm kiếm thông thường (trả về đường đi lời giải) ---
    # BFS, UCS, DFS, IDDFS, Greedy, A*, IDA*, Beam Search (giữ nguyên như code trước)
    # ... (Code cho các thuật toán này không thay đổi) ...
    def bfs(self):
        print("BFS: Bắt đầu..."); q=deque([(self.start,[self.start])]); v={state_to_tuple(self.start)}; c=0
        while q and c<MAX_BFS_STATES:
            s,p=q.popleft(); c+=1;
            if self.is_goal(s): print(f"BFS: Đã giải ({c} trạng thái)"); return p
            for n in self.get_neighbors(s):
                nt=state_to_tuple(n);
                if nt not in v: v.add(nt); q.append((n,p+[n]))
        st='Đạt giới hạn' if c>=MAX_BFS_STATES else 'Hàng đợi rỗng'; print(f"BFS: Thất bại/Giới hạn ({st})"); return None

    def ucs(self):
        print("UCS: Bắt đầu...")
        start_tuple = state_to_tuple(self.start)
        priority_queue = [(0, self.start, [self.start])] # (chi phí, trạng thái, đường đi)
        visited = {} # Lưu chi phí tốt nhất đến mỗi trạng thái: {state_tuple: cost}
        count = 0
        max_expansions = MAX_BFS_STATES # Dùng chung giới hạn với BFS

        while priority_queue and count < max_expansions:
            cost, current_state, path = heapq.heappop(priority_queue)
            current_tuple = state_to_tuple(current_state)

            if current_tuple in visited and visited[current_tuple] <= cost:
                continue
            visited[current_tuple] = cost
            count += 1

            if self.is_goal(current_state):
                print(f"UCS: Đã giải! Chi phí={cost}, Số lần mở rộng={count}")
                return path

            for neighbor in self.get_neighbors(current_state):
                neighbor_tuple = state_to_tuple(neighbor)
                new_cost = cost + 1 # Giả sử mỗi bước đi tốn 1 chi phí
                if neighbor_tuple not in visited or new_cost < visited[neighbor_tuple]:
                    heapq.heappush(priority_queue, (new_cost, neighbor, path + [neighbor]))

        if count >= max_expansions: print(f"UCS: Thất bại/Giới hạn (Giới hạn={max_expansions})")
        else: print("UCS: Thất bại (Hàng đợi rỗng)")
        return None

    def dfs(self):
        print("DFS: Bắt đầu..."); st=[(self.start,[self.start])]; v={state_to_tuple(self.start)}; c=0
        while st and c<MAX_DFS_POPS:
            s,p=st.pop(); c+=1;
            if self.is_goal(s): print(f"DFS: Đã giải ({c} lần pop)"); return p
            if len(p)>self.max_depth_limit+15: continue
            for n in reversed(self.get_neighbors(s)):
                nt=state_to_tuple(n);
                if nt not in v: v.add(nt); st.append((n,p+[n]))
        st='Đạt giới hạn' if c>=MAX_DFS_POPS else 'Stack rỗng'; print(f"DFS: Thất bại/Giới hạn ({st})"); return None

    def iddfs(self):
        print(f"IDDFS: Bắt đầu..."); s_t=state_to_tuple(self.start); nodes=0
        def dls(state, path, depth_limit, visited_in_path):
            nonlocal nodes; nodes+=1;
            if self.is_goal(state): return path
            if depth_limit == 0: return None
            for neighbor in self.get_neighbors(state):
                neighbor_tuple = state_to_tuple(neighbor);
                if neighbor_tuple not in visited_in_path:
                    result = dls(neighbor, path + [neighbor], depth_limit - 1, visited_in_path | {neighbor_tuple});
                    if result: return result
            return None
        for depth in range(self.max_depth_limit + 1):
            # print(f"IDDFS: Thử độ sâu {depth}...") # Bỏ bớt log cho gọn
            result_path = dls(self.start, [self.start], depth, frozenset({s_t}))
            if result_path:
                print(f"IDDFS: Đã giải (Độ sâu={depth}, Số nút ~{nodes})"); return result_path
        print(f"IDDFS: Thất bại (Đạt giới hạn độ sâu D={self.max_depth_limit}, Số nút ~{nodes})"); return None

    def greedy(self):
        print("Greedy: Bắt đầu..."); s_t=state_to_tuple(self.start);
        pq=[(self.heuristic(self.start), self.start, [self.start])];
        visited={s_t}; count=0
        while pq and count<MAX_GREEDY_EXPANSIONS:
            _, current_state, path = heapq.heappop(pq); count+=1
            if self.is_goal(current_state): print(f"Greedy: Đã giải ({count} lần mở rộng)"); return path
            for neighbor in self.get_neighbors(current_state):
                neighbor_tuple=state_to_tuple(neighbor);
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple);
                    heapq.heappush(pq,(self.heuristic(neighbor), neighbor, path+[neighbor]))
        st='Đạt giới hạn' if count>=MAX_GREEDY_EXPANSIONS else 'Hàng đợi rỗng'; print(f"Greedy: Thất bại/Giới hạn ({st})"); return None

    def a_star(self):
        print("A*: Bắt đầu..."); s_t=state_to_tuple(self.start);
        open_list=[(self.heuristic(self.start) + 0, 0, self.start, [self.start])]; # (f, g, state, path)
        g_costs={s_t:0}; came_from = {s_t: None} # came_from dùng để tái tạo path hiệu quả hơn (tùy chọn)
        closed_set = set()
        count=0
        while open_list:
            f, g, current_state, _ = heapq.heappop(open_list); # Không cần lấy path từ heap nữa
            current_tuple = state_to_tuple(current_state);
            count+=1

            if current_tuple in closed_set: continue
            closed_set.add(current_tuple)

            if self.is_goal(current_state):
                # Tái tạo đường đi từ came_from
                path = []
                curr = current_state
                while curr is not None:
                    path.append(curr)
                    # Chuyển state thành tuple để làm key cho came_from
                    curr_tuple = state_to_tuple(curr)
                    curr = came_from.get(curr_tuple) # Lấy state cha
                path.reverse() # Đảo ngược để có thứ tự từ start đến goal
                print(f"A*: Đã giải (Mở rộng={count}, Chi phí={g})");
                return path

            for neighbor in self.get_neighbors(current_state):
                neighbor_tuple=state_to_tuple(neighbor);
                if neighbor_tuple in closed_set: continue

                tentative_g = g + 1
                if tentative_g < g_costs.get(neighbor_tuple, float('inf')):
                    g_costs[neighbor_tuple] = tentative_g;
                    came_from[neighbor_tuple] = current_state # Lưu trạng thái cha (không phải tuple)
                    h = self.heuristic(neighbor);
                    if h == float('inf'): continue # Bỏ qua nếu heuristic không hợp lệ
                    f_new = tentative_g + h;
                    heapq.heappush(open_list,(f_new, tentative_g, neighbor, None)) # Path không cần lưu trong heap

        print("A*: Thất bại (Open List rỗng)"); return None

    def ida_star(self):
        print("IDA*: Bắt đầu..."); start_tuple=state_to_tuple(self.start);
        bound = self.heuristic(self.start)
        path = [self.start]
        nodes_expanded = 0

        def search(current_path, g_cost, current_bound):
            nonlocal nodes_expanded
            nodes_expanded += 1
            if nodes_expanded > MAX_IDA_STAR_NODES: raise MemoryError("Đạt giới hạn số nút IDA*")

            current_state = current_path[-1]
            h_cost = self.heuristic(current_state)
            if h_cost == float('inf'): return False, float('inf') # Trạng thái không hợp lệ
            f_cost = g_cost + h_cost

            if f_cost > current_bound: return False, f_cost
            if self.is_goal(current_state): return True, current_path

            min_f_cost_above_bound = float('inf')
            current_path_tuples = frozenset(state_to_tuple(s) for s in current_path)

            for neighbor in self.get_neighbors(current_state):
                if state_to_tuple(neighbor) not in current_path_tuples:
                    found, result = search(current_path + [neighbor], g_cost + 1, current_bound)
                    if found: return True, result
                    min_f_cost_above_bound = min(min_f_cost_above_bound, result)

            return False, min_f_cost_above_bound

        iteration = 0
        while True:
            iteration += 1;
            # print(f"IDA*: Lần lặp {iteration}, Ngưỡng F = {bound}") # Bớt log
            nodes_expanded = 0
            try:
                found, result = search(path, 0, bound)
                if found:
                    print(f"IDA*: Đã giải (Ngưỡng cuối {bound}, Nút ~{nodes_expanded})"); return result
                elif result == float('inf'):
                    print("IDA*: Thất bại (Đã khám phá hết?)"); return None
                else:
                    bound = result
            except MemoryError:
                print(f"IDA*: Thất bại (Đạt giới hạn bộ nhớ/số nút {MAX_IDA_STAR_NODES})"); return None
            if bound > 100:
                 print(f"IDA*: Dừng lại do ngưỡng quá lớn ({bound})"); return None

    def beam_search(self, beam_width=5):
        print(f"Beam Search: Bắt đầu (Beam Width={beam_width})...");
        start_tuple=state_to_tuple(self.start);
        # Beam: [(heuristic, state, path)]
        beam = [(self.heuristic(self.start), [r[:] for r in self.start], [[r[:] for r in self.start]])];
        visited = {start_tuple};
        step = 0
        best_goal_path = None # Lưu đường đi tốt nhất đến đích đã tìm thấy

        while beam and step < MAX_BEAM_STEPS:
            step += 1;
            candidates = [];
            found_goal_in_step = False

            for h, current_state, path in beam:
                if self.is_goal(current_state):
                    found_goal_in_step = True
                    if best_goal_path is None or len(path) < len(best_goal_path):
                        best_goal_path = path
                    # Không dừng ngay, tiếp tục duyệt beam vì có thể có path ngắn hơn

                # Chỉ mở rộng nếu chưa tìm thấy đích HOẶC nếu path hiện tại ngắn hơn path đích tốt nhất đã biết
                # (heuristic pruning - nếu h đã lớn hơn cost của best_goal_path thì có thể bỏ qua)
                if best_goal_path is None or h < len(best_goal_path) -1 : # Ước tính f = h + g, g = len(path)-1
                    for neighbor in self.get_neighbors(current_state):
                        neighbor_tuple = state_to_tuple(neighbor);
                        if neighbor_tuple not in visited:
                             neighbor_h = self.heuristic(neighbor)
                             if neighbor_h != float('inf'): # Chỉ thêm nếu heuristic hợp lệ
                                candidates.append((neighbor_h, neighbor, path + [neighbor]));
                                visited.add(neighbor_tuple)

            # Nếu đã tìm thấy đích ở bước trước hoặc bước này, và không còn ứng viên nào tốt hơn
            if best_goal_path is not None:
                 # Kiểm tra xem có ứng viên nào có heuristic < chi phí hiện tại không
                 if not any(cand[0] < (len(best_goal_path) - 1) for cand in candidates):
                      print(f"Beam Search: Đã giải tại bước {step} (Tối ưu cục bộ trong beam)")
                      return best_goal_path

            if not candidates:
                 if best_goal_path: # Trả về path tốt nhất nếu bị kẹt nhưng đã thấy đích
                      print(f"Beam Search: Đã giải tại bước {step} (Không còn ứng viên)")
                      return best_goal_path
                 else: # Bị kẹt và chưa thấy đích
                      print(f"Beam Search: Thất bại (Không có ứng viên mới tại bước {step})");
                      return None

            candidates.sort(key=lambda x: x[0]);
            beam = candidates[:beam_width]

        # Hết vòng lặp
        if best_goal_path:
             print(f"Beam Search: Đã giải (Đạt giới hạn bước {step})")
             return best_goal_path
        else:
             print(f"Beam Search: Thất bại/Đạt giới hạn bước ({step})");
             return None

    # --- Backtracking và các thuật toán trả về chuỗi khám phá ---
    def backtracking(self): # Trả về chuỗi khám phá
        print("Backtrack: Bắt đầu...")
        start_tuple = state_to_tuple(self.start)
        stack = [(self.start, [self.start])] # (state, path_to_state)
        visited = {start_tuple}
        exploration_sequence = [] # Lưu chuỗi khám phá
        pops_count = 0
        max_pops = MAX_DFS_POPS

        while stack and pops_count < max_pops:
            current_state, path = stack.pop()
            pops_count += 1
            exploration_sequence.append(current_state) # Thêm vào chuỗi

            if self.is_goal(current_state):
                print(f"Backtrack: Đã giải ({pops_count} lần pop, độ dài path {len(path)-1})")
                return exploration_sequence

            if len(path) > self.max_depth_limit + 10: continue

            for neighbor in reversed(self.get_neighbors(current_state)):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in visited:
                    visited.add(neighbor_tuple)
                    stack.append((neighbor, path + [neighbor]))

        status = 'Đạt giới hạn' if pops_count >= max_pops else 'Stack rỗng'
        print(f"Backtrack: Thất bại/Giới hạn ({status}, {pops_count} lần pop)")
        return exploration_sequence # Vẫn trả về chuỗi đã khám phá

    def csp_solve(self): # Trả về chuỗi khám phá
        print("CSP Solve (thông qua Backtrack): Bắt đầu...")
        return self.backtracking()

    def and_or_search(self): # DLS Visit - Trả về path nếu tìm thấy
        max_depth = self.max_depth_limit;
        print(f"DLS Visit: Bắt đầu (Độ sâu tối đa={max_depth})...");
        visited_global = set()

        def dls_recursive(state, current_path_tuples, depth):
            nonlocal visited_global
            state_tuple = state_to_tuple(state)
            visited_global.add(state_tuple)

            if self.is_goal(state): return []
            if depth <= 0: return None

            shortest_sub_path = None
            for neighbor in self.get_neighbors(state):
                neighbor_tuple = state_to_tuple(neighbor)
                if neighbor_tuple not in current_path_tuples:
                    sub_path = dls_recursive(neighbor, current_path_tuples | {neighbor_tuple}, depth - 1)
                    if sub_path is not None:
                        full_path_from_neighbor = [neighbor] + sub_path
                        if shortest_sub_path is None or len(full_path_from_neighbor) < len(shortest_sub_path):
                            shortest_sub_path = full_path_from_neighbor
            return shortest_sub_path

        start_tuple = state_to_tuple(self.start)
        found_sub_path = dls_recursive(self.start, frozenset({start_tuple}), max_depth)

        if found_sub_path is not None:
            print(f"DLS Visit: Đã giải (Độ sâu {len(found_sub_path)}, Đã thăm {len(visited_global)})")
            return [self.start] + found_sub_path
        else:
            print(f"DLS Visit: Thất bại (Độ sâu {max_depth}, Đã thăm {len(visited_global)})")
            return None

    # --- Thuật toán Local Search (trả về chuỗi trạng thái) ---
    # Simple HC, Steepest HC, Stochastic HC, Simulated Annealing (giữ nguyên)
    # ... (Code cho các thuật toán này không thay đổi) ...
    def _hill_climbing_base(self, find_next_state_func, name):
        print(f"{name}: Bắt đầu...");
        current_state = [r[:] for r in self.start]
        path_taken = [[r[:] for r in current_state]]
        visited_tuples = {state_to_tuple(current_state)}
        steps = 0
        while not self.is_goal(current_state) and steps < MAX_HILL_CLIMBING_STEPS:
            steps += 1;
            current_heuristic = self.heuristic(current_state)
            next_state = find_next_state_func(current_state, current_heuristic)
            if next_state is None:
                print(f"{name}: Bị kẹt tại bước {steps}, H={current_heuristic}")
                return path_taken
            next_tuple = state_to_tuple(next_state)
            if next_tuple in visited_tuples:
                print(f"{name}: Phát hiện chu trình tại bước {steps}, dừng lại.")
                return path_taken
            current_state = next_state;
            visited_tuples.add(next_tuple);
            path_taken.append([r[:] for r in current_state])
        if self.is_goal(current_state): print(f"{name}: Đã giải sau {steps} bước.")
        else: print(f"{name}: Đạt giới hạn số bước ({steps})")
        return path_taken

    def simple_hill_climbing(self):
        def find_first_better(state, current_h):
            neighbors = self.get_neighbors(state); random.shuffle(neighbors)
            for neighbor in neighbors:
                h = self.heuristic(neighbor)
                if h < current_h: return neighbor
            return None
        return self._hill_climbing_base(find_first_better,"Simple HC")

    def steepest_ascent_hill_climbing(self):
        def find_best_neighbor(state, current_h):
            best_neighbor = None; best_h = current_h
            neighbors = self.get_neighbors(state)
            for neighbor in neighbors:
                h = self.heuristic(neighbor)
                if h < best_h: best_h = h; best_neighbor = neighbor
            return best_neighbor if best_neighbor is not None else None
        return self._hill_climbing_base(find_best_neighbor,"Steepest HC")

    def stochastic_hill_climbing(self):
        def find_random_better(state, current_h):
            better_neighbors = [n for n in self.get_neighbors(state) if self.heuristic(n) < current_h]
            if better_neighbors: return random.choice(better_neighbors)
            else: return None
        return self._hill_climbing_base(find_random_better,"Stochastic HC")

    def simulated_annealing(self, initial_temp=1000, cooling_rate=0.997, min_temp=0.1):
        print(f"SA: Bắt đầu (T0={initial_temp}, Rate={cooling_rate}, Tmin={min_temp})...");
        current_state=[r[:] for r in self.start];
        current_h = self.heuristic(current_state);
        temp = initial_temp;
        path_taken = [[r[:] for r in current_state]];
        iterations = 0;
        last_accepted_tuple=state_to_tuple(current_state)
        while temp > min_temp and iterations < MAX_SA_ITERATIONS:
            iterations += 1
            if self.is_goal(current_state):
                 print(f"SA: Đã giải tại vòng lặp {iterations}, Nhiệt độ={temp:.2f}"); return path_taken
            neighbors = self.get_neighbors(current_state)
            if not neighbors:
                print(f"SA: Bị kẹt (không có hàng xóm) tại vòng lặp {iterations}"); break
            next_state = random.choice(neighbors);
            next_h = self.heuristic(next_state);
            delta_e = current_h - next_h
            accept_probability = 1.0
            if delta_e <= 0:
                 try:
                     if temp > 1e-9: accept_probability = math.exp(delta_e / temp)
                     else: accept_probability = 0.0
                 except OverflowError: accept_probability = 0.0
            if accept_probability >= random.random():
                current_state = next_state;
                current_h = next_h;
                current_tuple = state_to_tuple(current_state)
                if current_tuple != last_accepted_tuple:
                    path_taken.append([r[:] for r in current_state]);
                    last_accepted_tuple = current_tuple
            temp *= cooling_rate
        final_h = self.heuristic(current_state)
        goal_reached_str = " (Đích)" if self.is_goal(current_state) else ""
        print(f"SA: Kết thúc tại vòng lặp {iterations}, Nhiệt độ={temp:.2f}, Heuristic cuối={final_h}{goal_reached_str}");
        return path_taken

    # --- Thuật toán Genetic Algorithm (trả về trạng thái tốt nhất) ---
    def genetic_algorithm_solve(self, population_size=60, mutation_rate=0.15, elite_size=5, tournament_k=3):
        print(f"GA: Bắt đầu (Pop={population_size}, Mut={mutation_rate}, Elite={elite_size}, Tourn={tournament_k})...");
        state_to_flat = lambda s: sum(s, [])
        flat_to_state = lambda f: [f[i:i+3] for i in range(0, 9, 3)] if len(f)==9 else None

        population = []
        attempts = 0
        while len(population) < population_size and attempts < population_size * 5:
             state = generate_random_solvable_state()
             if state: population.append(state)
             attempts += 1
        if not population:
             print("GA: Lỗi! Không thể tạo quần thể ban đầu.")
             return None

        best_solution_overall = None
        best_heuristic_overall = float('inf')

        for generation in range(MAX_GA_GENERATIONS):
            population_fitness = []
            for state in population:
                h = self.heuristic(state)
                population_fitness.append({'state': state, 'heuristic': h})
                if h < best_heuristic_overall:
                    best_heuristic_overall = h
                    best_solution_overall = [r[:] for r in state]
                    if best_heuristic_overall == 0:
                        print(f"GA: Đã giải tại thế hệ {generation}!")
                        return [best_solution_overall]

            population_fitness.sort(key=lambda x: x['heuristic'])

            next_population = []
            for i in range(min(elite_size, len(population_fitness))):
                next_population.append(population_fitness[i]['state'])

            def tournament_selection(pop_fit, k):
                if not pop_fit: return generate_random_solvable_state() # Dự phòng
                k = min(k, len(pop_fit)) # Đảm bảo k không lớn hơn số lượng cá thể
                tournament_contenders = random.sample(pop_fit, k)
                tournament_contenders.sort(key=lambda x: x['heuristic'])
                return tournament_contenders[0]['state']

            def cycle_crossover(parent1_state, parent2_state):
                p1 = state_to_flat(parent1_state); p2 = state_to_flat(parent2_state)
                size = len(p1); child1_flat = [-1] * size; child2_flat = [-1] * size
                cycles = []; visited_indices = [False] * size
                for i in range(size):
                    if not visited_indices[i]:
                        current_cycle = []; start_index = i; current_index = i
                        while not visited_indices[current_index]:
                            visited_indices[current_index] = True
                            current_cycle.append(current_index)
                            value_in_p2 = p2[current_index]
                            try: current_index = p1.index(value_in_p2)
                            except ValueError: return parent1_state, parent2_state
                        cycles.append(current_cycle)
                for i, cycle in enumerate(cycles):
                    source1, source2 = (p1, p2) if i % 2 == 0 else (p2, p1)
                    for index in cycle:
                        if 0 <= index < size:
                             child1_flat[index] = source1[index]
                             child2_flat[index] = source2[index]
                child1_state = flat_to_state(child1_flat); child2_state = flat_to_state(child2_flat)
                if not child1_state or not is_valid_state(child1_state): child1_state = parent1_state
                if not child2_state or not is_valid_state(child2_state): child2_state = parent2_state
                return child1_state, child2_state

            def mutate(state):
                flat = state_to_flat(state)
                idx1, idx2 = random.sample(range(len(flat)), 2)
                flat[idx1], flat[idx2] = flat[idx2], flat[idx1]
                mutated_state = flat_to_state(flat)
                return mutated_state if mutated_state and is_valid_state(mutated_state) else state

            while len(next_population) < population_size:
                parent1 = tournament_selection(population_fitness, tournament_k)
                parent2 = tournament_selection(population_fitness, tournament_k)
                child1, child2 = cycle_crossover(parent1, parent2)
                if random.random() < mutation_rate: child1 = mutate(child1)
                if random.random() < mutation_rate: child2 = mutate(child2)
                # Kiểm tra lại tính hợp lệ trước khi thêm
                if child1 and is_valid_state(child1) and len(next_population) < population_size: next_population.append(child1)
                elif len(next_population) < population_size: next_population.append(generate_random_solvable_state()) # Thêm ngẫu nhiên nếu con không hợp lệ
                if child2 and is_valid_state(child2) and len(next_population) < population_size: next_population.append(child2)
                elif len(next_population) < population_size: next_population.append(generate_random_solvable_state())

            population = next_population

        print(f"GA: Đạt giới hạn thế hệ ({MAX_GA_GENERATIONS}). Heuristic tốt nhất: {best_heuristic_overall}")
        return [best_solution_overall] if best_solution_overall else None

# --- END Lớp Puzzle ---


# --- Vẽ Giao diện Pygame thuần túy ---
# draw_grid, draw_buttons, get_button_click, draw_results_area (giữ nguyên)
# ... (Code cho các hàm vẽ này không thay đổi) ...
def draw_grid(screen, state):
    grid_rect = pygame.Rect(GRID_X, GRID_Y, GRID_SIZE, GRID_SIZE)
    pygame.draw.rect(screen, DARK_GREY, grid_rect)

    if not state or not isinstance(state, list) or len(state) != 3 or \
       not all(isinstance(row, list) and len(row) == 3 for row in state):
        err_surf = MSG_FONT.render("Trạng thái không hợp lệ!", True, RED)
        err_rect = err_surf.get_rect(center=grid_rect.center)
        screen.blit(err_surf, err_rect)
        return

    for r in range(3):
        for c in range(3):
            try: num = state[r][c]
            except IndexError: continue

            rect_x = GRID_X + c * TILE_SIZE + 2
            rect_y = GRID_Y + r * TILE_SIZE + 2
            rect = pygame.Rect(rect_x, rect_y, TILE_SIZE - 4, TILE_SIZE - 4)

            tile_color = DARK_GREY
            if num != 0:
                try:
                    color = pygame.Color(0)
                    hue = (num * 35) % 360; saturation = 65 + (num % 2) * 10; lightness = 50 + (num % 3) * 5
                    color.hsla = (hue, saturation, lightness, 100); tile_color = color
                except ValueError: tile_color = BLUE

                pygame.draw.rect(screen, tile_color, rect, border_radius=8)
                pygame.draw.rect(screen, BLACK, rect, 2, border_radius=8)

                text_surf = FONT.render(str(num), True, WHITE)
                text_rect = text_surf.get_rect(center=rect.center)
                shadow_surf = FONT.render(str(num), True, (20, 20, 20))
                shadow_rect = shadow_surf.get_rect(center=(rect.centerx + 2, rect.centery + 2))
                screen.blit(shadow_surf, shadow_rect)
                screen.blit(text_surf, text_rect)
            else:
                 pygame.draw.rect(screen, LIGHT_GREY, rect, border_radius=8)
                 pygame.draw.rect(screen, DARK_GREY, rect, 1, border_radius=8)

def draw_buttons(screen, buttons_config):
    buttons_drawn_info = {}
    panel_rect = pygame.Rect(BUTTON_PANEL_X, BUTTON_PANEL_Y, BUTTON_PANEL_WIDTH, HEIGHT - BUTTON_PANEL_Y - PANEL_PADDING)

    button_width = (BUTTON_PANEL_WIDTH - PANEL_PADDING * 3) // 2
    button_height = 28
    margin = 8
    start_x = BUTTON_PANEL_X + PANEL_PADDING
    msg_height_allowance = 30
    start_y = BUTTON_PANEL_Y + msg_height_allowance + margin

    row, col = 0, 0
    max_algo_y_bottom = start_y

    algo_button_keys = [k for k, v in buttons_config.items() if v['type'] != 'action']
    action_keys = [k for k, v in buttons_config.items() if v['type'] == 'action']

    for key in algo_button_keys:
        data = buttons_config[key]; text = key
        func_name = data['func_name']; algo_type = data['type']
        base_color = data['color']; is_enabled = data.get('enabled', False) # Lấy enabled, mặc định False

        button_x = start_x + col * (button_width + margin)
        button_y = start_y + row * (button_height + margin)
        rect = pygame.Rect(button_x, button_y, button_width, button_height)

        required_action_height = (button_height + margin) if action_keys else 0
        if rect.bottom < panel_rect.bottom - required_action_height:
            button_color = base_color if is_enabled else DARK_GREY
            text_color = WHITE if is_enabled else GREY
            pygame.draw.rect(screen, button_color, rect, border_radius=5)
            pygame.draw.rect(screen, BLACK, rect, 1, border_radius=5)
            label_surf = BUTTON_FONT.render(text, True, text_color)
            label_rect = label_surf.get_rect(center=rect.center)
            screen.blit(label_surf, label_rect)
            buttons_drawn_info[text] = {'rect': rect, 'func': func_name, 'type': algo_type, 'enabled': is_enabled}
            max_algo_y_bottom = max(max_algo_y_bottom, rect.bottom)
            col += 1
            if col >= 2: col = 0; row += 1
        else: pass

    action_start_y = max_algo_y_bottom + margin * 2
    num_action_buttons = len(action_keys)
    if num_action_buttons > 0:
         action_button_width = (BUTTON_PANEL_WIDTH - PANEL_PADDING * (num_action_buttons + 1)) // num_action_buttons
         action_start_x = BUTTON_PANEL_X + PANEL_PADDING

         for i, key in enumerate(action_keys):
            data = buttons_config[key]; text = key
            func_name = data['func_name']; algo_type = data['type']
            base_color = data['color']; is_enabled = data.get('enabled', False) # Lấy enabled
            button_x = action_start_x + i * (action_button_width + margin)
            if action_start_y + button_height < panel_rect.bottom:
                 rect = pygame.Rect(button_x, action_start_y, action_button_width, button_height)
                 button_color = base_color if is_enabled else DARK_GREY
                 text_color = WHITE if is_enabled else GREY
                 pygame.draw.rect(screen, button_color, rect, border_radius=5)
                 pygame.draw.rect(screen, BLACK, rect, 1, border_radius=5)
                 label_surf = BUTTON_FONT.render(text, True, text_color)
                 label_rect = label_surf.get_rect(center=rect.center)
                 screen.blit(label_surf, label_rect)
                 buttons_drawn_info[text] = {'rect': rect, 'func': func_name, 'type': algo_type, 'enabled': is_enabled}

    return buttons_drawn_info

def get_button_click(pos, buttons_drawn_info):
    for text, data in buttons_drawn_info.items():
        if 'rect' in data and data.get('enabled', False) and data['rect'].collidepoint(pos):
            return data.get('func'), data.get('type'), text
    return None, None, None

def draw_results_area(screen, lines):
    results_rect = pygame.Rect(RESULTS_AREA_X, RESULTS_AREA_Y, RESULTS_AREA_WIDTH, RESULTS_AREA_HEIGHT)
    pygame.draw.rect(screen, LIGHT_GREY, results_rect)
    pygame.draw.rect(screen, BLACK, results_rect, 1)

    line_height = RESULTS_FONT.get_linesize() + 2  # Thêm khoảng cách giữa các dòng
    start_y = results_rect.top + 5
    max_lines_display = (results_rect.height - 10) // line_height
    start_render_index = max(0, len(lines) - max_lines_display)  # Hiện các dòng cuối nếu quá dài
    line_count_drawn = 0

    for i in range(start_render_index, len(lines)):
        line = lines[i]
        if line_count_drawn >= max_lines_display:
            break

        # Tính toán độ rộng dòng chữ để đảm bảo không bị tràn
        wrap_width = (RESULTS_AREA_WIDTH - 15) // (RESULTS_FONT.size("X")[0])
        wrapped_lines = textwrap.wrap(line, width=max(10, int(wrap_width)))

        for wrapped_line in wrapped_lines:
            if line_count_drawn >= max_lines_display:
                break
            y_pos = start_y + line_count_drawn * line_height
            line_surf = RESULTS_FONT.render(wrapped_line, True, BLACK)
            line_rect = line_surf.get_rect(topleft=(results_rect.left + 5, y_pos))

            # Đảm bảo dòng không vượt quá vùng hiển thị
            if line_rect.right > results_rect.right:
                line_rect.width = results_rect.width - 10  # Giảm chiều rộng để vừa khung
                line_surf = pygame.transform.scale(line_surf, (line_rect.width, line_rect.height))

            screen.blit(line_surf, line_rect)
            line_count_drawn += 1

    # Hiển thị dấu "..." nếu có dòng bị ẩn ở trên
    if start_render_index > 0:
        more_surf = RESULTS_FONT.render("...", True, DARK_GREY)
        more_rect = more_surf.get_rect(topleft=(results_rect.left + 5, results_rect.top + 1))
        screen.blit(more_surf, more_rect)

# --- Hàm Main ---
def main():
    global current_grid_state

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("8-Puzzle AI Solver - Layout 3 Cột")
    background_color = GREY
    clock = pygame.time.Clock()
    running = True

    # --- Trạng thái ứng dụng ---
    # Thêm trạng thái PLACING_RANDOM
    app_state = "INPUT" # Các trạng thái: INPUT, READY, FAILED, PLACING_RANDOM, RUNNING, VISUALIZING, BENCHMARKING, SOLVED
    current_grid_state = [[0] * 3 for _ in range(3)]
    num_to_place = 1
    user_start_state = None
    selected_algorithm = None
    solution_path = None
    vis_step_index = 0
    puzzle_instance = None
    benchmark_results = []
    message = "Click vao luoi de chon so ."
    message_color = BLACK
    message_timer_end = 0
    results_lines = []
    buttons_info = {}
    final_state_after_vis = None

    # --- Biến cho trạng thái PLACING_RANDOM ---
    placing_target_state = None # Trạng thái đích ngẫu nhiên cần tạo
    placing_current_number = 0 # Số đang được đặt (1-8)

    # --- Cấu hình các nút ---
    # Sửa 'Backtrack' -> không cần enabled ban đầu dựa trên user_start_state
    # Đổi type của Backtrack để xử lý đặc biệt
    buttons_config = {
        'BFS': {'func_name': 'bfs', 'type': 'path', 'color': BLUE, 'enabled': False},
        'UCS': {'func_name': 'ucs', 'type': 'path', 'color': BLUE, 'enabled': False},
        'DFS': {'func_name': 'dfs', 'type': 'path', 'color': BLUE, 'enabled': False},
        'IDDFS': {'func_name': 'iddfs', 'type': 'path', 'color': BLUE, 'enabled': False},
        'Greedy': {'func_name': 'greedy', 'type': 'path', 'color': BLUE, 'enabled': False},
        'A*': {'func_name': 'a_star', 'type': 'path', 'color': PURPLE, 'enabled': False},
        'IDA*': {'func_name': 'ida_star', 'type': 'path', 'color': PURPLE, 'enabled': False},
        'Backtrack': {'func_name': 'backtracking', 'type': 'generate_explore', 'color': DARK_BLUE, 'enabled': True}, # Type mới, luôn enabled khi có thể tương tác
        'CSP Solve': {'func_name': 'csp_solve', 'type': 'explore_path', 'color': DARK_BLUE, 'enabled': False}, # Vẫn cần input
        'DLS Visit': {'func_name': 'and_or_search', 'type': 'path_if_found', 'color': DARK_BLUE, 'enabled': False}, # Cần input
        'Simple HC': {'func_name': 'simple_hill_climbing', 'type': 'local', 'color': DARK_GREEN, 'enabled': False},
        'Steepest HC': {'func_name': 'steepest_ascent_hill_climbing', 'type': 'local', 'color': DARK_GREEN, 'enabled': False},
        'Stoch HC': {'func_name': 'stochastic_hill_climbing', 'type': 'local', 'color': DARK_GREEN, 'enabled': False},
        'Sim Anneal': {'func_name': 'simulated_annealing', 'type': 'local', 'color': DARK_GREEN, 'enabled': False},
        'Beam Srch': {'func_name': 'beam_search', 'type': 'path', 'color': BLUE, 'enabled': False},
        'Genetic': {'func_name': 'genetic_algorithm_solve', 'type': 'state_only', 'color': ORANGE, 'enabled': True},
        'Benchmark': {'func_name': 'benchmark', 'type': 'action', 'color': DARK_GREY, 'enabled': False},
        'Reset': {'func_name': 'reset', 'type': 'action', 'color': DARK_RED, 'enabled': True},
    }
    pathfinding_algos_for_benchmark = [
        'bfs', 'ucs', 'dfs', 'iddfs', 'greedy', 'a_star', 'ida_star',
        'csp_solve', 'and_or_search', 'beam_search' # Bỏ backtrack khỏi benchmark vì nó explore
    ]
    algo_name_map = {data['func_name']: text for text, data in buttons_config.items()}

    # --- Hàm tiện ích nội bộ ---
    def set_message(text, color=BLACK, duration_ms=3500):
        nonlocal message, message_color, message_timer_end
        message = text
        message_color = color
        message_timer_end = pygame.time.get_ticks() + duration_ms if duration_ms >= 0 else float('inf')

    def update_button_states():
        nonlocal buttons_config
        is_ready_for_solve = user_start_state and is_solvable(user_start_state)
        # Thêm PLACING_RANDOM vào các trạng thái không tương tác
        can_interact = app_state not in ["PLACING_RANDOM", "RUNNING", "VISUALIZING", "BENCHMARKING"]

        for text, data in buttons_config.items():
            enabled = False
            func_name = data['func_name']
            algo_type = data['type']

            if can_interact:
                # Các thuật toán cần input từ người dùng
                if algo_type in ['path', 'explore_path', 'path_if_found', 'local']:
                    enabled = is_ready_for_solve
                # Các thuật toán/hành động không cần input người dùng ban đầu
                elif algo_type in ['generate_explore', 'state_only', 'action']:
                    enabled = True # Backtrack, GA, Reset luôn bật khi có thể tương tác
                    if func_name == 'benchmark':
                         enabled = is_ready_for_solve # Benchmark vẫn cần input
                else: enabled = True # Mặc định bật nếu không thuộc các loại trên

            # Xử lý riêng nút Reset
            if func_name == 'reset':
                 enabled = can_interact

            buttons_config[text]['enabled'] = enabled

    def set_results(lines):
        nonlocal results_lines
        results_lines = lines

    def reset_app():
        global current_grid_state
        nonlocal num_to_place, user_start_state, selected_algorithm
        nonlocal solution_path, vis_step_index, puzzle_instance, benchmark_results, app_state
        nonlocal final_state_after_vis
        nonlocal placing_target_state, placing_current_number # Reset biến placement

        print("\n--- Đặt lại Bảng ---")
        current_grid_state = [[0] * 3 for _ in range(3)]
        num_to_place = 1
        user_start_state = None; selected_algorithm = None; solution_path = None
        vis_step_index = 0; puzzle_instance = None; benchmark_results = []
        final_state_after_vis = None
        placing_target_state = None; placing_current_number = 0
        set_message("Click luoi hoac chon Backtrack/Genetic.", BLACK, 5000)
        set_results([])
        app_state = "INPUT"
        update_button_states()

    def run_algorithm(algo_func_name, algo_display_text, start_state_input):
        global current_grid_state
        nonlocal solution_path, vis_step_index, app_state, results_lines, selected_algorithm, puzzle_instance, final_state_after_vis

        print(f"\n--- Đang chạy {algo_display_text} ---")
        selected_algorithm = algo_display_text
        app_state = "RUNNING"
        update_button_states()
        set_message(f"Đang chạy {algo_display_text}...", BLUE, -1)
        solution_path = None; vis_step_index = 0; set_results([])
        final_state_after_vis = "FAILED" # Mặc định

        # --- Vẽ màn hình "Đang chạy" ---
        screen.fill(background_color)
        draw_grid(screen, start_state_input)
        buttons_info_run = draw_buttons(screen, buttons_config)
        draw_results_area(screen, ["Đang chạy...", f"Thuật toán: {algo_display_text}"])
        msg_rect_area_run = pygame.Rect(BUTTON_PANEL_X, BUTTON_PANEL_Y, BUTTON_PANEL_WIDTH, 30)
        msg_surf_run = MSG_FONT.render(f"Đang chạy {algo_display_text}...", True, BLUE, LIGHT_GREY)
        msg_rect_run = msg_surf_run.get_rect(center=msg_rect_area_run.center)
        pygame.draw.rect(screen, LIGHT_GREY, msg_rect_area_run)
        screen.blit(msg_surf_run, msg_rect_run)
        pygame.display.flip()
        pygame.event.pump()

        # --- Thực thi thuật toán ---
        try:
            puzzle_instance = Puzzle(start_state_input, GOAL_STATE) # Khởi tạo puzzle
        except ValueError as e:
             error_msg = f"Lỗi khởi tạo Puzzle: {e}"
             print(error_msg); set_message(error_msg, RED); app_state = "FAILED"
             set_results([f"Trạng thái {algo_display_text}:", "Lỗi trạng thái bắt đầu."])
             update_button_states()
             return # Dừng thực thi

        t_start = time.perf_counter();
        temp_result = None; error_msg = None
        solver_method = getattr(puzzle_instance, algo_func_name, None);
        status_desc = "Lỗi không xác định"
        # Lấy type từ config, sử dụng get để tránh lỗi nếu text không có trong config
        algo_type_config = buttons_config.get(algo_display_text, {})
        algo_type = algo_type_config.get('type', 'unknown') # Lấy type, mặc định là unknown


        if solver_method:
            try: temp_result = solver_method()
            except MemoryError as me: error_msg = f"{algo_display_text}: Lỗi bộ nhớ!"; status_desc = "Lỗi bộ nhớ"; print(f"Lỗi bộ nhớ: {me}")
            except Exception as e: error_msg = f"{algo_display_text}: Lỗi runtime!"; status_desc = "Lỗi runtime"; print(f"--- Lỗi trong {algo_display_text} ---"); traceback.print_exc()
        else: error_msg = f"Lỗi nội bộ: Không tìm thấy hàm giải '{algo_func_name}'!"; status_desc = "Không tìm thấy Solver"
        t_elapsed = time.perf_counter() - t_start

        # --- Xử lý kết quả ---
        result_lines_display = []
        if error_msg:
            print(error_msg); set_message(error_msg, RED); app_state = "FAILED"
            result_lines_display = [f"Trạng thái {algo_display_text}:", error_msg]
            final_state_after_vis = "FAILED"
        elif temp_result is not None and isinstance(temp_result, list) and len(temp_result) > 0:
            # Xử lý dựa trên type LẤY TỪ CONFIG
            if algo_type in ['path', 'path_if_found']:
                 solution_path = temp_result; steps = len(solution_path) - 1
                 res_msg = f"{algo_display_text}: Đã giải! ({steps} bước, {t_elapsed:.3f}s)"
                 print(res_msg); set_message(res_msg, GREEN, -1); app_state = "VISUALIZING"
                 final_state_after_vis = "SOLVED"; moves = get_solution_moves(solution_path)
                 result_lines_display = [f"{algo_display_text} Đã giải ({t_elapsed:.3f}s):", "--- Cac buoc giai ---"] + [f"{i+1}. {m}" for i,m in enumerate(moves)]
                 status_desc = "Da giai"
            # Xử lý cho Backtrack (generate_explore) và CSP Solve (explore_path)
            elif algo_type in ['explore_path', 'generate_explore']:
                 solution_path = temp_result; last_state = solution_path[-1]
                 is_goal = puzzle_instance.is_goal(last_state); steps_explored = len(solution_path)
                 if is_goal:
                     status_desc = "Đã giải (Khám phá)"; res_msg = f"{algo_display_text}: Đã giải! ({steps_explored} trạng thái khám phá, {t_elapsed:.3f}s)"
                     print(res_msg); set_message(res_msg, GREEN, -1); final_state_after_vis = "SOLVED"
                     result_lines_display = [f"{algo_display_text} Đã giải ({t_elapsed:.3f}s):", f"Đã khám phá {steps_explored} trạng thái."]
                 else:
                     status_desc = "Thất bại/Dừng (Khám phá)"; res_msg = f"{algo_display_text}: Thất bại/Dừng ({steps_explored} trạng thái khám phá, {t_elapsed:.3f}s)"
                     print(res_msg); set_message(res_msg, ORANGE, -1); final_state_after_vis = "FAILED"
                     result_lines_display = [f"{algo_display_text} Thất bại/Dừng ({t_elapsed:.3f}s):", f"Đã khám phá {steps_explored} trạng thái.", "Không đạt đích."]
                 app_state = "VISUALIZING"
            elif algo_type == 'local':
                 solution_path = temp_result; final_state=solution_path[-1]; final_h=puzzle_instance.heuristic(final_state)
                 is_goal=puzzle_instance.is_goal(final_state); status_desc="Đã giải (Local)" if is_goal else f"Kết thúc (H={final_h})"
                 steps_local = len(solution_path) - 1
                 res_msg = f"{algo_display_text}: {status_desc} ({steps_local} bước, {t_elapsed:.3f}s)";
                 print(res_msg); set_message(res_msg, GREEN if is_goal else ORANGE, -1)
                 app_state = "VISUALIZING"; final_state_after_vis = "SOLVED" if is_goal else "FAILED"
                 result_lines_display = [f"{algo_display_text} Kết quả ({t_elapsed:.3f}s):", status_desc, f"Số bước: {steps_local}"]
            elif algo_type == 'state_only':
                 final_state=temp_result[0]; final_h=puzzle_instance.heuristic(final_state)
                 is_goal=(final_h==0); status_desc="Đã giải (GA)" if is_goal else f"Tìm thấy tốt nhất (H={final_h})"
                 res_msg = f"{algo_display_text}: {status_desc} ({t_elapsed:.3f}s)";
                 print(res_msg); set_message(res_msg, GREEN if is_goal else ORANGE, -1)
                 current_grid_state = [r[:] for r in final_state]; app_state = "SOLVED" if is_goal else "FAILED"
                 final_state_after_vis = app_state
                 result_lines_display = [f"{algo_display_text} Kết quả ({t_elapsed:.3f}s):", status_desc]
            else:
                 error_msg = f"Lỗi: Loại thuật toán không xử lý được '{algo_type}'"; print(error_msg); set_message(error_msg, RED); app_state = "FAILED"; status_desc = "Loại không xác định"; final_state_after_vis = "FAILED"
        elif temp_result is None or (isinstance(temp_result, list) and not temp_result): # Thất bại
            if algo_type == 'path_if_found': status_desc = f"Thất bại/Giới hạn (DLS)"
            else: status_desc = "Thất bại/Dừng"
            res_msg = f"{algo_display_text}: {status_desc} ({t_elapsed:.3f}s)"; print(res_msg); set_message(res_msg, RED, -1); app_state = "FAILED"; final_state_after_vis = "FAILED"
            if status_desc == "Lỗi không xác định": status_desc = "Thất bại/Dừng"
            result_lines_display = [f"Trạng thái {algo_display_text}:", status_desc, f"(Thời gian: {t_elapsed:.3f}s)"]

        set_results(result_lines_display)
        update_button_states()

    # --- Vòng lặp chính ---
    reset_app()
    while running:
        current_time = pygame.time.get_ticks()
        time_delta = clock.tick(60) / 1000.0

        # --- Xử lý sự kiện ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

            # Chỉ xử lý input khi có thể tương tác
            can_interact_now = app_state not in ["PLACING_RANDOM", "RUNNING", "VISUALIZING", "BENCHMARKING"]
            if can_interact_now:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos; x, y = pos

                    # 1. Click vào lưới (chỉ khi đang INPUT)
                    if GRID_X <= x < GRID_X + GRID_SIZE and GRID_Y <= y < GRID_Y + GRID_SIZE and app_state == "INPUT":
                        r = (y - GRID_Y) // TILE_SIZE; c = (x - GRID_X) // TILE_SIZE
                        if 0 <= r < 3 and 0 <= c < 3:
                            if num_to_place <= 8 and current_grid_state[r][c] == 0:
                                current_grid_state[r][c] = num_to_place; num_to_place += 1
                                if num_to_place == 9:
                                    user_start_state = [row[:] for row in current_grid_state]
                                    if is_solvable(user_start_state): set_message("Bang da san sang! Chon thuat toan/Benchmark.", GREEN, -1); app_state = "READY"
                                    else: set_message("Bang khong giai duoc! Reset.", ORANGE, -1); app_state = "FAILED"
                                    update_button_states()
                                else: set_message(f"Click luoi de dat so {num_to_place}", BLACK, 2000)
                            elif current_grid_state[r][c] != 0: set_message("O da co so!", RED, 2000)
                            elif num_to_place > 8: set_message("Bang day. Chọn Algo/Reset.", BLACK, 3000)

                    # 2. Click vào nút
                    elif BUTTON_PANEL_X <= x < BUTTON_PANEL_X + BUTTON_PANEL_WIDTH and BUTTON_PANEL_Y <= y < HEIGHT - PANEL_PADDING :
                        func_name_clicked, type_clicked, text_clicked = get_button_click(pos, buttons_info)

                        if func_name_clicked == 'reset': reset_app()
                        elif func_name_clicked == 'benchmark':
                            if user_start_state and is_solvable(user_start_state):
                                # --- Chạy Benchmark (Giữ nguyên logic benchmark) ---
                                app_state = "BENCHMARKING"; update_button_states()
                                set_message("Đang Benchmark...", BLUE, -1); set_results(["Ket qua Benchmark:", "Đang chay..."])
                                screen.fill(background_color); draw_grid(screen, user_start_state); buttons_info_bench = draw_buttons(screen, buttons_config); draw_results_area(screen, results_lines)
                                msg_rect_area_bench = pygame.Rect(BUTTON_PANEL_X, BUTTON_PANEL_Y, BUTTON_PANEL_WIDTH, 30); msg_surf_bench = MSG_FONT.render("Đang Benchmark...", True, BLUE, LIGHT_GREY); msg_rect_bench = msg_surf_bench.get_rect(center=msg_rect_area_bench.center); pygame.draw.rect(screen, LIGHT_GREY, msg_rect_area_bench); screen.blit(msg_surf_bench, msg_rect_bench); pygame.display.flip(); pygame.event.pump()
                                start_bench = [r[:] for r in user_start_state]; total_t=0; all_res=[]
                                for algo_func_bench in pathfinding_algos_for_benchmark:
                                    if algo_func_bench in algo_name_map:
                                        algo_txt_bench = algo_name_map[algo_func_bench]
                                        print(f"Benchmark: {algo_txt_bench}"); pygame.event.pump()
                                        puzzle_b = Puzzle(start_bench); solver_b = getattr(puzzle_b, algo_func_bench, None)
                                        path_b, time_b, status_b = None, 0, "Loi"; t_start_b = time.perf_counter()
                                        if solver_b:
                                            try: path_b = solver_b()
                                            except MemoryError: status_b = "Loi bo nho"
                                            except Exception: status_b = "Loi runtime"
                                        else: status_b = "Khong tim thay"
                                        time_b = time.perf_counter() - t_start_b
                                        if status_b == "Lỗi":
                                            if path_b and isinstance(path_b, list) and len(path_b)>0 and is_valid_state(path_b[-1]) and puzzle_b.is_goal(path_b[-1]): status_b="Đã giải"
                                            elif path_b and isinstance(path_b, list) and len(path_b)>0 and is_valid_state(path_b[-1]) and buttons_config[algo_txt_bench]['type']=='explore_path' and puzzle_b.is_goal(path_b[-1]): status_b="Đã giải (Khám phá)"
                                            else: status_b = "That bai /Dung"
                                        steps_b = -1
                                        if "Đa giai" in status_b:
                                            if path_b and isinstance(path_b, list) and len(path_b)>0:
                                                if buttons_config[algo_txt_bench]['type'] in ['path', 'path_if_found']: steps_b = len(path_b)-1
                                                elif buttons_config[algo_txt_bench]['type'] == 'explore_path': steps_b = len(path_b)
                                            else: status_b = "Loi (Path k hợp lệ)"
                                        all_res.append({'name': algo_txt_bench, 'time': time_b, 'steps': steps_b, 'status': status_b}); total_t += time_b
                                        print(f"... {algo_txt_bench}: {status_b}, Buoc / Kham pha: {steps_b}, Thoi gian: {time_b:.4f}s")
                                                            # --- SỬA ĐỔI: HIỂN THỊ KẾT QUẢ BENCHMARK RÚT GỌN ---
                                results_display = ["Ket qua benchmark:", "---------------------------------"]

                                # Sắp xếp kết quả theo tên thuật toán cho nhất quán
                                all_res.sort(key=lambda x: x['name'])

                                # Tạo các dòng hiển thị chỉ gồm Tên và Thời gian
                                # for res in all_res:
                                #     # Định dạng thời gian
                                #     t_str = f"{res['time']:.3f}s" if res['time'] > 0.0001 else "-"

                                #     # Định dạng tên (có thể cắt ngắn nếu muốn)
                                #     name_s = res['name'][:11] + "." if len(res['name']) > 11 else res['name']

                                #     # Tạo chuỗi kết quả rút gọn
                                #     results_display.append(f"- {name_s:<12}: {t_str:>8}") # Căn chỉnh cột

                                # results_display.append("---------------------------------")
                                # Không cần phần gợi ý nữa

                                # Cập nhật giao diện
                                set_results(results_display)
                                set_message(f"Benchmark hoàn thành ({total_t:.2f}s).", BLACK, -1)
                                app_state = "READY"; update_button_states()
                                # --- KẾT THÚC SỬA ĐỔI HIỂN THỊ BENCHMARK ---
                                    
                                    
                                    
                                
                                solved_res = [r for r in all_res if "Đã giải" in r['status'] and r['steps'] != -1 and buttons_config[r['name']]['type'] != 'explore_path']
                                min_s = min((r['steps'] for r in solved_res), default=-1)
                                all_res.sort(key=lambda x: (("Đã giải" not in x['status']), x['name']))
                                for res in all_res:
                                    # Định dạng thời gian
                                    t_str = f"{res['time']:.3f}s" if res['time'] > 0.0001 else "-"
                                    # Định dạng tên
                                    name_s = res['name'][:11] + "." if len(res['name']) > 11 else res['name']
                                    # Tạo chuỗi kết quả rút gọn <<< CHỈ CÓ DÒNG NÀY
                                    results_display.append(f"- {name_s:<12}: {t_str:>8}") # Căn chỉnh cột
                                if solved_res and min_s != -1:
                                    optimal_solvers = [r for r in solved_res if r['steps']==min_s];
                                    if optimal_solvers: optimal_solvers.sort(key=lambda x:x['time']); best = optimal_solvers[0]; sug = f"Goi y: {best['name']} tối ưu ({best['steps']} bước, {best['time']:.3f}s)"
                                    else: sug = "Loi tinh toan goi y."
                                elif solved_res: solved_res.sort(key=lambda x:x['time']); best = solved_res[0]; s_disp = f"{best['steps']} bước" if best['steps']!=-1 else "N/A bước"; sug = f"Goi y: Nhanh nhat: {best['name']} ({s_disp}, {best['time']:.3f}s)"
                                else: sug = "."#khong co goi y
                                results_display.append(sug)
                                set_results(results_display); set_message(f"Benchmark hoàn thành ({total_t:.2f}s).", BLACK, -1); app_state = "READY"; update_button_states()
                            else: set_message("Basng chưa san sang khong Benchmark duoc.", RED)

                        # --- Xử lý nút Backtrack (Loại: generate_explore) ---
                        elif type_clicked == 'generate_explore':
                            print("Nút Backtrack (generate_explore) được nhấn.")
                            set_message("Đang tạo bảng ngẫu nhiên...", BLUE, -1)
                            # Tạo trạng thái mục tiêu ngay lập tức
                            placing_target_state = generate_random_solvable_state()
                            if placing_target_state:
                                # Bắt đầu quá trình hiển thị đặt số
                                current_grid_state = [[0] * 3 for _ in range(3)] # Reset lưới hiển thị
                                placing_current_number = 1 # Bắt đầu từ số 1
                                app_state = "PLACING_RANDOM" # Chuyển sang trạng thái đặt số
                                update_button_states() # Vô hiệu hóa các nút khác
                                # Vẽ lại màn hình để hiển thị lưới trống và thông báo
                                screen.fill(background_color); draw_grid(screen, current_grid_state); buttons_info = draw_buttons(screen, buttons_config); draw_results_area(screen, ["Đang tạo bảng..."]);
                                msg_rect_area_place = pygame.Rect(BUTTON_PANEL_X, BUTTON_PANEL_Y, BUTTON_PANEL_WIDTH, 30); msg_surf_place = MSG_FONT.render("Đang tạo bảng...", True, BLUE, LIGHT_GREY); msg_rect_place = msg_surf_place.get_rect(center=msg_rect_area_place.center); pygame.draw.rect(screen, LIGHT_GREY, msg_rect_area_place); screen.blit(msg_surf_place, msg_rect_place);
                                pygame.display.flip()
                            else:
                                set_message("Lỗi tạo trạng thái ngẫu nhiên!", RED, -1)
                                app_state = "FAILED" # Quay lại trạng thái lỗi nếu không tạo được

                        # --- Xử lý các nút thuật toán còn lại ---
                        elif func_name_clicked:
                            start_s = None
                            # Xác định trạng thái bắt đầu
                            if type_clicked in ['path', 'local', 'path_if_found', 'explore_path']:
                                if user_start_state and is_solvable(user_start_state): start_s = [r[:] for r in user_start_state]
                                else: set_message("Lỗi: Bảng chưa đặt hoặc không giải được!", RED)
                            elif type_clicked == 'state_only': # GA
                                print(f"Lưu ý: {text_clicked} tự tạo trạng thái."); set_message(f"{text_clicked} dùng trạng thái ngẫu nhiên.", BLUE, 4000)
                                start_s = generate_random_solvable_state()
                                if not start_s: set_message("Lỗi tạo trạng thái ngẫu nhiên!", RED); start_s = None
                            else: set_message(f"Lỗi: Loại nút không xác định '{type_clicked}'!", RED)
                            # Chạy thuật toán nếu có trạng thái bắt đầu
                            if start_s: run_algorithm(func_name_clicked, text_clicked, start_s)

        # --- Cập nhật logic & Trạng thái ---

        # NEU: Xử lý trạng thái PLACING_RANDOM
        if app_state == "PLACING_RANDOM":
            if placing_target_state and 1 <= placing_current_number <= 8:
                # Tìm vị trí của số hiện tại trong trạng thái đích
                found_pos = False
                target_r, target_c = -1, -1
                for r in range(3):
                    for c in range(3):
                        if placing_target_state[r][c] == placing_current_number:
                            target_r, target_c = r, c
                            found_pos = True
                            break
                    if found_pos: break

                if found_pos:
                    # Đặt số vào lưới hiển thị
                    current_grid_state[target_r][target_c] = placing_current_number
                    set_message(f"Đang đặt số {placing_current_number}...", BLACK, 500)

                    # --- Vẽ lại màn hình trong lúc đặt số ---
                    screen.fill(background_color)
                    draw_grid(screen, current_grid_state)
                    buttons_info = draw_buttons(screen, buttons_config) # Các nút bị vô hiệu hóa
                    draw_results_area(screen, ["Đang tạo bảng..."])
                    # Vẽ message area
                    msg_rect_area = pygame.Rect(BUTTON_PANEL_X, BUTTON_PANEL_Y, BUTTON_PANEL_WIDTH, 30); pygame.draw.rect(screen, LIGHT_GREY, msg_rect_area); pygame.draw.rect(screen, DARK_GREY, msg_rect_area, 1)
                    msg_surf = MSG_FONT.render(message, True, message_color); msg_rect = msg_surf.get_rect(center=msg_rect_area.center); msg_rect.clamp_ip(msg_rect_area.inflate(-10, -4)); screen.blit(msg_surf, msg_rect)
                    pygame.display.flip()
                    pygame.time.delay(20) # Delay để nhìn thấy số được đặt

                    placing_current_number += 1 # Chuyển sang số tiếp theo
                else: # Lỗi không tìm thấy số (không nên xảy ra)
                    print(f"Lỗi: Không tìm thấy số {placing_current_number} trong trạng thái đích!")
                    set_message("Lỗi trong quá trình tạo bảng!", RED, -1)
                    app_state = "FAILED"; update_button_states()
                    placing_target_state = None # Dừng quá trình

            elif placing_current_number > 8: # Đã đặt xong số 8
                print("Hoàn tất đặt số ngẫu nhiên. Bắt đầu giải Backtrack.")
                # Đảm bảo ô cuối cùng (số 0) cũng được đặt vào grid hiển thị
                # (Thường thì grid đã chứa 0 sẵn từ lúc khởi tạo [[0]*3...])
                try:
                     br, bc = puzzle_instance.get_blank_position(placing_target_state)
                     current_grid_state[br][bc] = 0 # Đảm bảo ô trống đúng vị trí
                except: pass # Bỏ qua nếu lỗi tìm ô trống

                # Gọi hàm chạy thuật toán Backtracking với bảng vừa tạo
                # Sử dụng func_name và text_name từ nút Backtrack đã lưu (nếu cần)
                # Hoặc gọi trực tiếp run_algorithm('backtracking', 'Backtrack', ...)
                run_algorithm('backtracking', 'Backtrack', placing_target_state)
                # run_algorithm sẽ chuyển app_state sang RUNNING -> VISUALIZING/etc.
                placing_target_state = None # Xóa trạng thái đích sau khi dùng
                placing_current_number = 0
            else: # placing_target_state là None (lỗi trước đó)
                 # Không làm gì, chờ reset hoặc thoát
                 pass

        # Xử lý trạng thái VISUALIZING (giữ nguyên logic cũ)
        elif app_state == "VISUALIZING":
            if solution_path and vis_step_index < len(solution_path):
                current_step_state = solution_path[vis_step_index]
                if isinstance(current_step_state, list) and is_valid_state(current_step_state):
                    current_grid_state = current_step_state
                    steps_total = len(solution_path)
                    vis_type = buttons_config.get(selected_algorithm, {}).get('type', 'path')
                    prefix = "Bước"; step_disp = vis_step_index; total_disp = steps_total - 1
                    if vis_type in ['explore_path', 'generate_explore']: prefix = "Khám phá"; step_disp = vis_step_index + 1; total_disp = steps_total
                    elif vis_type == 'local': prefix = "HC/SA Bước"; step_disp = vis_step_index + 1; total_disp = steps_total
                    set_message(f"{prefix}: {step_disp}/{total_disp}", BLACK, 500)
                    vis_step_index += 1
                else:
                    print(f"Lỗi Visualize: Trạng thái không hợp lệ tại bước {vis_step_index}.")
                    set_message("Lỗi hiển thị!", RED, -1)
                    app_state = final_state_after_vis if final_state_after_vis else "FAILED"
                    update_button_states(); solution_path = None
            else: # Kết thúc visualize
                 if solution_path is not None:
                     app_state = final_state_after_vis if final_state_after_vis else "FAILED"
                     # Cập nhật message cuối cùng
                     algo_name_vis = selected_algorithm if selected_algorithm else "Thuật toán"
                     if app_state == "SOLVED":
                         time_str_vis = ""
                         if results_lines and ("Đã giải" in results_lines[0] or "Khám phá" in results_lines[0] or "Kết quả" in results_lines[0]):
                              try:
                                  parts = results_lines[0].split('(');
                                  if len(parts) > 1: time_part = parts[-1].split('s)')[0]; time_str_vis = time_part.split(' ')[-1] + "s" if ' ' in time_part else time_part + "s"
                              except: pass
                         set_message(f"{algo_name_vis} Hoàn thành! ({time_str_vis})", GREEN, -1)
                     elif app_state == "FAILED":
                          set_message(f"{algo_name_vis} Thất bại/Dừng.", RED, -1)
                     update_button_states()
                 solution_path = None; vis_step_index = 0; final_state_after_vis = None; selected_algorithm = None

        # --- Vẽ lại toàn bộ giao diện mỗi frame ---
        screen.fill(background_color)
        draw_grid(screen, current_grid_state)
        buttons_info = draw_buttons(screen, buttons_config) # Lấy rect mới nhất
        draw_results_area(screen, results_lines)

        # --- Vẽ khu vực thông báo ---
        msg_rect_area = pygame.Rect(BUTTON_PANEL_X, BUTTON_PANEL_Y, BUTTON_PANEL_WIDTH, 30)
        pygame.draw.rect(screen, LIGHT_GREY, msg_rect_area)
        pygame.draw.rect(screen, DARK_GREY, msg_rect_area, 1)
        current_msg_text = ""; current_msg_color = BLACK
        # Logic hiển thị message (giữ nguyên)
        if message and current_time < message_timer_end: current_msg_text = message; current_msg_color = message_color
        elif message and app_state not in ["PLACING_RANDOM", "VISUALIZING", "RUNNING", "BENCHMARKING", "INPUT"]: current_msg_text = message; current_msg_color = message_color
        elif app_state == "INPUT":
            if num_to_place <= 8 : current_msg_text = f"Click vao luoi de chon so {num_to_place}"; current_msg_color = BLACK
            else: current_msg_text = "Bảng đầy. Chọn Algo/Reset."; current_msg_color = BLACK
        elif app_state == "PLACING_RANDOM": # Giữ message của placement
             current_msg_text = message; current_msg_color = message_color

        if current_msg_text:
            msg_surf = MSG_FONT.render(current_msg_text, True, current_msg_color)
            msg_rect = msg_surf.get_rect(center=msg_rect_area.center)
            msg_rect.clamp_ip(msg_rect_area.inflate(-10, -4))
            screen.blit(msg_surf, msg_rect)

        # --- Vẽ Ghi Chú Hướng Dẫn (giữ nguyên) ---
        note_y = HEIGHT - PANEL_PADDING - NOTE_FONT.get_linesize() * 5
        note1_surf = NOTE_FONT.render("'Benchmark' so sanh cac thuat toan tim duong.", True, NOTE_COLOR)
        note2_surf = NOTE_FONT.render("'Reset' xoa bang va xem ket qua.", True, NOTE_COLOR)
        screen.blit(note1_surf, (RESULTS_AREA_X + 5, note_y))
        screen.blit(note2_surf, (RESULTS_AREA_X + 5, note_y + NOTE_FONT.get_linesize() + 2))

        pygame.display.flip()

        # --- Điều khiển tốc độ --- (Delay đã có trong PLACING_RANDOM và VISUALIZING)

    pygame.quit()
    print("\nĐa thoat ung dung.")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except Exception as main_exception:
        print("\n--- LỖI KHÔNG XÁC ĐỊNH TRONG LUỒNG CHÍNH ---")
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)