from heapq import heappop, heappush

class PuzzleState:
    def __init__(self, grid, moves=0, parent=None):
        self.grid = grid
        self.blank_pos = grid.index(0)
        self.moves = moves
        self.parent = parent
        self.f = 0 

    def __lt__(self, other):
        return self.f < other.f

def print_grid(grid):
    for i in range(3):
        print(grid[i*3:(i+1)*3])
    print()

def manhattan_distance(grid):
    distance = 0
    goal_positions = {num: (i, j) for i, row in enumerate([[1, 2, 3], [4, 5, 6], [7, 8, 0]]) for j, num in enumerate(row)}
    for i, num in enumerate(grid):
        if num != 0:
            target_x, target_y = goal_positions[num]
            current_x, current_y = divmod(i, 3)
            distance += abs(current_x - target_x) + abs(current_y - target_y)
    return distance

def get_neighbors(state):
    def swap(grid, pos1, pos2):
        new_grid = list(grid)
        new_grid[pos1], new_grid[pos2] = new_grid[pos2], new_grid[pos1]
        return new_grid

    neighbors = []
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    blank_x, blank_y = divmod(state.blank_pos, 3)

    for move in moves:
        new_x, new_y = blank_x + move[0], blank_y + move[1]
        if 0 <= new_x < 3 and 0 <= new_y < 3:
            new_pos = new_x * 3 + new_y
            new_grid = swap(state.grid, state.blank_pos, new_pos)
            neighbors.append(PuzzleState(new_grid, state.moves + 1, state))
    
    return neighbors

def a_star_search(start_grid):
    start_state = PuzzleState(start_grid)
    start_state.f = manhattan_distance(start_grid)
    
    open_set = []
    heappush(open_set, start_state)
    
    closed_set = set()
    
    while open_set:
        current_state = heappop(open_set)
        
        if current_state.grid == [1, 2, 3, 4, 5, 6, 7, 8, 0]:
            path = []
            while current_state:
                path.append(current_state.grid)
                current_state = current_state.parent
            return path[::-1]
        
        closed_set.add(tuple(current_state.grid))
        
        for neighbor in get_neighbors(current_state):
            if tuple(neighbor.grid) in closed_set:
                continue
            
            neighbor.f = neighbor.moves + manhattan_distance(neighbor.grid)
            
            if neighbor not in open_set:
                heappush(open_set, neighbor)
    
    return None

start_grid = [1, 2, 3, 4, 6, 8, 7, 5, 0]
# start_grid = [1, 2, 0, 3, 4, 6, 7, 5, 8]
# start_grid = [8, 6, 7, 2, 5, 4, 3, 0, 1]
solution_path = a_star_search(start_grid)
for step in solution_path:
    print_grid(step)

