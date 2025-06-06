diff --git a//dev/null b/maze_solver.py
index 0000000000000000000000000000000000000000..8ccd3991bc0de9ae1e0c1a400250d67e813f16b1 100644
--- a//dev/null
+++ b/maze_solver.py
@@ -0,0 +1,185 @@
+#!/usr/bin/env python3
+"""Maze solver using multiple algorithms.
+
+This script demonstrates solving a perfect maze using BFS, DFS, Dijkstra,
+and A* search algorithms. The maze is represented as a grid of characters:
+    'S' for the start,
+    'E' for the end,
+    '#' for walls,
+    ' ' (space) for open paths.
+
+Usage:
+    python maze_solver.py [algorithm]
+
+Where [algorithm] is one of: bfs, dfs, dijkstra, astar.
+If not provided, bfs is used by default.
+
+"""
+
+from __future__ import annotations
+import sys
+from collections import deque
+from heapq import heappush, heappop
+from typing import Dict, List, Optional, Tuple
+
+Grid = List[List[str]]
+Position = Tuple[int, int]
+
+
+def parse_maze(text: str) -> Tuple[Grid, Position, Position]:
+    """Parse maze text into a grid, start and end positions."""
+    grid: Grid = []
+    start: Optional[Position] = None
+    end: Optional[Position] = None
+    for r, line in enumerate(text.splitlines()):
+        row = []
+        for c, ch in enumerate(line):
+            if ch == 'S':
+                start = (r, c)
+                row.append(' ')
+            elif ch == 'E':
+                end = (r, c)
+                row.append(' ')
+            else:
+                row.append(ch)
+        grid.append(row)
+    if start is None or end is None:
+        raise ValueError("Maze must have start 'S' and end 'E'")
+    return grid, start, end
+
+
+def neighbors(pos: Position, grid: Grid) -> List[Position]:
+    """Return passable neighbor cells."""
+    r, c = pos
+    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
+    result = []
+    for dr, dc in offsets:
+        nr, nc = r + dr, c + dc
+        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == ' ':
+            result.append((nr, nc))
+    return result
+
+
+def reconstruct_path(came_from: Dict[Position, Position], end: Position) -> List[Position]:
+    path = [end]
+    while path[-1] in came_from:
+        path.append(came_from[path[-1]])
+    path.reverse()
+    return path
+
+
+def bfs(grid: Grid, start: Position, end: Position) -> List[Position]:
+    queue = deque([start])
+    came_from: Dict[Position, Position] = {}
+    visited = {start}
+    while queue:
+        current = queue.popleft()
+        if current == end:
+            return reconstruct_path(came_from, end)
+        for nb in neighbors(current, grid):
+            if nb not in visited:
+                visited.add(nb)
+                came_from[nb] = current
+                queue.append(nb)
+    return []
+
+
+def dfs(grid: Grid, start: Position, end: Position) -> List[Position]:
+    stack = [start]
+    came_from: Dict[Position, Position] = {}
+    visited = {start}
+    while stack:
+        current = stack.pop()
+        if current == end:
+            return reconstruct_path(came_from, end)
+        for nb in neighbors(current, grid):
+            if nb not in visited:
+                visited.add(nb)
+                came_from[nb] = current
+                stack.append(nb)
+    return []
+
+
+def dijkstra(grid: Grid, start: Position, end: Position) -> List[Position]:
+    heap: List[Tuple[int, Position]] = [(0, start)]
+    came_from: Dict[Position, Position] = {}
+    cost: Dict[Position, int] = {start: 0}
+    while heap:
+        dist, current = heappop(heap)
+        if current == end:
+            return reconstruct_path(came_from, end)
+        for nb in neighbors(current, grid):
+            new_cost = dist + 1  # all edges cost 1
+            if nb not in cost or new_cost < cost[nb]:
+                cost[nb] = new_cost
+                came_from[nb] = current
+                heappush(heap, (new_cost, nb))
+    return []
+
+
+def heuristic(a: Position, b: Position) -> int:
+    return abs(a[0] - b[0]) + abs(a[1] - b[1])
+
+
+def astar(grid: Grid, start: Position, end: Position) -> List[Position]:
+    heap: List[Tuple[int, int, Position]] = []
+    heappush(heap, (0 + heuristic(start, end), 0, start))
+    came_from: Dict[Position, Position] = {}
+    cost: Dict[Position, int] = {start: 0}
+    while heap:
+        _, g, current = heappop(heap)
+        if current == end:
+            return reconstruct_path(came_from, end)
+        for nb in neighbors(current, grid):
+            new_cost = g + 1
+            if nb not in cost or new_cost < cost[nb]:
+                cost[nb] = new_cost
+                priority = new_cost + heuristic(nb, end)
+                came_from[nb] = current
+                heappush(heap, (priority, new_cost, nb))
+    return []
+
+
+ALGOS = {
+    'bfs': bfs,
+    'dfs': dfs,
+    'dijkstra': dijkstra,
+    'astar': astar,
+}
+
+SAMPLE_MAZE = """
+########
+#S     #
+### #E##
+#      #
+########
+"""
+
+
+def main(argv: List[str]) -> None:
+    algo_name = argv[1] if len(argv) > 1 else 'bfs'
+    if algo_name not in ALGOS:
+        print(f"Unknown algorithm '{algo_name}'. Available: {', '.join(ALGOS)}")
+        return
+    grid, start, end = parse_maze(SAMPLE_MAZE.strip('\n'))
+    path = ALGOS[algo_name](grid, start, end)
+    if not path:
+        print("No path found")
+        return
+    path_set = set(path)
+    for r, row in enumerate(grid):
+        line = ''
+        for c, ch in enumerate(row):
+            if (r, c) == start:
+                line += 'S'
+            elif (r, c) == end:
+                line += 'E'
+            elif (r, c) in path_set:
+                line += '.'
+            else:
+                line += ch
+        print(line)
+
+
+if __name__ == "__main__":
+    main(sys.argv)
