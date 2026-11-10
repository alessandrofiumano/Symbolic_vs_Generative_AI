from pathfinding.core.diagonal_movement import DiagonalMovement
from collections import deque

def dfs(start, goal, grid):
    start.g = 0
    start.f = 0
    start.parent = None

    frontier = deque([start])
    reached = {(start.x, start.y): 0}

    while frontier:
        current_node = frontier.pop()
        if current_node == goal:
            path = []
            while current_node:
                path.append(current_node)
                current_node = current_node.parent
            path.reverse()
            return path, len(path)

        for child in grid.neighbors(current_node, DiagonalMovement.never):

            new_g = current_node.g + 1

            if (child.x, child.y) not in reached or new_g < reached[(child.x, child.y)]:
                reached[(child.x, child.y)] = new_g
                child.g = new_g
                child.f = child.g
                child.parent = current_node
                frontier.append(child)

    return [], 0


# def DFS(start, goal, grid):
#     start.parent = None
#     frontier = [start]
#     if start.x == goal.x and start.y == goal.y: # GOAL CHECK (POSTICIPATO)
#         return [start]
#     while frontier:
#         current = frontier.pop()        # estrae l'ultimo
#         if current.x == goal.x and current.y == goal.y:  # GOAL CHECK (POSTICIPATO)
#             path = []
#             while current is not None:
#                 path.append(current)
#                 current = current.parent
#             path.reverse()
#             return path
#         successors = grid.neighbors(current, diagonal_movement=DiagonalMovement.never)  # su, giù, dx, sx
#         for succ in successors:
#             new_node = GridNode(succ.x, succ.y)
#             new_node.parent = current
#             frontier.append(new_node)
#     return None
