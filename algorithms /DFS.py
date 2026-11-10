
from pathfinding.core.diagonal_movement import DiagonalMovement
from collections import deque

def dfs(grid, start, goal):
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
