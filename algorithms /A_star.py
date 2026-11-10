from pathfinding.core.diagonal_movement import DiagonalMovement

def manhattan_heuristic(current, goal):
    dx = abs(current.x - goal.x)
    dy = abs(current.y - goal.y)
    return dx + dy



def A_star(start, goal, grid):
    start.g = 0
    start.f = manhattan_heuristic(start, goal)
    start.parent = None

    frontier = [start]
    reached = {(start.x, start.y): 0}

    while frontier:
        lowest_f_node = min(frontier, key=lambda n: n.f)
        frontier.remove(lowest_f_node)
        current_node = lowest_f_node

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
                child.f = child.g + manhattan_heuristic(child, goal)
                child.parent = current_node
                frontier.append(child)

    return [], 0
