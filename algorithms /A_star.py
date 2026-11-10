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


# def AStar(start, goal, grid, heuristic):
#     start.g =0
#     start.h = heuristic(start, goal)
#     start.parent = None
#     start.f = start.g + start.h
#     frontier = [start]
#     raggiunti = {(start.x, start.y): start}     # start è il primo in raggiunti
#     while frontier:
#         current = min (frontier, key = lambda node: node.f)
#         frontier.remove(current)
#
#         # GOAL CHECK
#         if current.x == goal.x and current.y == goal.y:
#             path = []
#             while current is not None:
#                 path.append(current)
#                 current = current.parent
#             path.reverse()
#             return path
#
#         successors = grid.neighbors(current, diagonal_movement=DiagonalMovement.never) # su, giù, dx, sx
#         for succ in successors:
#             newg = current.g +1
#             if not (succ.x, succ.y) in raggiunti or newg < raggiunti[(succ.x, succ.y)].g:
#                 succ.parent = current
#                 succ.g = current.g+1        # aggiorno g, comune a tutti gli algoritmi
#                 succ.h = heuristic(succ, goal)
#                 succ.f = succ.g + succ.h
#                 raggiunti[(succ.x, succ.y)] = succ
#                 frontier.append(succ)
#     return None