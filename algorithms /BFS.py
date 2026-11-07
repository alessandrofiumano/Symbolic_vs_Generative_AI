from pathfinding.core.diagonal_movement import DiagonalMovement

def bfs(start, goal, grid):
    start.parent = None
    frontier = [start]
    raggiunti = {(start.x, start.y): start}  # start è il primo in raggiunti
    if start.x == goal.x and start.y == goal.y: # GOAL CHECK
        return [start]
    while frontier:
        current = frontier.pop(0) # LIFO policy
        successors = grid.neighbors(current, diagonal_movement=DiagonalMovement.never)  # su, giù, dx, sx
        for succ in successors:
            if (succ.x, succ.y) not in raggiunti:
                succ.parent = current
                # GOAL CHECK ANTICIPATO
                if succ.x == goal.x and succ.y == goal.y:
                    temp = succ
                    path = []
                    while temp is not None:
                        path.append(temp)
                        temp = temp.parent
                    path.reverse()
                    return path
                raggiunti[(succ.x, succ.y)] = succ
                frontier.append(succ)
    return None