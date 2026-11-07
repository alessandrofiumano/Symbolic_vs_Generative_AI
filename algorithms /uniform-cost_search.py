from pathfinding.core.diagonal_movement import DiagonalMovement

def dijkstra(start, goal, grid):
    start.parent = None
    frontier = [start]
    raggiunti = {(start.x, start.y): start}  # start è il primo in raggiunti
    start.g = 0
    if start.x == goal.x and start.y == goal.y: # GOAL CHECK
        return [start]
    while frontier:
        current = min(frontier, key= lambda node: node.g)
        frontier.remove(current)

        # GOAL CHECK POSTICIPATO
        if current.x == goal.x and current.y == goal.y:
            temp = current
            path = []
            while temp is not None:
                path.append(temp)
                temp = temp.parent
            path.reverse()
            return path

        successors = grid.neighbors(current, diagonal_movement=DiagonalMovement.never)  # su, giù, dx, sx
        for succ in successors:
            newg = current.g + 1
            if not (succ.x, succ.y) in raggiunti or newg < raggiunti[(succ.x, succ.y)].g:
                succ.parent = current
                succ.g = newg
                raggiunti[(succ.x, succ.y)] = succ
                frontier.append(succ)
    return None