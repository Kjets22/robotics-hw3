
import sys
import math
import random

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np

'''
Set up matplotlib to create a plot with an empty square
'''
def setupPlot():
    fig = plt.figure(num=None, figsize=(5, 5), dpi=120, facecolor='w', edgecolor='k')
    plt.autoscale(False)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1)
    ax.set_axis_off()
    ax.add_patch(patches.Rectangle(
        (0,0),   # (x,y)
        1,          # width
        1,          # height
        fill=False
        ))
    return fig, ax

'''
Make a patch for a single polygon
'''
def createPolygonPatch(polygon, color):
    verts = []
    codes= []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0]/10., xy[1]/10.))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=1)

    return patch


'''
Render the problem
'''
def drawProblem(robotStart, robotGoal, polygons):
    fig, ax = setupPlot()
    patch = createPolygonPatch(robotStart, 'green')
    ax.add_patch(patch)
    patch = createPolygonPatch(robotGoal, 'red')
    ax.add_patch(patch)
    for p in range(0, len(polygons)):
        patch = createPolygonPatch(polygons[p], 'gray')
        ax.add_patch(patch)
    plt.show()

'''
Grow a simple RRT
'''
def growSimpleRRT(points):
    # Copy input so we don't modify it in-place unexpectedly
    newPoints = dict(points)

    # Make sure we have the root at (5, 5)
    root_id = None
    for vid, (x, y) in newPoints.items():
        if abs(x - 5.0) < 1e-6 and abs(y - 5.0) < 1e-6:
            root_id = vid
            break
    if root_id is None:
        # If the root is not already in the map, append it.
        root_id = max(newPoints.keys(), default=0) + 1
        newPoints[root_id] = (5.0, 5.0)

    # Tree represented as an adjacency list: vertex -> list of neighbours
    adjListMap = {root_id: []}

    # Convenience: current set of vertices that are already in the tree
    tree_vertices = set([root_id])

    def distance(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    def closest_point_on_segment(p, a, b):
        ax, ay = a
        bx, by = b
        vx = bx - ax
        vy = by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 == 0.0:
            return a, 0.0
        t = ((p[0] - ax) * vx + (p[1] - ay) * vy) / seg_len2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        closest = (ax + t * vx, ay + t * vy)
        return closest, t

    # We use the original keys of the input dict as the sequence of samples.
    sample_ids = sorted(points.keys())

    for sid in sample_ids:
        sample = points[sid]
        # Skip the sample if it is exactly the root
        if sid == root_id and abs(sample[0] - 5.0) < 1e-6 and abs(sample[1] - 5.0) < 1e-6:
            # Ensure the root is present in the adjacency list
            if root_id not in adjListMap:
                adjListMap[root_id] = []
            tree_vertices.add(root_id)
            continue

        # If this is the very first non-root sample, the tree only contains the root.
        # The nearest point is then simply the root itself.
        best_dist = None
        best_type = None  # 'vertex' or 'edge'
        best_vertex = None
        best_edge = None  # (u, v, t, closest_point)

        # Check all existing vertices
        for vid in tree_vertices:
            d = distance(sample, newPoints[vid])
            if best_dist is None or d < best_dist:
                best_dist = d
                best_type = 'vertex'
                best_vertex = vid
                best_edge = None

        # Check all existing edges for a closer point in the middle of an edge
        for u in adjListMap:
            for v in adjListMap[u]:
                if v <= u:
                    # We treat the tree as undirected and only look at each edge once.
                    continue
                p_u = newPoints[u]
                p_v = newPoints[v]
                closest, t = closest_point_on_segment(sample, p_u, p_v)
                d = distance(sample, closest)
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_type = 'edge'
                    best_vertex = None
                    best_edge = (u, v, t, closest)

        # Now attach the sampled point to the nearest location on the tree
        parent_id = None
        if best_type == 'vertex' or best_edge is None:
            # Nearest point is an existing vertex
            parent_id = best_vertex
        else:
            # Nearest point lies in the interior of an edge; we need to
            # split that edge by introducing a new vertex at the closest point.
            u, v, t, closest = best_edge
            # Create a new vertex for the closest point
            new_id = max(newPoints.keys()) + 1
            newPoints[new_id] = closest
            tree_vertices.add(new_id)

            # Replace the edge (u, v) with (u, new_id) and (new_id, v)
            # First remove v from u's adjacency and u from v's adjacency
            if v in adjListMap.get(u, []):
                adjListMap[u].remove(v)
            if u in adjListMap.get(v, []):
                adjListMap[v].remove(u)

            # Add the two new edges
            adjListMap.setdefault(u, []).append(new_id)
            adjListMap.setdefault(v, []).append(new_id)
            adjListMap[new_id] = [u, v]

            parent_id = new_id

        # Finally, connect the sampled point (with its existing label sid) to the tree.
        # Make sure sid exists in the point set.
        if sid not in newPoints:
            newPoints[sid] = sample

        adjListMap.setdefault(parent_id, []).append(sid)
        adjListMap.setdefault(sid, []).append(parent_id)
        tree_vertices.add(sid)

    return newPoints, adjListMap

def basicSearch(tree, start, goal):
    if start == goal:
        return [start]

    if start not in tree or goal not in tree:
        # If either vertex is missing, there is no path to return.
        return []

    visited = set()
    parent = {}
    queue = [start]
    visited.add(start)

    # Standard BFS
    idx = 0
    while idx < len(queue):
        current = queue[idx]
        idx += 1
        if current == goal:
            break
        for nbr in tree.get(current, []):
            if nbr not in visited:
                visited.add(nbr)
                parent[nbr] = current
                queue.append(nbr)

    if goal not in parent and goal != start:
        # Goal not reached
        return []

    # Reconstruct path from goal back to start using parent map
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path

def displayRRTandPath(points, tree, path, robotStart=None, robotGoal=None, polygons=None):
    fig, ax = setupPlot()

    # Draw the environment if provided
    if robotStart is not None and robotGoal is not None and polygons is not None:
        patch = createPolygonPatch(robotStart, 'green')
        ax.add_patch(patch)
        patch = createPolygonPatch(robotGoal, 'red')
        ax.add_patch(patch)
        for poly in polygons:
            patch = createPolygonPatch(poly, 'gray')
            ax.add_patch(patch)

    # Draw all tree edges in black
    for u, nbrs in tree.items():
        if u not in points:
            continue
        x1, y1 = points[u]
        for v in nbrs:
            if v not in points or v < u:
                # Skip edges we've already drawn (treat tree as undirected)
                continue
            x2, y2 = points[v]
            ax.plot([x1 / 10.0, x2 / 10.0],
                    [y1 / 10.0, y2 / 10.0],
                    linewidth=0.5,
                    color='black')

    # Draw the path, if there is one, in orange
    if path is not None and len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            if u in points and v in points:
                x1, y1 = points[u]
                x2, y2 = points[v]
                ax.plot([x1 / 10.0, x2 / 10.0],
                        [y1 / 10.0, y2 / 10.0],
                        linewidth=2.0,
                        color='orange')

    plt.show()
    return

def isCollisionFree(robot, point, obstacles):

    qx, qy = point

    # Compute robot polygon in workspace coordinates
    world_robot = [(qx + vx, qy + vy) for (vx, vy) in robot]

    # 1. Check against workspace boundary [0, 10] x [0, 10]
    for (x, y) in world_robot:
        if x < 0.0 or x > 10.0 or y < 0.0 or y > 10.0:
            return False

    # Helper functions for geometry
    def on_segment(p, q, r, eps=1e-9):
        """Check if point q lies on segment pr."""
        (px, py), (qx, qy), (rx, ry) = p, q, r
        return (min(px, rx) - eps <= qx <= max(px, rx) + eps and
                min(py, ry) - eps <= qy <= max(py, ry) + eps)

    def orientation(p, q, r, eps=1e-9):
        (px, py), (qx, qy), (rx, ry) = p, q, r
        val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        if abs(val) < eps:
            return 0
        return 1 if val > 0 else 2

    def segments_intersect(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        # Special Cases
        # p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if o1 == 0 and on_segment(p1, p2, q1):
            return True

        # p1, q1 and q2 are colinear and q2 lies on segment p1q1
        if o2 == 0 and on_segment(p1, q2, q1):
            return True

        # p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if o3 == 0 and on_segment(p2, p1, q2):
            return True

        # p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]

            # Check if point is exactly on an edge
            if segments_intersect(pt, pt, (x1, y1), (x2, y2)):
                return True

            # Check for ray intersection with edge
            if ((y1 > y) != (y2 > y)):
                x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x_int >= x:
                    inside = not inside
        return inside

    def polys_intersect(poly1, poly2):
        # Check all edge pairs for intersection
        n1 = len(poly1)
        n2 = len(poly2)
        for i in range(n1):
            p1 = poly1[i]
            q1 = poly1[(i + 1) % n1]
            for j in range(n2):
                p2 = poly2[j]
                q2 = poly2[(j + 1) % n2]
                if segments_intersect(p1, q1, p2, q2):
                    return True

        # One polygon fully inside the other?
        if point_in_poly(poly1[0], poly2):
            return True
        if point_in_poly(poly2[0], poly1):
            return True
        return False

    # 2. Check against each obstacle
    for obs in obstacles:
        if polys_intersect(world_robot, obs):
            return False

    # No collisions detected
    return True

def RRT(robot, obstacles, startPoint, goalPoint):

    # Small helper for distances
    def distance(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    # Helper: check that the robot is collision-free along the segment p->q.
    # We simply sample intermediate points on the segment.
    def edge_is_collision_free(p, q):
        steps = max(int(distance(p, q) / 0.2), 1)
        for i in range(steps + 1):
            t = float(i) / float(steps)
            x = p[0] + t * (q[0] - p[0])
            y = p[1] + t * (q[1] - p[1])
            if not isCollisionFree(robot, (x, y), obstacles):
                return False
        return True

    # Ensure the start and goal are collision-free before planning
    if not isCollisionFree(robot, startPoint, obstacles):
        print("Start configuration is in collision; no plan found.")
        return {}, {}, []
    if not isCollisionFree(robot, goalPoint, obstacles):
        print("Goal configuration is in collision; no plan found.")
        return {}, {}, []

    # RRT data structures
    points = {1: startPoint}
    tree = {1: []}
    start_id = 1
    next_id = 2
    goal_id = None

    max_iterations = 5000
    step_size = 0.5
    goal_sample_rate = 0.05  # with this probability we sample the goal directly
    goal_threshold = 0.4     # how close we need to get before trying a direct connection

    for _ in range(max_iterations):
        # Sample a random configuration in the workspace
        if random.random() < goal_sample_rate:
            q_rand = goalPoint
        else:
            q_rand = (random.uniform(0.0, 10.0), random.uniform(0.0, 10.0))

        # Only consider samples that are collision-free themselves
        if not isCollisionFree(robot, q_rand, obstacles):
            continue

        # Find the nearest existing vertex in the tree
        nearest_id = None
        nearest_dist = None
        for vid, cfg in points.items():
            d = distance(cfg, q_rand)
            if nearest_dist is None or d < nearest_dist:
                nearest_dist = d
                nearest_id = vid

        q_near = points[nearest_id]

        # Steer from q_near towards q_rand by at most step_size
        dx = q_rand[0] - q_near[0]
        dy = q_rand[1] - q_near[1]
        d = math.hypot(dx, dy)
        if d < 1e-9:
            continue
        scale = min(step_size, d) / d
        q_new = (q_near[0] + dx * scale, q_near[1] + dy * scale)

        # Check that the new edge is collision-free
        if not edge_is_collision_free(q_near, q_new):
            continue

        new_id = next_id
        next_id += 1
        points[new_id] = q_new
        tree.setdefault(new_id, [])
        tree.setdefault(nearest_id, []).append(new_id)
        tree[new_id].append(nearest_id)

        # Check if we can connect to the goal from q_new
        if distance(q_new, goalPoint) <= goal_threshold and edge_is_collision_free(q_new, goalPoint):
            goal_id = next_id
            points[goal_id] = goalPoint
            tree.setdefault(goal_id, [])
            tree[goal_id].append(new_id)
            tree[new_id].append(goal_id)
            break

    # If we never explicitly created a goal node, we still try to connect to
    # the closest vertex to the goal, if possible.
    if goal_id is None:
        closest_id = None
        closest_dist = None
        for vid, cfg in points.items():
            d = distance(cfg, goalPoint)
            if closest_dist is None or d < closest_dist:
                closest_dist = d
                closest_id = vid
        # Try to connect directly if it is collision-free
        if closest_id is not None and edge_is_collision_free(points[closest_id], goalPoint):
            goal_id = next_id
            points[goal_id] = goalPoint
            tree.setdefault(goal_id, [])
            tree[goal_id].append(closest_id)
            tree.setdefault(closest_id, []).append(goal_id)

    # Compute a path in the tree if we have a goal vertex
    path = []
    if goal_id is not None:
        path = basicSearch(tree, start_id, goal_id)

    # Visualize the resulting RRT and path (tree only; main() draws robot + obstacles)
    displayRRTandPath(points, tree, path)

    return points, tree, path

if __name__ == "__main__":

    # --- read env file as you already do ---
    if len(sys.argv) < 6:
        print("Five arguments required: python rrt.py [env-file] [x1] [y1] [x2] [y2]")
        sys.exit(1)

    filename = sys.argv[1]
    x1 = float(sys.argv[2]); y1 = float(sys.argv[3])
    x2 = float(sys.argv[4]); y2 = float(sys.argv[5])

    lines = [line.rstrip('\n') for line in open(filename)]
    robot = []
    obstacles = []
    for line in range(len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(len(xys)):
            xy = xys[p].split(',')
            polygon.append((float(xy[0]), float(xy[1])))
        if line == 0:
            robot = polygon
        else:
            obstacles.append(polygon)

    print("Robot:")
    print(robot)
    print("Pologonal obstacles:")
    for p in obstacles:
        print(p)
    print("")

    # start/goal robot polygons for visualization
    robotStart = [(vx + x1, vy + y1) for (vx, vy) in robot]
    robotGoal  = [(vx + x2, vy + y2) for (vx, vy) in robot]

    # --- Figure 1: environment only ---
    drawProblem(robotStart, robotGoal, obstacles)

    # --- Task 1: Simple RRT (no obstacles) ---
    points = {
        1: (5, 5),
        2: (7, 8.2),
        3: (6.5, 5.2),
        4: (0.3, 4),
        5: (6, 3.7),
        6: (9.7, 6.4),
        7: (4.4, 2.8),
        8: (9.1, 3.1),
        9: (8.1, 6.5),
        10: (0.7, 5.4),
        11: (5.1, 3.9),
        12: (2, 6),
        13: (0.5, 6.7),
        14: (8.3, 2.1),
        15: (7.7, 6.3),
        16: (7.9, 5),
        17: (4.8, 6.1),
        18: (3.2, 9.3),
        19: (7.3, 5.8),
        20: (9, 0.6),
    }

    print("\nThe input points are:")
    print(points)
    print("")

    points, adjListMap = growSimpleRRT(points)
    path = basicSearch(adjListMap, 1, 20)

    # ✅ Figure 2: simple RRT ONLY (no obstacles)
    displayRRTandPath(points, adjListMap, path)

    # --- Task 5: Full RRT with collision checking ---
    points2, tree2, path2 = RRT(robot, obstacles, (x1, y1), (x2, y2))

    # ✅ Figure 3: full RRT + environment + valid path
    displayRRTandPath(points2, tree2, path2, robotStart, robotGoal, obstacles)
