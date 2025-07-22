import re
import heapq
from typing import Dict, List, Tuple, Optional

class Node:
    def __init__(self, node_id: int, pos: Tuple[float, float, float]):
        self.id = node_id
        self.pos = pos  # (x, y, z) coordinates
        self.edges = []  # List of connected edges

    def __repr__(self):
        return f"Node({self.id}, pos={self.pos})"

class Edge:
    def __init__(self, edge_id: int, node1: int, node2: int, resistance: float):
        self.id = edge_id
        self.node1 = node1
        self.node2 = node2
        self.resistance = resistance

    def __repr__(self):
        return f"Edge({self.id}, {self.node1}->{self.node2}, resistance={self.resistance})"

class Graph:
    def __init__(self):
        self.nodes = {}  # node_id -> Node
        self.edges = {}  # edge_id -> Edge
        self.name_to_id = {}  # node_name -> node_id
        self.id_to_name = {}  # node_id -> node_name

    def add_node(self, node_id: int, name: str, pos: Tuple[float, float, float]):
        self.nodes[node_id] = Node(node_id, pos)
        self.name_to_id[name] = node_id
        self.id_to_name[node_id] = name

    def add_edge(self, edge_id: int, node1_id: int, node2_id: int, resistance: float):
        edge = Edge(edge_id, node1_id, node2_id, resistance)
        self.edges[edge_id] = edge
        
        # Add edge to both nodes
        if node1_id in self.nodes:
            self.nodes[node1_id].edges.append(edge)
        if node2_id in self.nodes:
            self.nodes[node2_id].edges.append(edge)

    def heuristic(self, node1_id: int, node2_id: int) -> float:
        """Euclidean distance as heuristic for A*"""
        pos1 = self.nodes[node1_id].pos
        pos2 = self.nodes[node2_id].pos
        # Manhattan distance
        manhantan_distance =(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])) / 2000
        # Estimate resistance based on distance and z-coordinate        
        estimate = manhantan_distance * 0.03 / 0.15  + abs(pos1[2] - pos2[2]) * 0.02
            
        return estimate

    def a_star(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """A* algorithm to find shortest path based on resistance"""
        open_set = []
        closed_set = set()
        # Push (f_score, node_id) tuple to heap
        # f_score = g_score + heuristic, where:
        # - g_score is the actual cost from start to current node
        # - heuristic is the estimated cost from current to goal
        f_score_start = self.heuristic(start_id, goal_id)  # Initial f_score for start node
        heapq.heappush(open_set, (f_score_start, start_id))
        
        came_from = {}
        g_score = {node_id: float('inf') for node_id in self.nodes}
        g_score[start_id] = 0
        
        f_score = {node_id: float('inf') for node_id in self.nodes}
        f_score[start_id] = f_score_start
        
        while open_set:
            current_f_score, current_id = heapq.heappop(open_set)  # Pop node with lowest f_score
            
            if current_id == goal_id:
                # Reconstruct path
                path = [current_id]
                while current_id in came_from:
                    current_id = came_from[current_id]
                    path.append(current_id)
                return path[::-1]  # Reverse to get start->goal
            
            if current_id in closed_set:  # Skip if already processed
                continue
                
            closed_set.add(current_id)  # Mark as processed
            current_node = self.nodes[current_id]
            
            for edge in current_node.edges:
                neighbor_id = edge.node1 if edge.node2 == current_id else edge.node2
                
                if neighbor_id in closed_set:  # Skip processed neighbors
                    continue
                    
                tentative_g_score = g_score[current_id] + edge.resistance
                
                if tentative_g_score < g_score[neighbor_id]:
                    # This path is better than any previous one
                    came_from[neighbor_id] = current_id
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = tentative_g_score + self.heuristic(neighbor_id, goal_id)
                    heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
        
        return None  # No path found

    def dijkstra(self, start_id: int, goal_id: int) -> Optional[List[int]]:
        """Dijkstra's algorithm implementation by using A* with inf heuristic"""
        def inf_heuristic(node1_id: int, node2_id: int) -> float:
            return float('inf')
            
        # Temporarily replace heuristic function
        original_heuristic = self.heuristic
        self.heuristic = inf_heuristic
        
        # Run A* (which becomes Dijkstra with inf heuristic)
        path = self.a_star(start_id, goal_id)
        
        # Restore original heuristic
        self.heuristic = original_heuristic
        
        return path

    def _dfs(self, start_id: int, visited: set, path: list) -> None:
        """Helper method for DFS traversal (iterative implementation)"""
        stack = [(start_id, 0)]  # Stack of (node_id, edge_index) tuples
        visited.add(start_id)
        path.append(start_id)
        
        while stack:
            node_id, edge_idx = stack.pop()
            
            # If we've processed all edges for this node, backtrack
            if edge_idx >= len(self.nodes[node_id].edges):
                path.pop()  # Remove this node from path when we're done processing it
                continue
                
            edge = self.nodes[node_id].edges[edge_idx]
            next_id = edge.node1 if edge.node2 == node_id else edge.node2
            
            # Push the current node back to stack with next edge index
            stack.append((node_id, edge_idx + 1))
            
            if next_id not in visited:
                visited.add(next_id)
                path.append(next_id)
                # Push the next node to process
                stack.append((next_id, 0))

    def is_connected(self) -> Tuple[bool, Optional[List[int]]]:
        """Check if the graph is connected
        Returns:
            Tuple[bool, Optional[List[int]]]: (is_connected, disconnected_nodes)
            - is_connected: True if graph is connected
            - disconnected_nodes: List of node IDs that are not reachable from start node
        """
        if not self.nodes:
            return True, None
            
        # Start DFS from first node
        start_id = next(iter(self.nodes))
        visited = set()
        path = []
        try:
            self._dfs(start_id, visited, path)
        except Exception as e:
            print(f"Exception occurred: {e}")
            print("Path traversed:", " -> ".join([self.id_to_name[node_id] for node_id in path]))
            raise  # 重新抛出异常以便外部处理
        finally:
            print("graph traversal complete.")  # 回溯时移除节点
        
        # Check if all nodes were visited
        all_nodes = set(self.nodes.keys())
        disconnected = all_nodes - visited
        
        return len(disconnected) == 0, list(disconnected) if disconnected else None

def parse_input(data: str) -> Graph:
    """Parse input data to construct graph"""
    graph = Graph()
    node_pattern = re.compile(r'node_(\d+):\s*\n\s*([\w/:_-]+):\s*\[\s*(\d+)\s+(\d+)\s+(\d+)\s*\]')
    edge_pattern = re.compile(r'edge_(\d+):\s*\n\s*node1:\s*(\d+)\s*\n\s*node2:\s*(\d+)\s*\n\s*resistance:\s*([\d.]+)')
    
    # Parse nodes
    for match in node_pattern.finditer(data):
        node_id = int(match.group(1))
        node_name = match.group(2)
        x = float(match.group(3))
        y = float(match.group(4))
        z = float(match.group(5))
        graph.add_node(node_id, node_name, (x, y, z))
    
    # Parse edges
    for match in edge_pattern.finditer(data):
        edge_id = int(match.group(1))
        node1_id = int(match.group(2))
        node2_id = int(match.group(3))
        resistance = float(match.group(4))
        graph.add_edge(edge_id, node1_id, node2_id, resistance)
    
    # Check graph connectivity after parsing
    is_connected, disconnected_nodes = graph.is_connected()
    if not is_connected:
        print(f"Warning: Graph is not fully connected!")
        print(f"Disconnected nodes: {[graph.id_to_name[nid] for nid in disconnected_nodes]}")
    
    return graph

# Example usage
if __name__ == "__main__":
    with open('/home/taosimin/iEDA24/iEDA/bin/aes_pg_netlist_06_23.yaml', 'r', encoding='utf-8') as file:
        input_data = file.read()

        graph = parse_input(input_data)
        
        start_node = graph.name_to_id.get('VDD_bump', None)
        goal_node = graph.name_to_id.get('core/dec_block/U723:VDD', None)
        
        path = graph.dijkstra(start_node, goal_node)
        if path:
            print(f"最短路径: {' -> '.join(map(graph.id_to_name.get, path))}")
            
            # Calculate total resistance
            total_resistance = 0
            for i in range(len(path) - 1):
                node1 = path[i]
                node2 = path[i+1]
                # Find the edge between node1 and node2
                for edge in graph.nodes[node1].edges:
                    if (edge.node1 == node1 and edge.node2 == node2) or (edge.node1 == node2 and edge.node2 == node1):
                        total_resistance += edge.resistance
                        break
            
            print(f"总电阻: {total_resistance:.6f}")
        else:
            print("未找到路径")