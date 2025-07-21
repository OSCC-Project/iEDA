import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

nodes = []
edges = []
resistances = []

with open('/home/taosimin/ir_example/aes/pg_netlist/rc_data.yaml', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('node_'):
            node_name = line.split(':')[0]
            i += 1
            coord_line = lines[i].strip()
            # coord_str = coord_line.split('[')[1].split(']')[0].strip()
            # coord = [int(num) for num in coord_str.split()]
            nodes.append(coord_line)
        elif line.startswith('edge_'):
            edge_name = line.split(':')[0]
            i += 1
            node1_line = lines[i].strip()
            node1 = int(node1_line.split(':')[1].strip())
            i += 1
            node2_line = lines[i].strip()
            node2 = int(node2_line.split(':')[1].strip())
            
            i += 1
            resistance_line = lines[i].strip()
            resistance = float(resistance_line.split(':')[1].strip())
            
            resistances.append(resistance)
            
            edges.append((node1, node2))
        i += 1

# 假设列表为 resistances
max_value = max(resistances)  # 获取最大元素
count_greater_than_10 = sum(1 for x in resistances if x > 10)  # 统计大于 100 的元素个数
count_less_than_1 = sum(1 for x in resistances if x < 1)  # 统计小于 100 的元素个数

print("最大元素:", max_value)
print("总元素个数:", len(resistances))
print("大于 10 的元素个数: %d 占比：%f" % (count_greater_than_10, count_greater_than_10 / len(resistances)))
print("小于 1 的元素个数: %d 占比：%f" % (count_less_than_1, count_less_than_1 / len(resistances)))

def plot_resistance_distribution(resistance_values):
    plt.hist(resistance_values, bins=[0, 0.2, 0.4, 0.8, 1, 10, 50], edgecolor='black')
    plt.title('Resistance Distribution')
    plt.xlabel('Resistance')
    plt.ylabel('Frequency')
    plt.savefig('resistance_estimate.png', dpi=300)
        
plot_resistance_distribution(resistances)


is_plot_topo = False

if is_plot_topo:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.array(nodes)
    connections = edges

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='fuchsia', marker='o')

    for conn in connections:
        start = points[conn[0]]
        end = points[conn[1]]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'lightgreen', linestyle='--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev = 45, azim = 45)
    # plt.show()
    plt.savefig('pg_netlist.png', dpi=300)