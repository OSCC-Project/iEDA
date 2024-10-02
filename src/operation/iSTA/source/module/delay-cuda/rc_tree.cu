/**
 * @file rc_tree.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The rc tree datastrucure implemention for delay calculation.
 * @version 0.1
 * @date 2024-09-25
 */

#include <iostream>
#include <queue>

#include "common_cuda.hh"
#include "rc_tree.cuh"

#define GPU_ACC

namespace istagpu {

#if 1

const int THREAD_PER_BLOCK_NUM = 64;

/**
 * @brief levelization the rc tree.
 *
 * @param bfs_queue
 * @param level_to_points
 */
void delay_levelization(
    std::queue<DelayRcPoint*> bfs_queue,
    std::vector<std::vector<DelayRcPoint*>>& level_to_points) {
  std::queue<DelayRcPoint*> next_bfs_queue;
  std::vector<DelayRcPoint*> points;
  while (!bfs_queue.empty()) {
    auto* rc_point = bfs_queue.front();
    bfs_queue.pop();

    for (auto* fanout_edge : rc_point->_fanout_edges) {
      if (fanout_edge->_to != rc_point->_parent) {
        fanout_edge->_to->_parent = rc_point;
        next_bfs_queue.push(fanout_edge->_to);
        points.emplace_back(fanout_edge->_to);
      }
    }
  }

  if (!points.empty()) {
    level_to_points.emplace_back(std::move(points));
  }

  if (!next_bfs_queue.empty()) {
    delay_levelization(next_bfs_queue, level_to_points);
  }
}

/**
 * @brief levelization the rc tree.
 *
 * @param rc_network
 */
std::vector<std::vector<DelayRcPoint*>> delay_levelization(
    DelayRcNetwork* rc_network) {
  auto* root = rc_network->_root;

  std::vector<std::vector<DelayRcPoint*>> level_to_points;
  std::vector<DelayRcPoint*> points{root};
  level_to_points.emplace_back(points);

  std::queue<DelayRcPoint*> bfs_queue;
  bfs_queue.push(root);

  delay_levelization(std::move(bfs_queue), level_to_points);

  return level_to_points;
}

/**
 * @brief change the rc tree to the array, record the parent and children
 * positions.
 *
 */
void delay_change_rc_tree_to_array(
    DelayRcNetwork* rc_network,
    std::vector<std::vector<DelayRcPoint*>>& level_to_points) {
  int node_num = rc_network->get_node_num();
  std::vector<float> cap_array;
  std::vector<float> load_array(node_num, 0);
  std::vector<int> parent_pos_array(node_num, 0);

  // children use start and end pair to mark position.
  std::vector<int> children_pos_array(node_num * 2, 0);

  int flatten_pos = 0;
  for (auto& points : level_to_points) {
    for (auto* rc_point : points) {
      rc_point->_flatten_pos = flatten_pos;
      cap_array.emplace_back(rc_point->_cap);

      if (rc_point->_parent) {
        int parent_pos = rc_point->_parent->_flatten_pos;
        parent_pos_array[flatten_pos] = parent_pos;

        if (children_pos_array[parent_pos * 2] == 0) {
          children_pos_array[parent_pos * 2] = flatten_pos;
        } else {
          children_pos_array[parent_pos * 2 + 1] = flatten_pos;
        }
      }

      ++flatten_pos;
    }
  }

  std::swap(rc_network->_cap_array, cap_array);
  std::swap(rc_network->_load_array, load_array);
  std::swap(rc_network->_parent_pos_array, parent_pos_array);
  std::swap(rc_network->_children_pos_array, children_pos_array);
}

__global__ void update_load(float* cap_array, float* load_array,
                            int* parent_pos_array, int start_pos,
                            int num_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_count) {
    printf("thread id: %d, start pos %d\n", tid, start_pos);

    int current_pos = start_pos + tid;
    int parent_pos = parent_pos_array[current_pos];
    float cap = cap_array[current_pos];

    // update the current pos load and parent's load.
    load_array[current_pos] += cap;
    printf("current pos %d cap: %f \n", current_pos, cap);

    if (parent_pos != current_pos) {
      atomicAdd(&load_array[parent_pos], load_array[current_pos]);
    }

    printf("load array pos %d load: %f \n", parent_pos, load_array[parent_pos]);
  }
}

/**
 * @brief update load level by level.
 *
 * @param level_to_points
 */
void delay_update_point_load(
    DelayRcNetwork* rc_network,
    std::vector<std::vector<DelayRcPoint*>>& level_to_points) {
  auto node_num = rc_network->get_node_num();

  float* cap_array;
  float* load_array;
  int* parent_pos_array;

  // malloc gpu memory.
  cudaMalloc(&cap_array, node_num * sizeof(float));
  cudaMalloc(&load_array, node_num * sizeof(float));
  cudaMalloc(&parent_pos_array, node_num * sizeof(int));

  // copy cpu data to gpu memory.
  cudaMemcpy(cap_array, rc_network->_cap_array.data(), node_num * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(parent_pos_array, rc_network->_parent_pos_array.data(),
             node_num * sizeof(int), cudaMemcpyHostToDevice);

  // update load of the rc node, sum the children cap.
  for (int index = level_to_points.size() - 1; index >= 0; --index) {
    int num_count = level_to_points[index].size();
    int num_blocks =
        (num_count + THREAD_PER_BLOCK_NUM - 1) / THREAD_PER_BLOCK_NUM;
    int start_pos = node_num - num_count;
    node_num -= num_count;

    update_load<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
        cap_array, load_array, parent_pos_array, start_pos, num_count);

    cudaDeviceSynchronize();
  }

  node_num = rc_network->get_node_num();

  cudaMemcpy(rc_network->_load_array.data(), load_array,
             node_num * sizeof(float), cudaMemcpyDeviceToHost);

  // set the node load.
  for (auto& node : rc_network->_nodes) {
    node->_load = rc_network->_load_array[node->_flatten_pos];
  }

  // free the gpu memory.
  cudaFree(cap_array);
  cudaFree(load_array);
  cudaFree(parent_pos_array);
}

#else

/**
 * @brief update the point load of the rc tree.
 *
 * @param rc_point
 */
float delay_update_point_load(DelayRcPoint* parent, DelayRcPoint* rc_point) {
  if (rc_point->_is_update_load) {
    return rc_point->_load;
  }

  rc_point->_load += rc_point->_cap;
  for (auto& edge : rc_point->_fanout_edges) {
    if (edge->_to != parent) {
      rc_point->_load += delay_update_point_load(rc_point, edge->_to);
    }
  }

  rc_point->_is_update_load = true;

  return rc_point->_load;
}

/**
 * @brief update the point load of the rc tree.
 *
 * @param rc_net
 */
void delay_update_point_load(DelayRcNet* rc_net) {
  auto& rc_network = rc_net->_rc_network;
  auto* root = rc_network._root;
  delay_update_point_load(nullptr, root);
}

/**
 * @brief update the point delay of the rc tree.
 *
 * @param rc_net
 * @return float
 */
void delay_update_point_delay(DelayRcPoint* parent, DelayRcPoint* rc_point) {
  if (rc_point->_is_update_delay) {
    return;
  }

  rc_point->_is_update_delay = true;
  for (auto& edge : rc_point->_fanout_edges) {
    if (edge->_to != parent) {
      edge->_to->_delay =
          rc_point->_delay + edge->_resistance * edge->_to->_load;
      delay_update_point_delay(rc_point, edge->_to);
    }
  }
}

/**
 * @brief update the point delay of the rc tree.
 *
 * @param rc_net
 */
void delay_update_point_delay(DelayRcNet* rc_net) {
  auto& rc_network = rc_net->_rc_network;
  auto* root = rc_network._root;
  delay_update_point_delay(nullptr, root);
}

#endif

// Helper function to create a new tree node
DelayRcPoint* create_node(int cap, DelayRcNetwork* rc_network) {
  DelayRcPoint* new_node = new DelayRcPoint();
  new_node->_cap = cap;
  rc_network->_nodes.emplace_back(new_node);
  return new_node;
}

DelayRcEdge* create_edge(DelayRcPoint* from, DelayRcPoint* to, float resistance,
                         DelayRcNetwork* rc_network) {
  DelayRcEdge* new_edge = new DelayRcEdge();
  new_edge->_from = from;
  new_edge->_to = to;
  from->_fanout_edges.push_back(new_edge);
  to->_fanin_edges.push_back(new_edge);

  new_edge->_resistance = resistance;
  rc_network->_edges.emplace_back(new_edge);

  return new_edge;
}

////////////////////////////////////////////////////////////////////////////////
// for test code.

int test() {
  // Create a simple tree
  DelayRcNetwork rc_network;
  auto* root_node = create_node(1.0, &rc_network);
  rc_network._root = root_node;
  auto* node1 = create_node(2.0, &rc_network);
  auto* node2 = create_node(3.0, &rc_network);
  create_edge(root_node, node1, 1.0, &rc_network);
  create_edge(root_node, node2, 1.0, &rc_network);

  auto* node3 = create_node(4.0, &rc_network);
  auto* node4 = create_node(5.0, &rc_network);
  create_edge(node1, node3, 2.0, &rc_network);
  create_edge(node1, node4, 2.0, &rc_network);

  auto* node5 = create_node(6.0, &rc_network);
  auto* node6 = create_node(7.0, &rc_network);

  create_edge(node2, node5, 3.0, &rc_network);
  create_edge(node2, node6, 3.0, &rc_network);

  auto level_to_points = delay_levelization(&rc_network);
  delay_change_rc_tree_to_array(&rc_network, level_to_points);

  delay_update_point_load(&rc_network, level_to_points);

  for (auto& node : rc_network._nodes) {
    std::cout << "node: " << node->_cap << ", load: " << node->_load
              << ", delay: " << node->_delay << std::endl;
  }

  return 0;
}

}  // namespace istagpu