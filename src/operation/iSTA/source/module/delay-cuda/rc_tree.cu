/**
 * @file rc_tree.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The rc tree datastrucure implemention for delay calculation.
 * @version 0.1
 * @date 2024-09-25
 */

#include <cuda_runtime.h>

#include <queue>

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
  while (bfs_queue.empty()) {
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

  level_to_points.emplace_back(std::move(points));

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
    std::vector<std::vector<DelayRcPoint*>>& level_to_points,
    std::size_t node_num) {
  std::vector<float> cap_array;
  std::vector<float> load_array(node_num, 0);
  std::vector<std::size_t> parent_pos_array(node_num, 0);

  // children use start and end pair to mark position.
  std::vector<std::size_t> children_pos_array(node_num * 2, 0);

  std::size_t flatten_pos = 0;
  for (auto& points : level_to_points) {
    for (auto* rc_point : points) {
      rc_point->_flatten_pos = flatten_pos++;
      cap_array.emplace_back(rc_point->_cap);

      if (rc_point->_parent) {
        std::size_t parent_pos = rc_point->_parent->_flatten_pos;
        parent_pos_array[flatten_pos] = parent_pos;

        if (children_pos_array[parent_pos * 2] == 0) {
          children_pos_array[parent_pos * 2] = flatten_pos;
        } else {
          children_pos_array[parent_pos * 2 + 1] = flatten_pos;
        }
      }
    }
  }

  std::swap(rc_network->_cap_array, cap_array);
  std::swap(rc_network->_load_array, load_array);
  std::swap(rc_network->_parent_pos_array, parent_pos_array);
  std::swap(rc_network->_children_pos_array, children_pos_array);
}

__global__ void update_load(float* cap_array, float* load_array,
                            std::size_t* parent_pos_array, int start_pos,
                            int num_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t current_pos = start_pos + tid;
  if (tid < num_count) {
    std::size_t parent_pos = parent_pos_array[current_pos];
    float cap = cap_array[current_pos];
    atomicAdd(&load_array[parent_pos], cap);
  }

  //   printf("thread id: %d \n", tid);
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
  std::size_t* parent_pos_array;

  // malloc gpu memory.
  cudaMalloc(&cap_array, node_num * sizeof(int));
  cudaMalloc(&load_array, node_num * sizeof(int));
  cudaMalloc(&parent_pos_array, node_num * sizeof(int));

  // copy cpu data to gpu memory.
  cudaMemcpy(cap_array, rc_network->_cap_array.data(), node_num * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(load_array, rc_network->_load_array.data(),
             node_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(parent_pos_array, rc_network->_parent_pos_array.data(),
             node_num * sizeof(float), cudaMemcpyHostToDevice);

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

  // set the node load.
  for (auto& node : rc_network->_nodes) {
    node->_load = load_array[node->_flatten_pos];
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

////////////////////////////////////////////////////////////////////////////////
// for test code.

// 定义树节点结构体
struct TreeNode {
  int value;
  TreeNode* left;
  TreeNode* right;

  __host__ __device__ TreeNode(int v) : value(v), left(NULL), right(NULL) {}
};

// CUDA kernel for traversing the binary tree
__global__ void traverseTree(TreeNode* node) {
  printf("node %p \n", node);
  if (node == NULL) return;

  // Process the current node (for example, print its value)
  printf("node value %d \n", node->value);
}

// Helper function to create a new tree node
TreeNode* createNode(int value) {
  TreeNode* newNode = (TreeNode*)malloc(sizeof(TreeNode));
  newNode->value = value;
  newNode->left = NULL;
  newNode->right = NULL;
  return newNode;
}

int test() {
  // Create a simple tree
  TreeNode* root = createNode(1);
  root->left = createNode(2);
  root->right = createNode(3);
  root->left->left = createNode(4);
  root->left->right = createNode(5);

  // Allocate memory on the device
  TreeNode* d_root;
  cudaMalloc(&d_root, sizeof(TreeNode));

  // Copy the root node to the device
  cudaMemcpy(d_root, root, sizeof(TreeNode), cudaMemcpyHostToDevice);

  // Launch the kernel
  traverseTree<<<1, 1>>>(d_root);
  cudaDeviceSynchronize();

  // Clean up
  cudaFree(d_root);
  free(root);

  return 0;
}

}  // namespace istagpu