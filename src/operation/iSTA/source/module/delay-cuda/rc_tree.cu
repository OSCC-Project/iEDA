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

__global__ void update_load(DelayRcPoint* rc_points, size_t rc_point_num) {
  int tid = blockIdx.x;
  if (tid < rc_point_num) {
    // DelayRcPoint* rc_point = rc_points[tid];
  }

  //   printf("thread id: %d \n", tid);
}

/**
 * @brief update load level by level.
 *
 * @param level_to_points
 */
void delay_update_point_load(
    std::vector<std::vector<DelayRcPoint*>> level_to_points) {
  for (int level = 0; level < level_to_points.size(); ++level) {
    auto& points = level_to_points[level];
    for (auto* rc_point : points) {
      // delay_update_point_load(rc_point);
    }
  }
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