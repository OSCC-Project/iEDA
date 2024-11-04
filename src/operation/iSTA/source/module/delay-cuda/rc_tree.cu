/**
 * @file rc_tree.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The rc tree datastrucure implemention for delay calculation.
 * @version 0.1
 * @date 2024-09-25
 */

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>

#include "common_cuda.hh"
#include "rc_tree.cuh"

#define GPU_ACC

namespace istagpu {

std::unique_ptr<DelayRcNetCommonInfo> DelayRcNet::_delay_rc_net_common_info;

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
  // resistance record the parent resistance with the children.The first one is
  // root, resistance is 0.
  std::vector<float> resistance_array{0.0};
  std::vector<float> delay_array(node_num, 0);
  std::vector<float> ldelay_array(node_num, 0);
  std::vector<float> beta_array(node_num, 0);
  std::vector<float> impulse_array(node_num, 0);

  int flatten_pos = 0;
  for (auto& points : level_to_points) {
    for (auto* rc_point : points) {
      rc_point->_flatten_pos = flatten_pos;
      cap_array.emplace_back(rc_point->_cap);

      if (rc_point->_parent) {
        auto it =
            std::find_if(rc_network->_edges.begin(), rc_network->_edges.end(),
                         [rc_point](auto& edge) {
                           return ((edge->_from == rc_point->_parent) &&
                                   (edge->_to == rc_point));
                         });
        assert(it != rc_network->_edges.end());
        resistance_array.emplace_back((*it)->_resistance);

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
  std::swap(rc_network->_resistance_array, resistance_array);
  std::swap(rc_network->_parent_pos_array, parent_pos_array);
  std::swap(rc_network->_children_pos_array, children_pos_array);
  std::swap(rc_network->_delay_array, delay_array);
  std::swap(rc_network->_ldelay_array, ldelay_array);
  std::swap(rc_network->_beta_array, beta_array);
  std::swap(rc_network->_impulse_array, impulse_array);
}

/**
 * @brief gpu speed up update the load of the rc point.
 *
 * @param cap_array
 * @param load_array
 * @param parent_pos_array
 * @param start_pos
 * @param num_count
 * @return __global__
 */
__global__ void update_load(float* cap_array, float* load_array,
                            int* parent_pos_array, int start_pos,
                            int num_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_count) {
    // printf("thread id: %d, start pos %d\n", tid, start_pos);

    int current_pos = start_pos + tid;
    int parent_pos = parent_pos_array[current_pos];
    float cap = cap_array[current_pos];

    // update the current pos load and parent's load.
    load_array[current_pos] += cap;
    // printf("current pos %d cap: %f \n", current_pos, cap);

    if (parent_pos != current_pos) {
      atomicAdd(&load_array[parent_pos], load_array[current_pos]);
    }

    printf("load array pos %d load: %f \n", parent_pos, load_array[parent_pos]);
  }
}

__global__ void update_delay(float* resistance_array, float* load_array,
                             float* delay_array, int* parent_pos_array,
                             int start_pos, int num_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_count) {
    int current_pos = start_pos + tid;
    if (current_pos == 0) {
      delay_array[current_pos] = 0.0;  // root point delay is 0.0.
    } else {
      int parent_pos = parent_pos_array[current_pos];
      float parent_delay = delay_array[parent_pos];

      float resistance = resistance_array[current_pos];
      float load = load_array[current_pos];
      float delay = parent_delay + resistance * load;
      atomicAdd(&delay_array[current_pos], delay);

      printf("parent pos %d parent delay: %f\n", parent_pos, parent_delay);
    }

    printf("current pos %d resistance: %f delay: %f\n", current_pos,
           resistance_array[current_pos], delay_array[current_pos]);
  }
}

/**
 * @brief gpu speed up update the load delay of the rc point.
 *
 * @param cap_array
 * @param delay_array
 * @param ldelay_array
 * @param parent_pos_array
 * @param start_pos
 * @param num_count
 * @return __global__
 */
__global__ void update_ldelay(float* cap_array, float* delay_array,
                              float* ldelay_array, int* parent_pos_array,
                              int start_pos, int num_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_count) {
    // printf("thread id: %d, start pos %d\n", tid, start_pos);

    int current_pos = start_pos + tid;
    int parent_pos = parent_pos_array[current_pos];
    float cap = cap_array[current_pos];
    float delay = delay_array[current_pos];

    // update the current pos load delay and parent's load delay.
    ldelay_array[current_pos] += cap * delay;
    // printf("current pos %d ldelay: %f \n", current_pos, cap * delay);

    if (parent_pos != current_pos) {
      atomicAdd(&ldelay_array[parent_pos], ldelay_array[current_pos]);
    }

    printf("ldelay array pos %d ldelay: %f \n", parent_pos,
           ldelay_array[parent_pos]);
  }
}

__global__ void update_response(float* resistance_array, float* ldelay_array,
                                float* beta_array, float* delay_array,
                                float* impulse_array, int* parent_pos_array,
                                int start_pos, int num_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < num_count) {
    int current_pos = start_pos + tid;
    if (current_pos == 0) {
      beta_array[current_pos] = 0.0;     // root point beta is 0.0.
      impulse_array[current_pos] = 0.0;  // root point impulse is 0.0.
    } else {
      int parent_pos = parent_pos_array[current_pos];
      float parent_beta = beta_array[parent_pos];

      float resistance = resistance_array[current_pos];
      float ldelay = ldelay_array[current_pos];
      float beta = parent_beta + resistance * ldelay;
      atomicAdd(&beta_array[current_pos], beta);
      printf("parent pos %d parent beta: %f\n", parent_pos, parent_beta);
      float delay = delay_array[current_pos];
      float impulse = 2 * beta - std::pow(delay, 2);
      atomicAdd(&impulse_array[current_pos], impulse);
    }

    printf("current pos %d resistance: %f beta: %f impulse:%f\n", current_pos,
           resistance_array[current_pos], beta_array[current_pos],
           impulse_array[current_pos]);
  }
}

/**
 * @brief init gpu memory.
 *
 * @param rc_network
 */
void delay_init_gpu_memory(DelayRcNetwork* rc_network) {
  auto node_num = rc_network->get_node_num();
  // malloc gpu memory.
  cudaMalloc(&rc_network->_gpu_cap_array, node_num * sizeof(float));
  cudaMalloc(&rc_network->_gpu_load_array, node_num * sizeof(float));
  cudaMalloc(&rc_network->_gpu_resistance_array, node_num * sizeof(float));
  cudaMalloc(&rc_network->_gpu_delay_array, node_num * sizeof(float));
  cudaMalloc(&rc_network->_gpu_ldelay_array, node_num * sizeof(float));
  cudaMalloc(&rc_network->_gpu_beta_array, node_num * sizeof(float));
  cudaMalloc(&rc_network->_gpu_impulse_array, node_num * sizeof(float));
  cudaMalloc(&rc_network->_gpu_parent_pos_array, node_num * sizeof(int));
  // for rc point children, use start and end pair to record the children.
  cudaMalloc(&rc_network->_gpu_children_pos_array, 2 * node_num * sizeof(int));

  // copy cpu data to gpu memory.
  cudaMemcpy(rc_network->_gpu_cap_array, rc_network->_cap_array.data(),
             node_num * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(rc_network->_gpu_resistance_array,
             rc_network->_resistance_array.data(), node_num * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(rc_network->_gpu_parent_pos_array,
             rc_network->_parent_pos_array.data(), node_num * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(rc_network->_gpu_children_pos_array,
             rc_network->_children_pos_array.data(), 2 * node_num * sizeof(int),
             cudaMemcpyHostToDevice);
}

/**
 * @brief free the gpu memory.
 *
 * @param rc_network
 */
void delay_free_gpu_memory(DelayRcNetwork* rc_network) {
  cudaFree(rc_network->_gpu_cap_array);
  cudaFree(rc_network->_gpu_load_array);
  cudaFree(rc_network->_gpu_resistance_array);
  cudaFree(rc_network->_gpu_delay_array);
  cudaFree(rc_network->_gpu_delay_array);
  cudaFree(rc_network->_gpu_ldelay_array);
  cudaFree(rc_network->_gpu_impulse_array);
  cudaFree(rc_network->_gpu_parent_pos_array);
  cudaFree(rc_network->_gpu_children_pos_array);
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
  float* cap_array = rc_network->_gpu_cap_array;
  float* load_array = rc_network->_gpu_load_array;
  int* parent_pos_array = rc_network->_gpu_parent_pos_array;

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
    node->_nload = rc_network->_load_array[node->_flatten_pos];
  }
}

/**
 * @brief update delay level by level.
 *
 */
void delay_update_point_delay(
    DelayRcNetwork* rc_network,
    std::vector<std::vector<DelayRcPoint*>>& level_to_points) {
  float* resistance_array = rc_network->_gpu_resistance_array;
  float* load_array = rc_network->_gpu_load_array;
  float* delay_array = rc_network->_gpu_delay_array;
  int* parent_pos_array = rc_network->_gpu_parent_pos_array;

  int start_pos = 0;
  for (int index = 0; index < level_to_points.size(); ++index) {
    int num_count = level_to_points[index].size();
    int num_blocks =
        (num_count + THREAD_PER_BLOCK_NUM - 1) / THREAD_PER_BLOCK_NUM;
    update_delay<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
        resistance_array, load_array, delay_array, parent_pos_array, start_pos,
        num_count);
    start_pos += num_count;
  }

  auto node_num = rc_network->get_node_num();
  cudaMemcpy(rc_network->_delay_array.data(), delay_array,
             node_num * sizeof(float), cudaMemcpyDeviceToHost);

  // set the node load.
  for (auto& node : rc_network->_nodes) {
    node->_ndelay = rc_network->_delay_array[node->_flatten_pos];
  }
}

/**
 * @brief update load delay level by level.
 *
 * @param level_to_points
 */
void delay_update_point_ldelay(
    DelayRcNetwork* rc_network,
    std::vector<std::vector<DelayRcPoint*>>& level_to_points) {
  auto node_num = rc_network->get_node_num();
  float* cap_array = rc_network->_gpu_cap_array;
  float* delay_array = rc_network->_gpu_delay_array;
  float* ldelay_array = rc_network->_gpu_ldelay_array;
  int* parent_pos_array = rc_network->_gpu_parent_pos_array;

  // update load of the rc node, sum the children cap.
  for (int index = level_to_points.size() - 1; index >= 0; --index) {
    int num_count = level_to_points[index].size();
    int num_blocks =
        (num_count + THREAD_PER_BLOCK_NUM - 1) / THREAD_PER_BLOCK_NUM;
    int start_pos = node_num - num_count;
    node_num -= num_count;

    update_ldelay<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
        cap_array, delay_array, ldelay_array, parent_pos_array, start_pos,
        num_count);

    cudaDeviceSynchronize();
  }

  node_num = rc_network->get_node_num();

  cudaMemcpy(rc_network->_ldelay_array.data(), ldelay_array,
             node_num * sizeof(float), cudaMemcpyDeviceToHost);

  // set the node load delay.
  for (auto& node : rc_network->_nodes) {
    node->_ldelay = rc_network->_ldelay_array[node->_flatten_pos];
  }
}

/**
 * @brief the impulse and second moment of the input response for each rctree
 * level by level.
 *
 */
void delay_update_point_response(
    DelayRcNetwork* rc_network,
    std::vector<std::vector<DelayRcPoint*>>& level_to_points) {
  float* resistance_array = rc_network->_gpu_resistance_array;
  float* ldelay_array = rc_network->_gpu_ldelay_array;
  float* beta_array = rc_network->_gpu_beta_array;
  float* delay_array = rc_network->_gpu_delay_array;
  float* impulse_array = rc_network->_gpu_impulse_array;
  int* parent_pos_array = rc_network->_gpu_parent_pos_array;

  int start_pos = 0;
  for (int index = 0; index < level_to_points.size(); ++index) {
    int num_count = level_to_points[index].size();
    int num_blocks =
        (num_count + THREAD_PER_BLOCK_NUM - 1) / THREAD_PER_BLOCK_NUM;
    update_response<<<num_blocks, THREAD_PER_BLOCK_NUM>>>(
        resistance_array, ldelay_array, beta_array, delay_array, impulse_array,
        parent_pos_array, start_pos, num_count);
    start_pos += num_count;
  }

  auto node_num = rc_network->get_node_num();
  cudaMemcpy(rc_network->_beta_array.data(), beta_array,
             node_num * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(rc_network->_impulse_array.data(), impulse_array,
             node_num * sizeof(float), cudaMemcpyDeviceToHost);

  // set the node beta and impulse.
  for (auto& node : rc_network->_nodes) {
    node->_beta = rc_network->_beta_array[node->_flatten_pos];
    node->_impulse = rc_network->_impulse_array[node->_flatten_pos];
  }
}

void update_rc_timing(DelayRcNet* rc_net) {
  auto& rc_network = rc_net->_rc_network;
  auto level_to_points = delay_levelization(&rc_network);
  delay_change_rc_tree_to_array(&rc_network, level_to_points);
  delay_init_gpu_memory(&rc_network);

  delay_update_point_load(&rc_network, level_to_points);
  delay_update_point_delay(&rc_network, level_to_points);
  delay_update_point_ldelay(&rc_network, level_to_points);
  delay_update_point_response(&rc_network, level_to_points);

  //   // for debug
  //   // for (auto& node : rc_network._nodes) {
  //   //   std::cout << "node: " << node->_cap << ", load: " << node->_nload
  //   //             << ", delay: " << node->_ndelay << ",ldelay: " <<
  //   node->_ldelay
  //   //             << ", beta: " << node->_beta << ", impulse: " <<
  //   node->_impulse
  //   //             << std::endl;
  //   // }

  delay_free_gpu_memory(&rc_network);
}

void make_delay_rct(DelayRcNet* delay_rc_net, RustSpefNet* rust_spef_net) {
  auto& rc_network = delay_rc_net->_rc_network;

  static auto* rc_net_common_info = delay_rc_net->get_rc_net_common_info();
  static auto spef_cap_unit = rc_net_common_info->get_spef_cap_unit();
  static auto uniform_cap_unit = CapacitiveUnit::kPF;

  {
    void* spef_net_conn;
    FOREACH_VEC_ELEM(&(rust_spef_net->_conns), void, spef_net_conn) {
      auto* rust_spef_conn = static_cast<RustSpefConnEntry*>(
          rust_convert_spef_conn(spef_net_conn));

      rc_network.insert_node(rust_spef_conn->_name,
                             ConvertCapUnit(spef_cap_unit, uniform_cap_unit,
                                            rust_spef_conn->_load));
      rust_free_spef_conn(rust_spef_conn);
    }
  }

  {
    void* spef_net_cap;
    FOREACH_VEC_ELEM(&(rust_spef_net->_caps), void, spef_net_cap) {
      auto* rust_spef_cap = static_cast<RustSpefResCap*>(
          rust_convert_spef_net_cap_res(spef_net_cap));

      // Ground cap, otherwise couple cap
      std::string node1 = rust_spef_cap->_node1;
      std::string node2 = rust_spef_cap->_node2;
      if (node2.empty()) {
        rc_network.insert_node(node1,
                               ConvertCapUnit(spef_cap_unit, uniform_cap_unit,
                                              rust_spef_cap->_res_or_cap));
      } else {
        rc_network.insert_node(node1, node2, rust_spef_cap->_res_or_cap);
      }

      rust_free_spef_net_cap_res(rust_spef_cap);
    }
  }

  {
    void* spef_net_res;
    FOREACH_VEC_ELEM(&(rust_spef_net->_ress), void, spef_net_res) {
      auto* rust_spef_res = static_cast<RustSpefResCap*>(
          rust_convert_spef_net_cap_res(spef_net_res));

      std::string node1 = rust_spef_res->_node1;
      std::string node2 = rust_spef_res->_node2;

      rc_network.insert_segment(node1, node2, rust_spef_res->_res_or_cap);

      rust_free_spef_net_cap_res(rust_spef_res);
    }
  }
  rc_network.sync_nodes();
}

void update_rc_tree_info(DelayRcNet* delay_rc_net) {
  Net* net = delay_rc_net->_net;
  auto* driver = net->getDriver();
  auto pin_ports = net->get_pin_ports();

  // fix for net is only driver
  if (pin_ports.size() < 2) {
    return;
  }

  auto& rc_network = delay_rc_net->_rc_network;
  for (auto* pin : pin_ports) {
    if (auto* node = rc_network.rc_node(pin->getFullName()); node) {
      if (pin == driver) {
        rc_network._root = node;
        // node->set_is_root();
      }
      // node->set_obj(pin);
    } else {
      // const auto& nodes = rc_network.get_nodes();
      for (const auto& node : rc_network._nodes) {
        LOG_INFO << node->_node_name;
      }

      LOG_FATAL << "pin " << pin->getFullName() << " can not found in RCTree "
                << net->get_name() << std::endl;
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
    return rc_point->_nload;
  }

  rc_point->_nload += rc_point->_cap;
  for (auto& edge : rc_point->_fanout_edges) {
    if (edge->_to != parent) {
      rc_point->_nload += delay_update_point_load(rc_point, edge->_to);
    }
  }

  rc_point->_is_update_load = true;

  return rc_point->_nload;
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
      edge->_to->_ndelay =
          rc_point->_ndelay + edge->_resistance * edge->_to->_nload;
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

  delay_init_gpu_memory(&rc_network);

  delay_update_point_load(&rc_network, level_to_points);
  delay_update_point_delay(&rc_network, level_to_points);
  delay_update_point_ldelay(&rc_network, level_to_points);
  delay_update_point_response(&rc_network, level_to_points);

  for (auto& node : rc_network._nodes) {
    std::cout << "node: " << node->_cap << ", load: " << node->_nload
              << ", delay: " << node->_ndelay << ",ldelay: " << node->_ldelay
              << ", beta: " << node->_beta << ", impulse: " << node->_impulse
              << std::endl;
  }

  delay_free_gpu_memory(&rc_network);

  return 0;
}

}  // namespace istagpu