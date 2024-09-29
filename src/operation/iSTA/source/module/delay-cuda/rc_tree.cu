/**
 * @file rc_tree.cu
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The rc tree datastrucure implemention for delay calculation.
 * @version 0.1
 * @date 2024-09-25
 */

#include <cuda_runtime.h>

#include "rc_tree.cuh"

#define GPU_ACC

namespace istagpu {

__global__ void add(int *a, int *b, int *c) {
  int tid = blockIdx.x;  // this thread handles the data at its thread id
  if (tid < 100) {
    c[tid] = a[tid] * b[tid];
  }

//   printf("thread id: %d \n", tid);
}

/**
 * @brief update the point load of the rc tree.
 *
 * @param rc_point
 */
float delay_update_point_load(DelayRcPoint* rc_point) {
  rc_point->_load += rc_point->_cap;
  #pragma omp parallel for
  for (auto& edge : rc_point->_fanout_edge) {
    rc_point->_load += delay_update_point_load(rc_point);
  }
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
  delay_update_point_load(root);
}

////////////////////////////////////////////////////////////////////
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