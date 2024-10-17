#include <iostream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// Function to find all connected components
vector<vector<int32_t>> findConnectedComponents(vector<pair<int32_t, int32_t>>& relations)
{
  unordered_map<int32_t, vector<int32_t>> graph;
  unordered_set<int32_t> visited;

  // Build the graph
  for (const auto& relation : relations) {
    graph[relation.first].push_back(relation.second);
    graph[relation.second].push_back(relation.first);
  }

  vector<vector<int32_t>> connectedComponents;

  // Helper function to perform DFS and find all nodes in the current connected component
  auto dfs = [&](int32_t node, vector<int32_t>& component) {
    stack<int32_t> s;
    s.push(node);
    while (!s.empty()) {
      int32_t current = s.top();
      s.pop();
      if (visited.count(current) == 0) {
        visited.insert(current);
        component.push_back(current);
        for (int32_t neighbor : graph[current]) {
          if (visited.count(neighbor) == 0) {
            s.push(neighbor);
          }
        }
      }
    }
  };

  // Find all connected components
  for (const auto& node : graph) {
    if (visited.count(node.first) == 0) {
      vector<int32_t> component;
      dfs(node.first, component);
      connectedComponents.push_back(component);
    }
  }

  return connectedComponents;
}

int32_t main()
{
  // Example input: vector of pairs representing the relations
  vector<pair<int32_t, int32_t>> relations = {{8, 9}, {1, 2}, {2, 3}, {4, 5}, {7, 6}, {7, 8}};

  // Find all connected components
  vector<vector<int32_t>> connectedComponents = findConnectedComponents(relations);

  // Print the connected components
  for (const auto& component : connectedComponents) {
    for (int32_t node : component) {
      cout << node << " ";
    }
    cout << endl;
  }

  return 0;
}
