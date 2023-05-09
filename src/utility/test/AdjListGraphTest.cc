// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************

#include <iostream>

#include "AdjListGraphV.hh"
#include "gtest/gtest.h"

using ieda::Graph;
#define maxnum 120
#define INF 10000000

TEST(AdjGraphTest, test) {
  Graph graph(12);
  graph.createGraph(0, 3, 1);
  graph.createGraph(0, 8, 6);
  graph.createGraph(0, 1, 2);
  graph.createGraph(0, 11, 4);
  graph.createGraph(0, 2, 3);
  graph.createGraph(8, 11, 5);
  graph.createGraph(8, 9, 6);
  graph.createGraph(8, 10, 7);
  graph.createGraph(3, 4, 9);
  graph.createGraph(1, 2, 8);
  graph.createGraph(10, 5, 13);
  graph.createGraph(2, 4, 10);
  graph.createGraph(2, 6, 11);
  graph.createGraph(2, 7, 12);
  graph.createGraph(5, 7, 14);
  graph.createGraph(4, 6, 8);
  graph.createGraph(9, 11, 10);
  graph.printAdjVector();
  std::cout << "v2 indegree = " << graph.getIndegree(2) << std::endl;
  std::cout << "v2 outdegree = " << graph.getOutdegree(2) << std::endl;
  std::cout << "v4 indegree = " << graph.getIndegree(4) << std::endl;
  std::cout << "v4 indegree = " << graph.getOutdegree(4) << std::endl;
  // graph.deleteEdge(2, 4);
  std::cout << "v2 indegree = " << graph.getIndegree(2) << std::endl;
  std::cout << "v2 outdegree = " << graph.getOutdegree(2) << std::endl;
  std::cout << "v4 indegree = " << graph.getIndegree(4) << std::endl;
  std::cout << "v4 outdegree = " << graph.getOutdegree(4) << std::endl;

  // graph.insertEdge(2, 5, 5);
  // graph.printAdjVector();
  // graph.deleteEdge(2, 5);
  std::cout << "afer DFS" << std::endl;
  graph.DFS(0);
  std::cout << std::endl << "afer BFS" << std::endl;
  graph.BFS(0);
  std::cout << std::endl << "after topological sort" << std::endl;
  std::cout << "v0 indegree = " << graph.getIndegree(0) << std::endl;
  std::cout << "v1 indegree = " << graph.getIndegree(1) << std::endl;
  std::cout << "v2 indegree = " << graph.getIndegree(2) << std::endl;
  std::cout << "v3 indegree = " << graph.getIndegree(3) << std::endl;
  std::cout << "v4 indegree = " << graph.getIndegree(4) << std::endl;
  std::cout << "v5 indegree = " << graph.getIndegree(5) << std::endl;
  std::cout << "v6 indegree = " << graph.getIndegree(6) << std::endl;
  std::cout << "v0 outdegree = " << graph.getOutdegree(0) << std::endl;
  std::cout << "v1 outdegree = " << graph.getOutdegree(1) << std::endl;
  std::cout << "v2 outdegree = " << graph.getOutdegree(2) << std::endl;
  std::cout << "v3 outdegree = " << graph.getOutdegree(3) << std::endl;
  std::cout << "v4 outdegree = " << graph.getOutdegree(4) << std::endl;
  std::cout << "v5 outdegree = " << graph.getOutdegree(5) << std::endl;
  std::cout << "v6 outdegree = " << graph.getOutdegree(6) << std::endl;

  graph.topologicalSort();
  std::cout << std::endl;
  // graph.insertEdge(4, 0, 3);
  // graph.topologicalSort();
  bool loop = graph.checkLoop();
  std::cout << std::endl << loop << std::endl;

  graph.DijkstraMinLength(0);
  std::cout << std::endl;
  graph.DijkstraMaxLength(0);
}