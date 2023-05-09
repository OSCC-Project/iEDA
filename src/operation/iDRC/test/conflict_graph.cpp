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

#include "MultiPatterning.h"
#include "iostream"
#include "string"
using namespace idrc;

int main(int argc, char* argv[])
{
  // if (argc != 2) {
  //   std::cout << "Please run 'run_drc <drc_config_path>'!" << std::endl;
  //   exit(1);
  // }
  // std::string drc_config_path = argv[1];
  DrcConflictNode* node1 = new DrcConflictNode(1);
  DrcConflictNode* node2 = new DrcConflictNode(2);
  DrcConflictNode* node3 = new DrcConflictNode(3);
  DrcConflictNode* node4 = new DrcConflictNode(4);
  DrcConflictNode* node5 = new DrcConflictNode(5);
  DrcConflictNode* node6 = new DrcConflictNode(6);
  DrcConflictNode* node7 = new DrcConflictNode(7);
  // DrcConflictNode* node8 = new DrcConflictNode(8);
  // DrcConflictNode* node9 = new DrcConflictNode(9);
  // DrcConflictNode* node10 = new DrcConflictNode(10);
  // DrcConflictNode* node11 = new DrcConflictNode(11);

  // DrcConflictNode* node12 = new DrcConflictNode(12);
  // DrcConflictNode* node13 = new DrcConflictNode(13);
  // DrcConflictNode* node14 = new DrcConflictNode(14);
  // DrcConflictNode* node15 = new DrcConflictNode(15);
  // DrcConflictNode* node16 = new DrcConflictNode(16);
  // DrcConflictNode* node17 = new DrcConflictNode(17);

  // node1->addConflictNode(node2);
  // node2->addConflictNode(node1);

  // node1->addConflictNode(node3);
  // node3->addConflictNode(node1);

  // node2->addConflictNode(node4);
  // node4->addConflictNode(node2);

  // node3->addConflictNode(node4);
  // node4->addConflictNode(node3);

  // node1->addConflictNode(node4);
  // node4->addConflictNode(node1);

  // node5->addConflictNode(node6);
  // node6->addConflictNode(node5);

  // node6->addConflictNode(node7);
  // node7->addConflictNode(node6);

  // node7->addConflictNode(node8);
  // node8->addConflictNode(node7);

  // node8->addConflictNode(node9);
  // node9->addConflictNode(node8);

  // node5->addConflictNode(node9);
  // node9->addConflictNode(node5);

  // node10->addConflictNode(node11);
  // node11->addConflictNode(node10);

  // node12->addConflictNode(node13);
  // node13->addConflictNode(node12);

  // node13->addConflictNode(node14);
  // node14->addConflictNode(node13);

  // node12->addConflictNode(node14);
  // node14->addConflictNode(node12);

  // node13->addConflictNode(node15);
  // node15->addConflictNode(node13);

  // node15->addConflictNode(node17);
  // node17->addConflictNode(node15);

  // node15->addConflictNode(node16);
  // node16->addConflictNode(node15);

  // node17->addConflictNode(node16);
  // node16->addConflictNode(node17);

  /////////////////////////////////////////
  ////////////////////////////////////////
  node1->addConflictNode(node2);
  node2->addConflictNode(node1);

  node1->addConflictNode(node4);
  node4->addConflictNode(node1);

  node1->addConflictNode(node3);
  node3->addConflictNode(node1);

  node4->addConflictNode(node3);
  node3->addConflictNode(node4);

  node2->addConflictNode(node3);
  node3->addConflictNode(node2);

  node7->addConflictNode(node4);
  node4->addConflictNode(node7);

  node4->addConflictNode(node5);
  node5->addConflictNode(node4);

  // node5->addConflictNode(node3);
  // node3->addConflictNode(node5);

  node7->addConflictNode(node6);
  node6->addConflictNode(node7);

  node6->addConflictNode(node5);
  node5->addConflictNode(node6);

  node5->addConflictNode(node7);
  node7->addConflictNode(node5);

  node6->addConflictNode(node4);
  node4->addConflictNode(node6);

  node2->addConflictNode(node4);
  node4->addConflictNode(node2);

  MultiPatterning multi_pattening;
  DrcConflictGraph* conflict_graph = new DrcConflictGraph();
  conflict_graph->addNode(node1);
  conflict_graph->addNode(node2);
  conflict_graph->addNode(node3);
  conflict_graph->addNode(node4);
  conflict_graph->addNode(node5);
  conflict_graph->addNode(node6);
  conflict_graph->addNode(node7);
  // conflict_graph->addNode(node6);
  // conflict_graph->addNode(node7);
  // conflict_graph->addNode(node8);
  // conflict_graph->addNode(node9);
  // conflict_graph->addNode(node10);
  // conflict_graph->addNode(node11);

  // conflict_graph->addNode(node12);
  // conflict_graph->addNode(node13);
  // conflict_graph->addNode(node14);
  // conflict_graph->addNode(node15);
  // conflict_graph->addNode(node16);
  // conflict_graph->addNode(node17);

  multi_pattening.set_conflict_graph(conflict_graph);
  // std::vector<std::vector<DrcConflictNode*>> allAddCycles = multi_pattening.checkDoublePatterning ();
  std::vector<DrcConflictNode*> uncolorable_node_list = multi_pattening.checkTriplePatterning();

  // for (auto cycle_node_list : allAddCycles) {
  //   for (size_t i = 0; i < cycle_node_list.size(); ++i) {
  //     if (i == cycle_node_list.size() - 1) {
  //       std::cout << cycle_node_list[i]->get_node_id() << std::endl;
  //     } else {
  //       std::cout << cycle_node_list[i]->get_node_id() << "->";
  //     }
  //   }
  // }

  for (size_t i = 0; i < uncolorable_node_list.size(); ++i) {
    std::cout << "(" << uncolorable_node_list[i]->get_node_id() << "," << uncolorable_node_list[i]->get_color() << ")"
              << " ";
    if (i == uncolorable_node_list.size() - 1) {
      std::cout << std::endl;
    }
  }

  return 0;
}