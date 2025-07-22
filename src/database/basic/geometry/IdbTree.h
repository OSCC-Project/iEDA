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
#pragma once
/**
 * @project		iDB
 * @file		IdbTree.h
 * @date		25/05/2021
 * @version		0.1
 * @description


        #Describe tree data.
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <iostream>
#include <list>
#include <string>
#include <vector>

using std::list;

namespace idb {


struct TreeNode;               // Define a structure prototype
class Tree;                    // Define a class prototype
class Iterator;                // Define a class prototype
typedef list<TreeNode*> List;  // Rename a node list

TreeNode* clone(TreeNode*, List&, TreeNode*);  // Clone function

struct TreeNode
{

  int _data;                             // Data
  TreeNode* _parent;                     // Parent node
  List _children;                        // Children nodes
  TreeNode(int type = 0, TreeNode* = 0); // Constructor
  void SetParent(TreeNode&);             // Set parent node
  void InsertChildren(TreeNode&);        // Insert child node
};

class Tree
{
 public:
  // Below are constructors and operator overloads

  Tree();                               // Default constructor
  Tree(const Tree&);                    // Copy constructor
  Tree(const int);                      // Parameterized constructor
  Tree(const int, const list<Tree*>&);  // Parameterized constructor
  ~Tree();                              // Destructor
  Tree& operator=(const Tree&);         // Assignment operator overload
  bool operator==(const Tree&);         // Equality operator overload
  bool operator!=(const Tree&);         // Inequality operator overload

  // Below are member functions
  void Clear();          // Clear
  bool IsEmpty() const;  // Check if empty
  int Size() const;      // Calculate number of nodes
  int Leaves();          // Calculate number of leaves
  int Root() const;      // Return root element
  int Height();          // Calculate tree height


  // Below are static member functions
  static bool IsRoot(Iterator);      // Check if root
  static bool isLeaf(Iterator);      // Check if leaf
  static Iterator Parent(Iterator);  // Return parent node
  static int NumChildren(Iterator);  // Return number of children


  // Iterator functions
  Iterator begin();       // Tree Begin
  Iterator end();         // Tree End
  friend class Iterator;  // Iterator SubClass

 private:
  list<TreeNode*> _nodes;         // Node array
  list<TreeNode*>::iterator LIt;  // A node iterator
  int height(TreeNode*);          
  int level(TreeNode*, Iterator);
};

// This is TreeSub Class Iterator
class Iterator
{
 public:

  Iterator();                                  // Default constructor
  Iterator(const Iterator&);                   // Copy constructor
  Iterator(Tree*, TreeNode*);                  // Constructor
  Iterator(Tree*, list<TreeNode*>::iterator);  // Constructor

  // Operator overloads
  void operator=(const Iterator&);   // Assignment operator overload
  bool operator==(const Iterator&);  // Equality operator overload
  bool operator!=(const Iterator&);  // Inequality operator overload
  Iterator& operator++();            // Prefix ++ operator
  Iterator operator++(int);          // Postfix ++ operator
  int operator*() const;             // Get node information
  bool operator!();                  // Assignment operator overload

  typedef list<TreeNode*>::iterator List;
  friend class Tree;

 private:
  Tree* _tree;                     // Tree data
  list<TreeNode*>::iterator _lit;  // List Iterator
};

}  // namespace idb
