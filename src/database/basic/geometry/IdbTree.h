#pragma once
/**
 * iEDA
 * Copyright (C) 2021  PCL
 *
 * This program is free software;
 *
 */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * @project		iDB
 * @file		IdbTree.h
 * @copyright	(c) 2021 All Rights Reserved.
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

  struct TreeNode;                // 定义一个结构体原形
  class Tree;                     // 定义一个类原形
  class Iterator;                 // 定义一个类原形
  typedef list<TreeNode *> List;  // 重命名一个节点链表

  TreeNode *clone(TreeNode *, List &, TreeNode *);  // Clone 复制函数

  struct TreeNode {
    int _data;                        // 数据
    TreeNode *_parent;                // 父节点
    List _children;                   // 子节点
    TreeNode(int, TreeNode *);        // 构造函数
    void SetParent(TreeNode &);       // 设置父节点
    void InsertChildren(TreeNode &);  // 插入子节点
  };

  class Tree {
   public:
    // 下面是构造器和运算符重载
    Tree();                                 // 默认构造函数
    Tree(const Tree &);                     // 复制构造函数
    Tree(const int);                        // 带参数构造函数
    Tree(const int, const list<Tree *> &);  // 带参数构造函数
    ~Tree();                                // 析构函数
    Tree &operator=(const Tree &);          //= 符号运算符重载
    bool operator==(const Tree &);          //== 符号运算符重载
    bool operator!=(const Tree &);          //!= 符号运算符重载

    // 下面是成员函数
    void Clear();          // 清空
    bool IsEmpty() const;  // 判断是否为空
    int Size() const;      // 计算节点数目
    int Leaves();          // 计算叶子数
    int Root() const;      // 返回根元素
    int Height();          // 计算树的高度

    // 下面是静态成员函数
    static bool IsRoot(Iterator);      // 判断是否是根
    static bool isLeaf(Iterator);      // 判断是否是叶子
    static Iterator Parent(Iterator);  // 返回其父节点
    static int NumChildren(Iterator);  // 返回其子节点数目

    // 跌代器函数
    Iterator begin();       // Tree Begin
    Iterator end();         // Tree End
    friend class Iterator;  // Iterator SubClass

   private:
    list<TreeNode *> _nodes;         // 节点数组
    list<TreeNode *>::iterator LIt;  // 一个节点迭代器
    int height(TreeNode *);
    int level(TreeNode *, Iterator);
  };

  // This is TreeSub Class Iterator
  class Iterator {
   public:
    Iterator();                                    // 默认构造函数
    Iterator(const Iterator &);                    // 复制构造函数
    Iterator(Tree *, TreeNode *);                  // 构造函数
    Iterator(Tree *, list<TreeNode *>::iterator);  // 构造函数
    // 运算符重载
    void operator=(const Iterator &);   // 赋值运算符重载
    bool operator==(const Iterator &);  // 关系运算符重载
    bool operator!=(const Iterator &);  // 关系运算符重载
    Iterator &operator++();             // 前缀 ++ 运算符
    Iterator operator++(int);           // 后缀 ++ 运算符
    int operator*() const;              // 获得节点信息
    bool operator!();                   // 赋值运算符重载

    typedef list<TreeNode *>::iterator List;
    friend class Tree;

   private:
    Tree *_tree;                      // Tree data
    list<TreeNode *>::iterator _lit;  // List Iterator
  };

}  // namespace idb
