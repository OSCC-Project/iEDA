/**
 * @file AdjListGraphV.h
 * @author Lh
 * @brief
 * @version 0.1
 * @date 2020-11-23
 */

#pragma once

#include <iostream>
#include <queue>

#include "Vector.hh"
namespace ieda {

class Edge;

struct Vertex
{
 public:
  unsigned get_id() { return _id; }
  unsigned _id;
  int _indegree = 0;
  int _outdegree = 0;
  Edge* next;
};

struct Edge
{
 public:
  unsigned get_id() { return _id; }
  unsigned _id;
  int adjvex = 0;
  int weight = 0;
  Edge* next;
};

class Graph
{
 private:
  int numVer;
  int numEdge;

  Vector<Vertex>* adjVector;

  bool* visited;
  std::queue<int> que;

 public:
  explicit Graph(unsigned numVer);
  void createGraph(int tail, int head, int weight);
  ~Graph();
  int getNumVer() { return numVer; }
  int getNumEdge() { return numEdge; }
  void insertEdge(int vertex, int adjvex, int weight);
  void deleteEdge(int tail, int head);
  void setWeight(int tail, int head, int weight);
  bool checkVer(int tail, int head);
  int getIndegree(int vertex);
  int getOutdegree(int vertex);
  Vector<Vector<int>> saveAdjVector();
  void printAdjVector();
  void BFS(int vertex);
  void DFS(int vertex);
  void topologicalSort();
  bool checkLoop();
  // void printMinPath(int vertex);
  // void printMaxPath(int vertex);
  // bool ckLoopDFS(int vis[], int vertex);
  void DijkstraMinLength(int v0);
  void DijkstraMaxLength(int v0);
};
}  // namespace ieda
