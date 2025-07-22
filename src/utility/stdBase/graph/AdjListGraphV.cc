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
/**
 * @file AdjListGraphV.cc
 * @author Lh
 * @brief
 * @version 0.1
 * @date 2020-11-23
 */
#include "AdjListGraphV.hh"

#include <iostream>
#include <queue>

#include "List.hh"
#include "Vector.hh"

#define maxnum 120
#define INF 10000000

namespace ieda {

struct Node
{
  int id;
  int w;

  friend bool operator<(struct Node a, struct Node b) { return a.w > b.w; }
};
struct Node1
{
  int id;
  int w;

  friend bool operator<(struct Node1 a, struct Node1 b) { return a.w < b.w; }
};

Graph::Graph(unsigned numVer)
{
  this->numVer = numVer;
  numEdge = 0;
  visited = new bool[numVer];
  adjVector = new Vector<Vertex>(numVer);
  for (unsigned i = 0; i < numVer; i++) {
    Vertex ver = {i, 0, 0, nullptr};
    adjVector->push_back(ver);
  }
}
Graph::~Graph()
{
  Edge *p, *q;
  for (int i = 0; i < numVer; i++) {
    if ((*adjVector)[i].next) {
      p = (*adjVector)[i].next;
      while (p) {
        q = p->next;
        delete p;
        p = q;
      }
    }
  }
  delete adjVector;
}
int Graph::getIndegree(int v)
{
  return (*adjVector)[v]._indegree;
}
int Graph::getOutdegree(int v)
{
  return (*adjVector)[v]._outdegree;
}
bool Graph::checkVer(int tail, int head)
{
  if (tail >= 0 && tail < numVer && head >= 0 && head < numVer)
    return true;
  else
    return false;
}
void Graph::createGraph(int tail, int head, int weight)
{
  insertEdge(tail, head, weight);
}
void Graph::insertEdge(int tail, int head, int weight)
{
  Edge *p, *q, *r;
  p = q = r = nullptr;
  if ((*adjVector)[tail].next) {
    p = (*adjVector)[tail].next;

    while (p && (p->adjvex < head)) {
      q = p;
      p = p->next;
    }
    if (p && (p->adjvex == head))
      p->weight = weight;
    else {
      r = new Edge;
      r->adjvex = head;
      r->weight = weight;
      r->next = p;

      if ((*adjVector)[tail].next == p)
        (*adjVector)[tail].next = r;
      else
        q->next = r;
      numEdge++;
      (*adjVector)[tail]._outdegree++;
      (*adjVector)[head]._indegree++;
    }
  } else {
    p = new Edge;
    p->adjvex = head;
    p->weight = weight;
    p->next = nullptr;
    (*adjVector)[tail].next = p;
    numEdge++;
    (*adjVector)[tail]._outdegree++;
    (*adjVector)[head]._indegree++;
  }
}
Vector<Vector<int>> Graph::saveAdjVector()
{
  Vector<int>* adj;
  Vector<Vector<int>> saveAdj;
  Edge* edge = nullptr;
  for (int i = 0; i < this->numVer; i++) {
    edge = (*adjVector)[i].next;
    adj = new Vector<int>();
    if (edge) {
      while (edge) {
        adj->push_back(i);
        adj->push_back(edge->adjvex);
        adj->push_back(edge->weight);
        saveAdj.push_back(*adj);
        adj->clear();
        edge = edge->next;
      }
      std::cout << std::endl;
    }
  }
  return saveAdj;
}
void Graph::printAdjVector()
{
  Vector<Vector<int>> temp = saveAdjVector();
  Vector<Vector<int>>::iterator iter;
  Vector<int>::iterator itera;
  for (iter = temp.begin(); iter != temp.end(); iter++) {
    for (itera = iter->begin(); itera != iter->end(); itera++) {
      std::cout << *itera << " ";
    }
    std::cout << std::endl;
  }
}
void Graph::deleteEdge(int tail, int head)
{
  Edge* p = (*adjVector)[tail].next;

  Edge* q = nullptr;

  while (p != nullptr) {
    if (p->adjvex == head) {
      break;
    }

    q = p;
    p = p->next;
  }

  if (p == nullptr) {
    std::cout << "edge[" << (*adjVector)[tail]._id << "->" << (*adjVector)[head]._id << "] is not exist" << std::endl;
    return;
  }

  if (q == nullptr) {
    (*adjVector)[tail].next = p->next;
  }

  else {
    q->next = p->next;
  }

  delete p;
  (*adjVector)[tail]._outdegree--;
  (*adjVector)[head]._indegree--;
}
void Graph::BFS(int startVertex)
{
  for (int i = 0; i < numVer; i++)
    visited[i] = false;

  List<int> queue;

  visited[startVertex] = true;
  queue.push_back(startVertex);

  Edge* e = nullptr;
  while (!queue.empty()) {
    int currVertex = queue.front();
    std::cout << "Visited " << currVertex << " ";
    queue.pop_front();
    e = (*adjVector)[currVertex].next;
    for (int i = 0; i < (*adjVector)[currVertex]._outdegree; ++i) {
      int adjVertex = e->adjvex;
      if (!visited[adjVertex]) {
        visited[adjVertex] = true;
        queue.push_back(adjVertex);
      }
      e = e->next;
    }
  }
}
void Graph::DFS(int id)
{
  visited[id] = true;

  std::cout << id << " ";
  Vertex ve = (*adjVector)[id];
  Edge* ed = ve.next;
  int vertex1 = 0;

  for (int i = 0; i < ve._outdegree; ++i) {
    vertex1 = ed->adjvex;
    ed = ed->next;
    if (!visited[vertex1])
      DFS(vertex1);
  }
}
void Graph::topologicalSort()
{
  for (int i = 0; i < numVer; ++i)
    if ((*adjVector)[i]._indegree == 0)
      que.push(i);

  while (!que.empty()) {
    int ver = que.front();
    que.pop();
    std::cout << ver << " ";
    Edge* ed = nullptr;
    ed = (*adjVector)[ver].next;
    int adjv = 0;
    for (int i = 0; i < (*adjVector)[ver]._outdegree; ++i) {
      adjv = ed->adjvex;
      if (!(--((*adjVector)[adjv]._indegree)))
        que.push(adjv);
      ed = ed->next;
    }
  }
}
// bool Graph::checkLoop(int id) {
//   int vis[numVer];
//   for (int i = 0; i < numVer; ++i) {
//     if (this->ckLoopDFS(vis, i)) return false;
//   }
//   return true;
// }
// bool Graph::ckLoopDFS(int vis[], int start) {
//   if (vis[start] == -1) return true;
//   if (vis[start] == 1) return false;

//   vis[start] = 1;

//   Edge *ed = (*adjVector)[start].next;
//   int adjv = 0;
//   for (int i = 0; i < (*adjVector)[start].outdegree; ++i) {
//     adjv = ed->adjvex;
//     ed = ed->next;
//     if (!ckLoopDFS(vis, adjv)) {
//       return false;
//     }
//   }
//   vis[start] = -1;
//   return true;
// }
bool Graph::checkLoop()
{
  for (int i = 0; i < numVer; ++i)
    if ((*adjVector)[i]._indegree == 0)
      que.push(i);

  int count = 0;
  while (!que.empty()) {
    int ver = que.front();
    que.pop();
    ++count;
    Edge* ed = nullptr;
    ed = (*adjVector)[ver].next;
    int adjv = 0;
    for (int i = 0; i < (*adjVector)[ver]._outdegree; ++i) {
      adjv = ed->adjvex;
      if (!(--((*adjVector)[adjv]._indegree)))
        que.push(adjv);
      ed = ed->next;
    }
  }
  if (count < numVer)
    return false;
  else
    return true;
}

void Graph::DijkstraMinLength(int v0)
{
  Vector<int> path(numVer, 0);
  Vector<int> visited(numVer, 0);
  Vector<Node> dist(numVer);
  std::priority_queue<Node> q;
  // 初始化

  for (int i = 0; i < numVer; i++) {
    dist[i].id = i;
    dist[i].w = INF;
    path[i] = -1;
    visited[i] = 0;
  }
  dist[v0].w = 0;
  q.push(dist[v0]);
  while (!q.empty()) {
    Node cd = q.top();
    q.pop();
    int u = cd.id;

    if (visited[u])
      continue;
    visited[u] = 1;
    Edge* p = (*adjVector)[u].next;

    while (p) {
      int tempv = p->adjvex;
      int tempw = p->weight;

      if (!visited[tempv] && dist[tempv].w > dist[u].w + tempw) {
        dist[tempv].w = dist[u].w + tempw;
        path[tempv] = u;
        q.push(dist[tempv]);
      }
      p = p->next;
    }
  }

  for (int i = 0; i < numVer; ++i) {
    std::cout << "0 to " << i << " min length is " << dist[i].w << std::endl;
  }
}

void Graph::DijkstraMaxLength(int v0)
{
  Vector<int> path(numVer, 0);
  Vector<int> visited(numVer, 0);

  Node1 dist[numVer];
  std::priority_queue<Node1> q;

  for (int i = 0; i < numVer; i++) {
    dist[i].id = i;
    dist[i].w = 0;
    path[i] = -1;
    visited[i] = 0;
  }
  dist[v0].w = 0;
  q.push(dist[v0]);
  while (!q.empty()) {
    Node1 cd = q.top();
    q.pop();
    int u = cd.id;

    // if (visited[u]) continue;
    visited[u] = 1;
    Edge* p = (*adjVector)[u].next;

    while (p) {
      int tempv = p->adjvex;
      int tempw = p->weight;

      if (dist[tempv].w < dist[u].w + tempw) {
        dist[tempv].w = dist[u].w + tempw;
        path[tempv] = u;
        q.push(dist[tempv]);
      }
      p = p->next;
    }
  }

  for (int i = 0; i < numVer; ++i) {
    std::cout << "0 to " << i << " max length is " << dist[i].w << std::endl;
  }
}
}  // namespace ieda
