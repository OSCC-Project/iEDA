#ifndef IMP_HYPERGRAPH_H
#define IMP_HYPERGRAPH_H
#include <memory>
#include <vector>
namespace imp {
class HyperGraph
{
 public:
  HyperGraph(int nvtxs, int nheges, std::vector<int>&& eptr, std::vector<int>&& eind, std::vector<int>&& vwgts = std::vector<int>{},
             std::vector<int>&& hewgts = std::vector<int>{});
  ~HyperGraph();
  int num_vertexs() { return _nvtxs; }
  int num_hedges() { return _nhedges; }

  int vertex_weight(int index) { return _vwgts[index]; }
  int hedge_weight(int index) { return _hewgts[index]; }
  int edge_index(int index) { return _eptr[index]; }
  const std::vector<int>& edges_ptr() const { return _eptr; }
  const std::vector<int>& edges_indices() const { return _eind; }
  const std::vector<int>& vertex_weights() const { return _vwgts; }
  const std::vector<int>& hedge_weights() const { return _hewgts; }

  // T* get() { return _Tp.get(); }
  // void set(T* Tp) { _Tp = std::make_shared<T>(Tp); }

  // T vertex(int index) { return _v[index]; }
  // V hedge2vertex(int index) { return _he2v[index]; }

 private:
  // The number of vertices and the number of hyperedges in the hypergraph, respectively.
  int _nvtxs, _nhedges;

  /**
   * @brief Two arrays that are used to describe the hyperedges in the graph. The first array, eptr, is of size nhedges+1, and it is used to
   * index the second array eind that stores the actual hyperedges. Each hyperedge is stored as a sequence of the vertices that it spans, in
   * consecutive locations in eind. Specifically, the i th hyperedge is stored starting at location eind[eptr[i]] up to (but not including)
   * eind[eptr[i + 1]]. Figure 6 illustrates this format for a simple hypergraph. The size of the array eind depends on the number and type
   * of hyperedges. Also note that the numbering of vertices starts from 0.
   *
   */
  std::vector<int> _eptr, _eind;

  std::vector<int> _vwgts;   // An array of size nvtxs that stores the weight of the vertices.
  std::vector<int> _hewgts;  // An array of size nhedges that stores the weight of the hyperedges.
  // std::vector<int> _vptr, _vind;

  // std::unique_ptr<T[]> _v;     // An array of size nvtxs that stores the property of vertex.
  // std::unique_ptr<V[]> _he2v;  // An array of size eind that stores the property of hedge.
};
HyperGraph::HyperGraph(int nvtxs, int nheges, std::vector<int>&& eptr, std::vector<int>&& eind, std::vector<int>&& vwgts,
                       std::vector<int>&& hewgts)
    : _nvtxs(nvtxs), _nhedges(nheges), _eptr(eptr), _eind(eind), _vwgts(vwgts), _hewgts(hewgts)
{
}
HyperGraph::~HyperGraph()
{
}

}  // namespace imp

#endif