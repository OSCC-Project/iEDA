/**
 * @file NetList.hh
 * @author Fuxing Huang (fxxhuang@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-07-13
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef IMP_NETLIST_H
#define IMP_NETLIST_H
#include <cstdint>
#include <functional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>
namespace imp {
/**
 * @brief A netlist is a hypergraph, and a hypergraph can be represented as a matrix and is generally a sparse matrix. A sparse matrix can
 * be stored in two ways, row-wise and column-wise. So our pinlist has two storage orders, row-wise order and col-wise order.
 *
 */
class NetList
{
 public:
  enum VertexType
  {
    kStdCell,
    kCluster,
    kMacro,
    kFixInst,
    kTerminal
  };

 public:
  // Constructor
  NetList() = default;
  NetList(size_t num_vertexs, size_t num_nets) : num_vertexs(num_vertexs), num_nets(num_nets) {}

  void set_region(int64_t lx, int64_t ly, int64_t dx, int64_t dy);

  void set_vertex_property(std::vector<VertexType>&& type, std::vector<int64_t>&& lx, std::vector<int64_t>&& ly, std::vector<int64_t>&& dx,
                           std::vector<int64_t>&& dy, std::vector<int64_t>&& area, std::vector<size_t>&& id_map = {});

  void set_connectivity(std::vector<size_t>&& net_span, std::vector<size_t>&& pin2vertex, std::vector<int64_t>&& pin_x_off,
                        std::vector<int64_t>&& pin_y_off);
  void sortToFit();

  std::vector<std::string> report();

  std::vector<size_t> cellsPartition(size_t npart);

  NetList makeClusters(const std::vector<size_t>& parts);

  void autoCellsClustering();

  void cellClustering(size_t npart) { clustering(cellsPartition(npart)); }

  void clustering(const std::vector<size_t>& parts);

  int64_t totalInstArea();

  void unFixMacro()
  {
    for (size_t i = num_moveable; i < num_moveable + num_fixinst; i++) {
      type[i] = kMacro;
    }
    sortToFit();
  }

 private:
  void updateVertexSpan();

 public:
  size_t num_vertexs;  // order by num_moveable(_num_cells, num_clusters, num_macros), _numfixinst, _numterm
  size_t num_nets;

  size_t num_moveable;  // num_cells, num_clusters, num_macros
  size_t num_cells;
  size_t num_clusters;
  size_t num_macros;
  size_t num_fixinst;
  size_t num_term;

  bool is_fit = false;

  int64_t region_lx;
  int64_t region_ly;
  int64_t region_dx;
  int64_t region_dy;
  double region_aspect_ratio;
  double utilization;

  std::vector<size_t> id_map;

  std::vector<VertexType> type;
  std::vector<int64_t> lx;
  std::vector<int64_t> ly;
  std::vector<int64_t> dx;
  std::vector<int64_t> dy;
  std::vector<int64_t> area;

  int64_t sum_vertex_area;
  int64_t sum_cells_area;
  int64_t sum_macro_area;
  int64_t sum_cluster_area;
  int64_t sum_fix_area;

  std::vector<size_t> net_span;    // A net list, indicating the span of each net in the column-wise pin list.
  std::vector<size_t> pin2vertex;  // A column-wise pin list, mapping the vertex corresponding to each pin.

  std::vector<size_t> vertex_span;  // A vertex list, indicating the span of each vertex in the row-wise pin list.
  std::vector<size_t> pin2net;      // A row-wise pin list, mapping the net corresponding to each pin.

  std::vector<size_t> row2col;  // A row-wise pin list, mapping each pin in the row-wise pin list to an index in the column-wise pin list.
  std::vector<int64_t> pin_x_off;  // A column-wise pin list, indicating the offset of each pin in the x-direction.
  std::vector<int64_t> pin_y_off;  // A column-wise pin list, indicating the offset of each pin in the y-direction.
};

template <typename Numeric, typename Type, typename Transform>
struct Block
{
  Numeric lx;
  Numeric ly;
  Numeric dx;
  Numeric dy;
  Type type;
  Transform tf;
};

template <typename Numeric>
struct Pin2
{
  Numeric x_off;
  Numeric y_off;
};

}  // namespace imp

#endif