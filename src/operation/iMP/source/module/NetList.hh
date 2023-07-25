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
#include <string>
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
  NetList(size_t num_vertexs, size_t num_nets) : _num_vertexs(num_vertexs), _num_nets(num_nets) {}

  // Getter functions for vectors
  const std::vector<int64_t>& get_lx() const { return _lx; }

  const std::vector<int64_t>& get_ly() const { return _ly; }

  const std::vector<int64_t>& get_x_size() const { return _dx; }

  const std::vector<int64_t>& get_y_size() const { return _dy; }

  const std::vector<int64_t>& get_pin_x_off() const { return _pin_x_off; }

  const std::vector<int64_t>& get_pin_y_off() const { return _pin_y_off; }

  const std::vector<size_t>& get_net_span() const { return _net_span; }

  const std::vector<size_t>& get_pin2vertex() const { return _pin2vertex; }

  const std::vector<size_t>& get_vertex_span() const { return _vertex_span; }

  const std::vector<size_t>& get_pin2net() const { return _pin2net; }

  const std::vector<size_t>& get_row2col() const { return _row2col; }

  void set_region(int64_t lx, int64_t ly, int64_t dx, int64_t dy);

  void set_vertex_property(std::vector<VertexType>&& type, std::vector<int64_t>&& lx, std::vector<int64_t>&& ly, std::vector<int64_t>&& dx,
                           std::vector<int64_t>&& dy, std::vector<int64_t>&& area, std::vector<size_t>&& id_map = {});

  void set_connectivity(std::vector<size_t>&& net_span, std::vector<size_t>&& pin2vertex, std::vector<int64_t>&& pin_x_off,
                        std::vector<int64_t>&& pin_y_off);
  void sort_to_fit();

  std::vector<std::string> report();

  std::vector<size_t> cellsPartition(size_t npart);

  NetList make_clusters(const std::vector<size_t>& parts);

  void autoCellsClustering();

  void cell_Clustering(size_t npart) { clustering(cellsPartition(npart)); }

  void clustering(const std::vector<size_t>& parts);

 private:
  void updateVertexSpan();

 private:
  size_t _num_vertexs;
  size_t _num_nets;

  size_t _num_moveable;
  size_t _num_cells;
  size_t _num_clusters;
  size_t _num_macros;
  size_t _num_fixinst;
  size_t _num_term;

  bool _is_fit = false;

  int64_t _region_lx;
  int64_t _region_ly;
  int64_t _region_dx;
  int64_t _region_dy;
  double _region_aspect_ratio;
  double _utilization;

  std::vector<size_t> _id_map;

  std::vector<VertexType> _type;
  std::vector<int64_t> _lx;
  std::vector<int64_t> _ly;
  std::vector<int64_t> _dx;
  std::vector<int64_t> _dy;
  std::vector<int64_t> _area;
  int64_t _sum_vertex_area;
  int64_t _sum_cells_area;
  int64_t _sum_macro_area;
  int64_t _sum_cluster_area;
  int64_t _sum_fix_area;

  std::vector<size_t> _net_span;    // A net list, indicating the span of each net in the column-wise pin list.
  std::vector<size_t> _pin2vertex;  // A column-wise pin list, mapping the vertex corresponding to each pin.

  std::vector<size_t> _vertex_span;  // A vertex list, indicating the span of each vertex in the row-wise pin list.
  std::vector<size_t> _pin2net;      // A row-wise pin list, mapping the net corresponding to each pin.

  std::vector<size_t> _row2col;  // A row-wise pin list, mapping each pin in the row-wise pin list to an index in the column-wise pin list.
  std::vector<int64_t> _pin_x_off;  // A column-wise pin list, indicating the offset of each pin in the x-direction.
  std::vector<int64_t> _pin_y_off;  // A column-wise pin list, indicating the offset of each pin in the y-direction.
};

}  // namespace imp

#endif