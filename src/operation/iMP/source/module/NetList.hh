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
    kMacro,
    kStdCell,
    kCluster,
    kFix
  };

 public:
  // Constructor
  NetList(size_t, size_t, size_t, size_t);

  NetList(size_t, size_t, size_t, size_t, const std::vector<VertexType>&, const std::vector<int32_t>&, const std::vector<int32_t>&,
          const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&, const std::vector<int32_t>&,
          const std::vector<size_t>&, const std::vector<size_t>&);

  NetList(size_t, size_t, size_t, size_t, std::vector<VertexType>&&, std::vector<int32_t>&&, std::vector<int32_t>&&, std::vector<int32_t>&&,
          std::vector<int32_t>&&, std::vector<int32_t>&&, std::vector<int32_t>&&, std::vector<size_t>&&, std::vector<size_t>&&);

  // Getter functions for vectors

  const std::vector<int32_t>& get_lx() const { return _lx; }

  const std::vector<int32_t>& get_ly() const { return _ly; }

  const std::vector<int32_t>& get_x_size() const { return _dx; }

  const std::vector<int32_t>& get_y_size() const { return _dy; }

  const std::vector<int32_t>& get_pin_x_off() const { return _pin_x_off; }

  const std::vector<int32_t>& get_pin_y_off() const { return _pin_y_off; }

  const std::vector<size_t>& get_net_span() const { return _net_span; }

  const std::vector<size_t>& get_pin2vertex() const { return _pin2vertex; }

  const std::vector<size_t>& get_vertex_span() const { return _vertex_span; }

  const std::vector<size_t>& get_pin2net() const { return _pin2net; }

  const std::vector<size_t>& get_row2col() const { return _row2col; }

  NetList make_clusters(const std::vector<size_t> parts);

  void clustering(const std::vector<size_t> parts);

 private:
  void initVertexSpan();

 private:
  size_t _num_vertexs;
  size_t _num_moveable;
  size_t _num_fixed;
  size_t _num_nets;
  int32_t _canvas_lx;
  int32_t _canvas_ly;
  int32_t _canvas_dx;
  int32_t _canvas_dy;
  double _utilization;
  std::vector<VertexType> _type;
  std::vector<int32_t> _lx;
  std::vector<int32_t> _ly;
  std::vector<int32_t> _dx;
  std::vector<int32_t> _dy;
  std::vector<int32_t> _area;
  std::vector<int32_t> _pin_x_off;   // A column-wise pin list, indicating the offset of each pin in the x-direction.
  std::vector<int32_t> _pin_y_off;   // A column-wise pin list, indicating the offset of each pin in the y-direction.
  std::vector<size_t> _net_span;     // A net list, indicating the span of each net in the column-wise pin list.
  std::vector<size_t> _pin2vertex;   // A column-wise pin list, mapping the vertex corresponding to each pin.
  std::vector<size_t> _vertex_span;  // A vertex list, indicating the span of each vertex in the row-wise pin list.
  std::vector<size_t> _pin2net;      // A row-wise pin list, mapping the net corresponding to each pin.
  std::vector<size_t> _row2col;  // A row-wise pin list, mapping each pin in the row-wise pin list to an index in the column-wise pin list.
};

}  // namespace imp

#endif