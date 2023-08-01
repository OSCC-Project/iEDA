#include "NetList.hh"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
namespace imp {
NetList::NetList(size_t numVertexs, size_t numMoveableVertexs, size_t numFixedVertexs, size_t numNets)
    : _num_vertexs(numVertexs), _num_moveable(numMoveableVertexs), _num_fixed(numFixedVertexs), _num_nets(numNets)
{
  // Initialize vectors with appropriate sizes
  _lx.resize(_num_vertexs);
  _ly.resize(_num_vertexs);
  _dx.resize(_num_vertexs);
  _dy.resize(_num_vertexs);
  _pin_x_off.resize(_num_vertexs);
  _pin_y_off.resize(_num_vertexs);
  _net_span.resize(_num_nets);
  _pin2vertex.resize(_num_vertexs);
  _vertex_span.resize(_num_vertexs);
  _pin2net.resize(_num_vertexs);
  _row2col.resize(_num_vertexs);
}

NetList::NetList(size_t numVertexs, size_t numMoveableVertexs, size_t numFixedVertexs, size_t numNets, const std::vector<VertexType>& vType,
                 const std::vector<int32_t>& vLx, const std::vector<int32_t>& vLy, const std::vector<int32_t>& vXSize,
                 const std::vector<int32_t>& vYSize, const std::vector<int32_t>& pinXOff, const std::vector<int32_t>& pinYOff,
                 const std::vector<size_t>& netSpan, const std::vector<size_t>& pin2Vertex)
    : _num_vertexs(numVertexs),
      _num_moveable(numMoveableVertexs),
      _num_fixed(numFixedVertexs),
      _num_nets(numNets),
      _type(vType),
      _lx(vLx),
      _ly(vLy),
      _dx(vXSize),
      _dy(vYSize),
      _pin_x_off(pinXOff),
      _pin_y_off(pinYOff),
      _net_span(netSpan),
      _pin2vertex(pin2Vertex)
{
  initVertexSpan();
}

NetList::NetList(size_t numVertexs, size_t numMoveableVertexs, size_t numFixedVertexs, size_t numNets, std::vector<VertexType>&& vType,
                 std::vector<int32_t>&& vLx, std::vector<int32_t>&& vLy, std::vector<int32_t>&& vXSize, std::vector<int32_t>&& vYSize,
                 std::vector<int32_t>&& pinXOff, std::vector<int32_t>&& pinYOff, std::vector<size_t>&& netSpan,
                 std::vector<size_t>&& pin2Vertex)
    : _num_vertexs(numVertexs),
      _num_moveable(numMoveableVertexs),
      _num_fixed(numFixedVertexs),
      _num_nets(numNets),
      _type(vType),
      _lx(vLx),
      _ly(vLy),
      _dx(vXSize),
      _dy(vYSize),
      _pin_x_off(pinXOff),
      _pin_y_off(pinYOff),
      _net_span(netSpan),
      _pin2vertex(pin2Vertex)
{
  initVertexSpan();
}

NetList NetList::make_clusters(const std::vector<size_t> parts)
{
  size_t num_clusters = *std::max_element(parts.begin(), parts.end()) + 1;
  // Determate area of clusters
  std::vector<int32_t> cluster_area(num_clusters, 0);
  std::vector<VertexType> cluster_type(num_clusters);
  std::unordered_map<size_t, size_t> single_cluster;
  for (size_t i = 0; i < parts.size(); i++) {
    if (single_cluster.contains(parts[i])) {
      cluster_type[parts[i]] = kCluster;
    } else {
      single_cluster[parts[i]] = i;
      cluster_type[parts[i]] = _type[i];
    }
    cluster_area[parts[i]] += _area[i];
  }
  // Determate shape of clusters
  std::vector<int32_t> cluster_lx(num_clusters, 0);
  std::vector<int32_t> cluster_ly(num_clusters, 0);
  std::vector<int32_t> cluster_dx(num_clusters);
  std::vector<int32_t> cluster_dy(num_clusters);
  double aspect_ratio = (double) _canvas_dy / (double) _canvas_dx;
  for (size_t i = 0; i < num_clusters; i++) {
    if (cluster_type[i] == kCluster) {
      // Make each cluster aspect ratio be canvas aspect ratio.
      cluster_area[i] /= _utilization;
      cluster_dy[i] = std::round(std::sqrt(cluster_area[i] * aspect_ratio));
      cluster_dx[i] = std::round(cluster_area[i] / cluster_dy[i]);
    } else {
      // Keep original aspect ratio.
      cluster_dx[i] = _dx[single_cluster[i]];
      cluster_dy[i] = _dy[single_cluster[i]];
      if (cluster_type[i] == kFix) {
        cluster_lx[i] = _lx[single_cluster[i]];
        cluster_lx[i] = _ly[single_cluster[i]];
      }
    }
  }

  // Extract subgraph of hypergraph
  std::vector<size_t> cluster_net_span;
  std::vector<size_t> cluster_pin2vertex;
  std::vector<int32_t> cluster_pin_x_off;
  std::vector<int32_t> cluster_pin_y_off;
  cluster_net_span.push_back(0);
  for (size_t i = 0; i < _num_nets; i++) {
    std::unordered_map<size_t, std::pair<int32_t, int32_t>> pins;
    for (size_t j = _net_span[i]; j < _net_span[i + 1]; j++) {
      pins[parts[_pin2vertex[j]]] = {_pin_x_off[j], _pin_y_off[j]};
    }
    if (pins.size() > 1) {
      for (auto [v, off] : pins) {
        cluster_pin2vertex.push_back(v);
        if (cluster_type[v] != kCluster) {
          cluster_pin_x_off.push_back(off.first);
          cluster_pin_y_off.push_back(off.second);
        } else {
          // As cluster, we make its pin in the center of cluster for now.
          cluster_pin_x_off.push_back(0);
          cluster_pin_y_off.push_back(0);
        }
      }
      cluster_net_span.push_back(cluster_pin2vertex.size());
    }
  }
  size_t num_clusters_net = cluster_net_span.size() - 1;
  return NetList(num_clusters, num_clusters - _num_fixed, _num_fixed, num_clusters_net, std::move(cluster_type), std::move(cluster_lx),
                 std::move(cluster_ly), std::move(cluster_dx), std::move(cluster_dy), std::move(cluster_pin_x_off),
                 std::move(cluster_pin_y_off), std::move(cluster_net_span), std::move(cluster_pin2vertex));
}

void NetList::clustering(const std::vector<size_t> parts)
{
  *this = make_clusters(parts);
}

void NetList::initVertexSpan()
{
  std::vector<std::vector<std::pair<size_t, size_t>>> temp_hmatrix(_num_vertexs, std::vector<std::pair<size_t, size_t>>());
  for (size_t i = 0; i < _net_span.size() - 1; i++) {
    for (size_t j = _net_span[i]; j < _net_span[i + 1]; j++) {
      temp_hmatrix[_pin2vertex[j]].emplace_back(i, j);
    }
  }
  _vertex_span.resize(_num_vertexs + 1);
  _pin2net.resize(_pin2vertex.size());
  _row2col.resize(_pin2vertex.size());
  _vertex_span[0] = 0;
  for (size_t i = 0, k = 0; i < temp_hmatrix.size(); i++) {
    for (size_t j = 0; j < temp_hmatrix[i].size(); j++) {
      _pin2net[k] = temp_hmatrix[i][j].first;
      _row2col[k++] = temp_hmatrix[i][j].second;
    }
    _vertex_span[i + 1] = k;
  }
}

}  // namespace imp
