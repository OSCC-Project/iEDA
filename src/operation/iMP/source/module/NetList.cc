#include "NetList.hh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

#include "Partitionner.hh"

namespace rv = std::views;

namespace imp {
NetList NetList::make_clusters(const std::vector<size_t>& parts)
{
  assert(parts.size() == _num_vertexs);

  NetList clusters;
  clusters.set_region(_region_lx, _region_ly, _region_dx, _region_dy);

  size_t clusters_num_vertex = *std::max_element(parts.begin(), parts.end()) + 1;
  // Determate area of clusters
  std::vector<int64_t> cluster_area(clusters_num_vertex, 0);
  std::vector<VertexType> cluster_type(clusters_num_vertex);
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

  std::vector<size_t> cluster_id_map(_id_map.size());
  for (size_t i = 0; i < _id_map.size(); i++) {
    cluster_id_map[i] = parts[_id_map[i]];
  }

  // Determate shape of clusters
  std::vector<int64_t> cluster_lx(clusters_num_vertex, 0);
  std::vector<int64_t> cluster_ly(clusters_num_vertex, 0);
  std::vector<int64_t> cluster_dx(clusters_num_vertex);
  std::vector<int64_t> cluster_dy(clusters_num_vertex);
  for (size_t i = 0; i < clusters_num_vertex; i++) {
    if (cluster_type[i] == kCluster) {
      // Make each cluster aspect ratio be canvas aspect ratio.
      cluster_area[i] /= _utilization;
      cluster_dy[i] = std::round(std::sqrt(cluster_area[i] * _region_aspect_ratio));
      cluster_dx[i] = std::round(cluster_area[i] / cluster_dy[i]);
    } else {
      // Keep original aspect ratio.
      cluster_dx[i] = _dx[single_cluster[i]];
      cluster_dy[i] = _dy[single_cluster[i]];
      if (cluster_type[i] == kTerminal || cluster_type[i] == kFixInst) {
        cluster_lx[i] = _lx[single_cluster[i]];
        cluster_lx[i] = _ly[single_cluster[i]];
      }
    }
  }

  // Extract subgraph of hypergraph
  std::vector<size_t> cluster_net_span;
  std::vector<size_t> cluster_pin2vertex;
  std::vector<int64_t> cluster_pin_x_off;
  std::vector<int64_t> cluster_pin_y_off;
  cluster_net_span.push_back(0);
  for (size_t i = 0; i < _num_nets; i++) {
    std::unordered_map<size_t, std::pair<int64_t, int64_t>> pins;
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
  clusters.set_vertex_property(std::move(cluster_type), std::move(cluster_lx), std::move(cluster_ly), std::move(cluster_dx),
                               std::move(cluster_dy), std::move(cluster_area), std::move(cluster_id_map));

  clusters.set_connectivity(std::move(cluster_net_span), std::move(cluster_pin2vertex), std::move(cluster_pin_x_off),
                            std::move(cluster_pin_y_off));
  clusters.sort_to_fit();

  return clusters;
}

void NetList::autoCellsClustering()
{
  if (!_is_fit)
    sort_to_fit();
  std::vector<int64_t> temp(_area.begin() + _num_cells, _area.begin() + _num_cells + _num_fixinst);
  std::sort(temp.begin(), temp.end());
  int64_t desird_area = temp[static_cast<size_t>(temp.size() / 2)];
  size_t npart = std::max(static_cast<size_t>(_sum_cells_area / desird_area), size_t{4});
  cell_Clustering(npart);
}

void NetList::clustering(const std::vector<size_t>& parts)
{
  *this = make_clusters(parts);
}

void NetList::set_region(int64_t lx, int64_t ly, int64_t dx, int64_t dy)
{
  _region_lx = lx;
  _region_ly = ly;
  _region_dx = dx;
  _region_dy = dy;
  _region_aspect_ratio = (double) dy / (double) dx;
}

void NetList::set_vertex_property(std::vector<VertexType>&& type, std::vector<int64_t>&& lx, std::vector<int64_t>&& ly,
                                  std::vector<int64_t>&& dx, std::vector<int64_t>&& dy, std::vector<int64_t>&& area,
                                  std::vector<size_t>&& id_map)
{
  assert(type.size() == lx.size());
  assert(lx.size() == ly.size());
  assert(ly.size() == dx.size());
  assert(dx.size() == dy.size());
  assert(dy.size() == area.size());

  _lx = std::move(lx);
  _ly = std::move(ly);
  _dx = std::move(dx);
  _dy = std::move(dy);
  _type = std::move(type);
  _area = std::move(area);
  _sum_vertex_area = 0;
  _sum_cells_area = 0;
  _sum_cluster_area = 0;
  _sum_macro_area = 0;
  _sum_fix_area = 0;

  _num_vertexs = _type.size();

  for (auto i : rv::iota((size_t) 0, _num_vertexs)) {
    auto i_type = _type[i];
    int64_t i_area = _area[i];
    if (i_type == kStdCell)
      _sum_cells_area += i_area;
    else if (i_type == kCluster)
      _sum_cluster_area += i_area;
    else if (i_type == kTerminal)
      continue;
    else if (i_type == kMacro)
      _sum_macro_area += i_area;
    else
      _sum_fix_area += i_area;
  }
  _sum_vertex_area = _sum_cells_area + _sum_cluster_area + _sum_macro_area + _sum_fix_area;

  if (id_map.empty()) {
    _id_map.resize(_num_vertexs);
    std::iota(_id_map.begin(), _id_map.end(), 0);
  } else {
    _id_map = std::move(id_map);
  }
}

void NetList::set_connectivity(std::vector<size_t>&& net_span, std::vector<size_t>&& pin2vertex, std::vector<int64_t>&& pin_x_off,
                               std::vector<int64_t>&& pin_y_off)
{
  assert(net_span.back() == pin2vertex.size());
  assert(pin2vertex.size() == pin_x_off.size());
  assert(pin_x_off.size() == pin_y_off.size());
  assert(_num_vertexs - 1 == *std::max_element(pin2vertex.begin(), pin2vertex.end()));
  _net_span = std::move(net_span);
  _pin2vertex = std::move(pin2vertex);
  _pin_x_off = std::move(pin_x_off);
  _pin_y_off = std::move(pin_y_off);
  _num_nets = _net_span.size() - 1;
}

void NetList::sort_to_fit()
{
  std::vector<size_t> map(_num_vertexs);
  _num_cells = 0;
  _num_clusters = 0;
  _num_macros = 0;
  _num_fixinst = 0;
  _num_term = 0;
  for (auto type : _type) {
    if (type == kStdCell)
      _num_cells++;
    else if (type == kCluster)
      _num_clusters++;
    else if (type == kTerminal)
      _num_term++;
    else if (type == kMacro)
      _num_macros++;
    else
      _num_fixinst++;
  }

  _num_moveable = _num_cells + _num_clusters + _num_macros;
  size_t i_cell = 0;
  size_t i_cluster = _num_cells;
  size_t i_macro = i_cluster + _num_clusters;
  size_t i_fix = i_macro + _num_macros;
  size_t i_term = i_fix + _num_fixinst;
  for (size_t i : rv::iota((size_t) 0, _num_vertexs)) {
    auto type = _type[i];
    if (type == kStdCell)
      map[i] = i_cell++;
    else if (type == kCluster)
      map[i] = i_cluster++;
    else if (type == kMacro)
      map[i] = i_macro++;
    else if (type == kFixInst)
      map[i] = i_fix++;
    else
      map[i] = i_term++;
  }
  assert(i_cell == _num_cells);
  assert(i_cluster - i_cell == _num_clusters);
  assert(i_macro - i_cluster == _num_macros);
  assert(i_fix - i_macro == _num_fixinst);
  assert(i_term - i_fix == _num_term);
  std::vector<VertexType> type(_num_vertexs);
  std::vector<int64_t> lx(_num_vertexs);
  std::vector<int64_t> ly(_num_vertexs);
  std::vector<int64_t> dx(_num_vertexs);
  std::vector<int64_t> dy(_num_vertexs);
  std::vector<int64_t> area(_num_vertexs);
  for (size_t i : rv::iota((size_t) 0, _num_vertexs)) {
    lx[map[i]] = _lx[i];
    ly[map[i]] = _ly[i];
    dx[map[i]] = _dx[i];
    dy[map[i]] = _dy[i];
    area[map[i]] = _area[i];
    type[map[i]] = _type[i];
  }
  _lx = std::move(lx);
  _ly = std::move(ly);
  _dx = std::move(dx);
  _dy = std::move(dy);
  _type = std::move(type);
  _area = std::move(area);

  for (size_t i = 0; i < _num_vertexs; i++) {
    _id_map[i] = map[_id_map[i]];
  }

  for (size_t i = 0; i < _pin2vertex.size(); i++) {
    _pin2vertex[i] = map[_pin2vertex[i]];
  }
  updateVertexSpan();
  _is_fit = true;
}

std::vector<std::string> NetList::report()
{
  using std::to_string;
  std::vector<std::string> reports;
  reports.emplace_back("Number of moveable cells: " + to_string(_num_cells));
  reports.emplace_back("Number of moveable clusters: " + to_string(_num_clusters));
  reports.emplace_back("Number of moveable macros: " + to_string(_num_macros));
  reports.emplace_back("Number of fixed objects: " + to_string(_num_fixinst));
  reports.emplace_back("Number of terminal: " + to_string(_num_term));
  reports.emplace_back("Number of nets: " + to_string(_num_nets));
  reports.emplace_back("Number of pins: " + to_string(_pin2vertex.size()));
  return reports;
}

std::vector<size_t> NetList::cellsPartition(size_t npart)
{
  std::vector<size_t> eptr;
  std::vector<size_t> eind;
  std::vector<int64_t> vwgt(_area.begin(), _area.begin() + _num_cells);
  eptr.push_back(0);
  auto pin2v = [&](size_t i) { return _pin2vertex[i]; };
  std::vector<size_t> he;
  for (size_t i : rv::iota((size_t) 0, _num_nets)) {
    he.clear();
    for (size_t j : rv::iota(_net_span[i], _net_span[i + 1]) | rv::transform(pin2v)) {
      if (j < _num_cells)
        he.push_back(j);
    }
    if (he.size() < 2)
      continue;
    eind.insert(eind.end(), he.begin(), he.end());
    eptr.push_back(eind.size());
  }
  std::vector<size_t> parts = std::move(Partitionner::hmetisSolve(_num_cells, eptr.size() - 1, eptr, eind, npart, 5, vwgt));
  size_t index = *std::max_element(parts.begin(), parts.end());
  parts.resize(_num_vertexs, 0);
  std::iota(parts.begin() + _num_cells, parts.end(), index + 1);
  return parts;
}

void NetList::updateVertexSpan()
{
  std::vector<std::vector<std::pair<size_t, size_t>>> temp_hmatrix(_num_vertexs, std::vector<std::pair<size_t, size_t>>());
  for (size_t i = 0; i < _num_nets; i++) {
    for (size_t j = _net_span[i]; j < _net_span[i + 1]; j++) {
      temp_hmatrix[_pin2vertex[j]].emplace_back(i, j);
    }
  }
  _vertex_span.resize(_num_vertexs + 1);
  _pin2net.resize(_pin2vertex.size());
  _row2col.resize(_pin2vertex.size());
  _vertex_span[0] = 0;
  for (size_t i = 0, k = 0; i < temp_hmatrix.size(); i++) {
    for (auto&& [net, col] : temp_hmatrix[i]) {
      _pin2net[k] = net;
      _row2col[k] = col;
    }
    _vertex_span[i + 1] = ++k;
  }
}

}  // namespace imp
