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
NetList NetList::makeClusters(const std::vector<size_t>& parts)
{
  assert(parts.size() == num_vertexs);

  NetList clusters;
  clusters.set_region(region_lx, region_ly, region_dx, region_dy);

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
      cluster_type[parts[i]] = type[i];
    }
    cluster_area[parts[i]] += area[i];
  }

  std::vector<size_t> cluster_id_map(id_map.size());
  for (size_t i = 0; i < id_map.size(); i++) {
    cluster_id_map[i] = parts[id_map[i]];
  }

  // Determate shape of clusters
  std::vector<int64_t> cluster_lx(clusters_num_vertex, 0);
  std::vector<int64_t> cluster_ly(clusters_num_vertex, 0);
  std::vector<int64_t> cluster_dx(clusters_num_vertex);
  std::vector<int64_t> cluster_dy(clusters_num_vertex);
  double cluster_util = (utilization > 0.85) ? 1 : utilization / 0.85;
  for (size_t i = 0; i < clusters_num_vertex; i++) {
    if (cluster_type[i] == kCluster) {
      // Make each cluster aspect ratio be canvas aspect ratio.
      cluster_area[i] /= cluster_util;
      cluster_dy[i] = std::round(std::sqrt(cluster_area[i] * region_aspect_ratio));
      cluster_dx[i] = std::round(cluster_area[i] / cluster_dy[i]);
    } else {
      // Keep original aspect ratio.
      cluster_dx[i] = dx[single_cluster[i]];
      cluster_dy[i] = dy[single_cluster[i]];
      if (cluster_type[i] == kTerminal || cluster_type[i] == kFixInst) {
        cluster_lx[i] = lx[single_cluster[i]];
        cluster_ly[i] = ly[single_cluster[i]];
      }
    }
  }

  // Extract subgraph of hypergraph
  std::vector<size_t> cluster_net_span;
  std::vector<size_t> cluster_pin2vertex;
  std::vector<int64_t> cluster_pin_x_off;
  std::vector<int64_t> cluster_pin_y_off;
  cluster_net_span.push_back(0);
  for (size_t i = 0; i < num_nets; i++) {
    std::unordered_map<size_t, std::pair<int64_t, int64_t>> pins;
    for (size_t j = net_span[i]; j < net_span[i + 1]; j++) {
      pins[parts[pin2vertex[j]]] = {pin_x_off[j], pin_y_off[j]};
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
  clusters.sortToFit();

  return clusters;
}

void NetList::autoCellsClustering()
{
  if (!is_fit)
    sortToFit();
  std::vector<int64_t> temp(area.begin() + num_cells, area.begin() + num_cells + num_fixinst);
  std::sort(temp.begin(), temp.end());
  int64_t desird_area = temp.empty() ? sum_cells_area : temp[static_cast<size_t>(temp.size() / 2)];
  size_t npart = std::max(static_cast<size_t>(sum_cells_area / desird_area), size_t{4});
  cellClustering(npart);
}

void NetList::clustering(const std::vector<size_t>& parts)
{
  *this = makeClusters(parts);
}

int64_t NetList::totalInstArea()
{
  return std::accumulate(std::begin(area), std::end(area) - num_term, int64_t(0));
}

void NetList::set_region(int64_t lx, int64_t ly, int64_t dx, int64_t dy)
{
  region_lx = lx;
  region_ly = ly;
  region_dx = dx;
  region_dy = dy;
  region_aspect_ratio = (double) dy / (double) dx;
  utilization = sum_vertex_area / region_dx * region_dy;
}

void NetList::set_vertex_property(std::vector<VertexType>&& type_new, std::vector<int64_t>&& lx_new, std::vector<int64_t>&& ly_new,
                                  std::vector<int64_t>&& dx_new, std::vector<int64_t>&& dy_new, std::vector<int64_t>&& area_new,
                                  std::vector<size_t>&& id_map_new)
{
  assert(type_new.size() == lx_new.size());
  assert(lx_new.size() == ly_new.size());
  assert(ly_new.size() == dx_new.size());
  assert(dx_new.size() == dy_new.size());
  assert(dy_new.size() == area_new.size());

  lx = std::move(lx_new);
  ly = std::move(ly_new);
  dx = std::move(dx_new);
  dy = std::move(dy_new);
  type = std::move(type_new);
  area = std::move(area_new);
  sum_vertex_area = 0;
  sum_cells_area = 0;
  sum_cluster_area = 0;
  sum_macro_area = 0;
  sum_fix_area = 0;

  num_vertexs = type.size();

  for (auto i : rv::iota((size_t) 0, num_vertexs)) {
    auto i_type = type[i];
    int64_t i_area = area[i];
    if (i_type == kStdCell)
      sum_cells_area += i_area;
    else if (i_type == kCluster)
      sum_cluster_area += i_area;
    else if (i_type == kTerminal)
      continue;
    else if (i_type == kMacro)
      sum_macro_area += i_area;
    else if (i_type == kFixInst)
      sum_fix_area += i_area;
  }
  sum_vertex_area = sum_cells_area + sum_cluster_area + sum_macro_area + sum_fix_area;

  if (region_dx != 0 && region_dy != 0)
    utilization = (double) sum_vertex_area / (double) region_dx / (double) region_dy;

  if (id_map_new.empty()) {
    id_map.resize(num_vertexs);
    std::iota(id_map.begin(), id_map.end(), 0);
  } else {
    id_map = std::move(id_map_new);
  }
}

void NetList::set_connectivity(std::vector<size_t>&& net_span_new, std::vector<size_t>&& pin2vertex_new,
                               std::vector<int64_t>&& pin_x_off_new, std::vector<int64_t>&& pin_y_off_new)
{
  assert(net_span_new.back() == pin2vertex_new.size());
  assert(pin2vertex_new.size() == pin_x_off_new.size());
  assert(pin_x_off_new.size() == pin_y_off_new.size());
  assert(num_vertexs - 1 == *std::max_element(pin2vertex_new.begin(), pin2vertex_new.end()));
  net_span = std::move(net_span_new);
  pin2vertex = std::move(pin2vertex_new);
  pin_x_off = std::move(pin_x_off_new);
  pin_y_off = std::move(pin_y_off_new);
  num_nets = net_span.size() - 1;
}

void NetList::sortToFit()
{
  std::vector<size_t> map(num_vertexs);
  num_cells = 0;
  num_clusters = 0;
  num_macros = 0;
  num_fixinst = 0;
  num_term = 0;
  for (auto type : type) {
    if (type == kStdCell)
      num_cells++;
    else if (type == kCluster)
      num_clusters++;
    else if (type == kTerminal)
      num_term++;
    else if (type == kMacro)
      num_macros++;
    else
      num_fixinst++;
  }

  num_moveable = num_cells + num_clusters + num_macros;
  size_t i_cell = 0;
  size_t i_cluster = num_cells;
  size_t i_macro = i_cluster + num_clusters;
  size_t i_fix = i_macro + num_macros;
  size_t i_term = i_fix + num_fixinst;
  for (size_t i : rv::iota((size_t) 0, num_vertexs)) {
    auto type = this->type[i];
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
  assert(i_cell == num_cells);
  assert(i_cluster - i_cell == num_clusters);
  assert(i_macro - i_cluster == num_macros);
  assert(i_fix - i_macro == num_fixinst);
  assert(i_term - i_fix == num_term);
  std::vector<VertexType> type(num_vertexs);
  std::vector<int64_t> lx(num_vertexs);
  std::vector<int64_t> ly(num_vertexs);
  std::vector<int64_t> dx(num_vertexs);
  std::vector<int64_t> dy(num_vertexs);
  std::vector<int64_t> area(num_vertexs);
  for (size_t i : rv::iota((size_t) 0, num_vertexs)) {
    lx[map[i]] = lx[i];
    ly[map[i]] = ly[i];
    dx[map[i]] = dx[i];
    dy[map[i]] = dy[i];
    area[map[i]] = area[i];
    type[map[i]] = type[i];
  }
  lx = std::move(lx);
  ly = std::move(ly);
  dx = std::move(dx);
  dy = std::move(dy);
  type = std::move(type);
  area = std::move(area);

  for (size_t i = 0; i < num_vertexs; i++) {
    id_map[i] = map[id_map[i]];
  }

  for (size_t i = 0; i < pin2vertex.size(); i++) {
    pin2vertex[i] = map[pin2vertex[i]];
  }
  updateVertexSpan();
  is_fit = true;
}

std::vector<std::string> NetList::report()
{
  using std::to_string;
  std::vector<std::string> reports;
  reports.emplace_back("Number of moveable cells: " + to_string(num_cells));
  reports.emplace_back("Number of moveable clusters: " + to_string(num_clusters));
  reports.emplace_back("Number of moveable macros: " + to_string(num_macros));
  reports.emplace_back("Number of fixed objects: " + to_string(num_fixinst));
  reports.emplace_back("Number of terminal: " + to_string(num_term));
  reports.emplace_back("Number of nets: " + to_string(num_nets));
  reports.emplace_back("Number of pins: " + to_string(pin2vertex.size()));
  reports.emplace_back("Total area of inst: " + to_string(totalInstArea()));
  reports.emplace_back("Core area: " + to_string(region_dx * region_dy));
  return reports;
}

std::vector<size_t> NetList::cellsPartition(size_t npart)
{
  std::vector<size_t> eptr;
  std::vector<size_t> eind;
  std::vector<int64_t> vwgt(area.begin(), area.begin() + num_cells);
  eptr.push_back(0);
  auto pin2v = [&](size_t i) { return pin2vertex[i]; };
  std::vector<size_t> he;
  for (size_t i : rv::iota((size_t) 0, num_nets)) {
    he.clear();
    for (size_t j : rv::iota(net_span[i], net_span[i + 1]) | rv::transform(pin2v)) {
      if (j < num_cells)
        he.push_back(j);
    }
    if (he.size() < 2)
      continue;
    eind.insert(eind.end(), he.begin(), he.end());
    eptr.push_back(eind.size());
  }
  std::vector<size_t> parts = std::move(Partitionner::hmetisSolve(num_cells, eptr.size() - 1, eptr, eind, npart, 5, vwgt));
  size_t index = *std::max_element(parts.begin(), parts.end());
  parts.resize(num_vertexs, 0);
  std::iota(parts.begin() + num_cells, parts.end(), index + 1);
  return parts;
}

void NetList::updateVertexSpan()
{
  std::vector<std::vector<std::pair<size_t, size_t>>> temp_hmatrix(num_vertexs, std::vector<std::pair<size_t, size_t>>());
  for (size_t i = 0; i < num_nets; i++) {
    for (size_t j = net_span[i]; j < net_span[i + 1]; j++) {
      temp_hmatrix[pin2vertex[j]].emplace_back(i, j);
    }
  }
  vertex_span.resize(num_vertexs + 1);
  pin2net.resize(pin2vertex.size());
  row2col.resize(pin2vertex.size());
  vertex_span[0] = 0;
  for (size_t i = 0, k = 0; i < temp_hmatrix.size(); i++) {
    for (auto&& [net, col] : temp_hmatrix[i]) {
      pin2net[k] = net;
      row2col[k] = col;
    }
    vertex_span[i + 1] = ++k;
  }
}

}  // namespace imp
