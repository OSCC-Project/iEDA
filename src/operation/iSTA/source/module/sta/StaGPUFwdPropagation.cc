/**
 * @file StaGPUPropagation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The host api for gpu propagation.
 * @version 0.1
 * @date 2025-02-02
 * 
 */

#include "propagation-cuda/fwd_propagation.cuh"
#include "propagation-cuda/lib_arc.cuh"
#include "Sta.hh"
#include "StaGPUFwdPropagation.hh"


namespace ista {

/**
 * @brief build gpu vertex slew data.
 *
 * @param the_vertex
 * @param gpu_vertex
 * @param flatten_slew_data
 */
void build_gpu_vertex_slew_data(
    StaVertex* the_vertex, GPU_Vertex& gpu_vertex,
    std::vector<GPU_Fwd_Data<int64_t>>& flatten_slew_data) {
  // build slew data.
  the_vertex->initSlewData();
  gpu_vertex._slew_data._start_pos = flatten_slew_data.size();
  StaData* slew_data;
  FOREACH_SLEW_DATA(the_vertex, slew_data) {
    GPU_Fwd_Data<int64_t> gpu_slew_data;
    auto* the_slew_data = dynamic_cast<StaSlewData*>(slew_data);
    auto slew_value = the_slew_data->get_slew();

    gpu_slew_data._data_value = slew_value;
    gpu_slew_data._trans_type =
        the_slew_data->get_trans_type() == TransType::kRise
            ? GPU_Trans_Type::kRise
            : GPU_Trans_Type::kFall;
    gpu_slew_data._analysis_mode =
        the_slew_data->get_delay_type() == AnalysisMode::kMax
            ? GPU_Analysis_Mode::kMax
            : GPU_Analysis_Mode::kMin;
    flatten_slew_data.emplace_back(gpu_slew_data);
  }
  gpu_vertex._slew_data._num_fwd_data =
      flatten_slew_data.size() - gpu_vertex._slew_data._start_pos;
}

/**
 * @brief build gpu vertex arrive time data.
 *
 * @param the_vertex
 * @param gpu_vertex
 * @param flatten_at_data
 */
void build_gpu_vertex_at_data(
    StaVertex* the_vertex, GPU_Vertex& gpu_vertex,
    std::vector<GPU_Fwd_Data<int64_t>>& flatten_at_data) {
  // build slew data.
  the_vertex->initPathDelayData();
  gpu_vertex._at_data._start_pos = flatten_at_data.size();
  StaData* at_data;
  FOREACH_DELAY_DATA(the_vertex, at_data) {
    GPU_Fwd_Data<int64_t> gpu_at_data;
    auto* path_delay_data = dynamic_cast<StaPathDelayData*>(at_data);
    auto at_value = path_delay_data->get_arrive_time();

    gpu_at_data._data_value = at_value;
    gpu_at_data._trans_type =
        path_delay_data->get_trans_type() == TransType::kRise
            ? GPU_Trans_Type::kRise
            : GPU_Trans_Type::kFall;
    gpu_at_data._analysis_mode =
        path_delay_data->get_delay_type() == AnalysisMode::kMax
            ? GPU_Analysis_Mode::kMax
            : GPU_Analysis_Mode::kMin;
    flatten_at_data.emplace_back(gpu_at_data);
  }
  gpu_vertex._at_data._num_fwd_data =
      flatten_at_data.size() - gpu_vertex._at_data._start_pos;
}

/**
 * @brief build gpu node cap data.
 *
 * @param the_vertex
 * @param gpu_vertex
 * @param flatten_node_cap_data
 */
void build_gpu_vertex_node_cap_data(
    StaVertex* the_vertex, GPU_Vertex& gpu_vertex,
    std::vector<GPU_Fwd_Data<double>>& flatten_node_cap_data) {
  gpu_vertex._node_cap_data._start_pos = flatten_node_cap_data.size();
  FOREACH_MODE_TRANS(mode, trans) {
    GPU_Fwd_Data<double> gpu_node_cap_data;
    auto the_vertex_load = the_vertex->getLoad(mode, trans);
    gpu_node_cap_data._data_value = the_vertex_load;
    gpu_node_cap_data._trans_type = trans == TransType::kRise
                                        ? GPU_Trans_Type::kRise
                                        : GPU_Trans_Type::kFall;
    gpu_node_cap_data._analysis_mode = mode == AnalysisMode::kMax
                                           ? GPU_Analysis_Mode::kMax
                                           : GPU_Analysis_Mode::kMin;
    flatten_node_cap_data.emplace_back(gpu_node_cap_data);
  }
  gpu_vertex._node_cap_data._num_fwd_data =
      flatten_node_cap_data.size() - gpu_vertex._node_cap_data._start_pos;
}

/**
 * @brief build gpu node delay data.
 *
 * @param the_vertex
 * @param gpu_vertex
 * @param flatten_node_delay_data
 */
void build_gpu_vertex_node_delay_data(
    StaVertex* the_vertex, GPU_Vertex& gpu_vertex,
    std::vector<GPU_Fwd_Data<double>>& flatten_node_delay_data) {
  gpu_vertex._node_cap_data._start_pos = flatten_node_delay_data.size();
  FOREACH_MODE_TRANS(mode, trans) {
    GPU_Fwd_Data<double> gpu_node_delay_data;
    gpu_node_delay_data._data_value =
        NS_TO_PS(the_vertex->getNetLoadDelay(mode, trans));
    gpu_node_delay_data._trans_type = trans == TransType::kRise
                                          ? GPU_Trans_Type::kRise
                                          : GPU_Trans_Type::kFall;
    gpu_node_delay_data._analysis_mode = mode == AnalysisMode::kMax
                                             ? GPU_Analysis_Mode::kMax
                                             : GPU_Analysis_Mode::kMin;
    flatten_node_delay_data.emplace_back(gpu_node_delay_data);
  }
  gpu_vertex._node_cap_data._num_fwd_data =
      flatten_node_delay_data.size() - gpu_vertex._node_cap_data._start_pos;
}

/**
 * @brief build gpu vertex node impulse data for calc net load slew.
 *
 * @param the_vertex
 * @param gpu_vertex
 * @param flatten_node_impulse_data
 */
void build_gpu_vertex_node_impulse_data(
    StaVertex* the_vertex, GPU_Vertex& gpu_vertex,
    std::vector<GPU_Fwd_Data<double>>& flatten_node_impulse_data) {
  gpu_vertex._node_impulse_data._start_pos = flatten_node_impulse_data.size();
  FOREACH_MODE_TRANS(mode, trans) {
    GPU_Fwd_Data<double> gpu_node_impulse_data;
    gpu_node_impulse_data._data_value =
        the_vertex->getNetSlewImpulse(mode, trans);
    gpu_node_impulse_data._trans_type = trans == TransType::kRise
                                            ? GPU_Trans_Type::kRise
                                            : GPU_Trans_Type::kFall;
    gpu_node_impulse_data._analysis_mode = mode == AnalysisMode::kMax
                                               ? GPU_Analysis_Mode::kMax
                                               : GPU_Analysis_Mode::kMin;
    flatten_node_impulse_data.emplace_back(gpu_node_impulse_data);
  }
  gpu_vertex._node_impulse_data._num_fwd_data =
      flatten_node_impulse_data.size() - gpu_vertex._node_cap_data._start_pos;
}

/**
 * @brief build gpu arc delay data.
 *
 * @param gpu_arc
 * @param flatten_arc_delay_data
 */
void build_gpu_arc_delay_data(
    GPU_Arc& gpu_arc,
    std::vector<GPU_Fwd_Data<int64_t>>& flatten_arc_delay_data) {
  gpu_arc._delay_values._start_pos = flatten_arc_delay_data.size();
  FOREACH_MODE_TRANS(mode, trans) {
    GPU_Fwd_Data<int64_t> gpu_arc_delay_data;
    gpu_arc_delay_data._data_value = 0.0;
    gpu_arc_delay_data._trans_type = trans == TransType::kRise
                                         ? GPU_Trans_Type::kRise
                                         : GPU_Trans_Type::kFall;
    gpu_arc_delay_data._analysis_mode = mode == AnalysisMode::kMax
                                            ? GPU_Analysis_Mode::kMax
                                            : GPU_Analysis_Mode::kMin;
    flatten_arc_delay_data.emplace_back(gpu_arc_delay_data);
  }
  gpu_arc._delay_values._num_fwd_data = flatten_arc_delay_data.size();
}

/**
 * @brief build gpu graph for gpu speed computation.
 *
 * @param the_sta_graph
 * @param vertex_data_size the flatten vertex data size
 * @param arc_data_size the flatten arc data size
 * @param arc_to_index sta arc to gpu index.
 * @return GPU_Graph
 */
GPU_Graph build_gpu_graph(StaGraph* the_sta_graph, unsigned& vertex_data_size,
                          unsigned& arc_data_size,
                          std::map<StaArc*, unsigned>& arc_to_index) {
  GPU_Graph gpu_graph;
  unsigned num_vertex = the_sta_graph->numVertex();
  unsigned num_arc = the_sta_graph->numArc();

  std::vector<GPU_Vertex> gpu_vertices;
  gpu_vertices.reserve(num_vertex);
  std::vector<GPU_Arc> gpu_arcs;
  gpu_arcs.reserve(num_arc);

  // init cpu memory.
  vertex_data_size = c_gpu_num_vertex_data * num_vertex;
  std::vector<GPU_Fwd_Data<int64_t>> flatten_slew_data;
  flatten_slew_data.reserve(vertex_data_size);
  std::vector<GPU_Fwd_Data<int64_t>> flatten_at_data;
  flatten_at_data.reserve(vertex_data_size);
  std::vector<GPU_Fwd_Data<double>> flatten_node_cap_data;
  flatten_node_cap_data.reserve(vertex_data_size);
  std::vector<GPU_Fwd_Data<double>> flatten_node_delay_data;
  flatten_node_delay_data.reserve(vertex_data_size);
  std::vector<GPU_Fwd_Data<double>> flatten_node_impulse_data;
  flatten_node_impulse_data.reserve(vertex_data_size);

  arc_data_size = c_gpu_num_arc_delay * num_arc;
  std::vector<GPU_Fwd_Data<int64_t>> flatten_arc_delay_data;
  flatten_arc_delay_data.reserve(arc_data_size);

  // build gpu vertex
  StaVertex* the_vertex;
  std::map<StaVertex*, unsigned> vertex_to_id;
  FOREACH_VERTEX(the_sta_graph, the_vertex) {
    GPU_Vertex gpu_vertex;

    build_gpu_vertex_slew_data(the_vertex, gpu_vertex, flatten_slew_data);
    build_gpu_vertex_at_data(the_vertex, gpu_vertex, flatten_at_data);
    build_gpu_vertex_node_cap_data(the_vertex, gpu_vertex,
                                   flatten_node_cap_data);
    build_gpu_vertex_node_delay_data(the_vertex, gpu_vertex,
                                     flatten_node_delay_data);
    build_gpu_vertex_node_impulse_data(the_vertex, gpu_vertex,
                                       flatten_node_impulse_data);
    vertex_to_id[the_vertex] = gpu_vertices.size();
    gpu_vertices.emplace_back(std::move(gpu_vertex));
  }

  // build gpu arc
  StaArc* the_arc;
  FOREACH_ARC(the_sta_graph, the_arc) {
    GPU_Arc gpu_arc;

    gpu_arc._arc_type = the_arc->isDelayArc()   ? GPU_Arc_Type::kInstDelayArc
                        : the_arc->isCheckArc() ? GPU_Arc_Type::kInstCheckArc
                                                : GPU_Arc_Type::kNet;

    gpu_arc._arc_trans_type =
        the_arc->isInstArc()
            ? the_arc->isPositiveArc()   ? GPU_Arc_Trans_Type::kPositive
              : the_arc->isNegativeArc() ? GPU_Arc_Trans_Type::kNegative
                                         : GPU_Arc_Trans_Type::kNonUnate
            : GPU_Arc_Trans_Type::kPositive;

    gpu_arc._src_vertex_id = vertex_to_id[the_arc->get_src()];
    gpu_arc._snk_vertex_id = vertex_to_id[the_arc->get_snk()];
    if (the_arc->isInstArc()) {
      gpu_arc._lib_data_arc_id =
          dynamic_cast<StaInstArc*>(the_arc)->get_arc_id();
    } else {
      gpu_arc._lib_data_arc_id = -1;
    }

    build_gpu_arc_delay_data(gpu_arc, flatten_arc_delay_data);

    gpu_arcs.emplace_back(std::move(gpu_arc));

    arc_to_index[the_arc] = gpu_arcs.size() - 1;
  }

  // copy cpu data to gpu memory.
  gpu_graph._vertices = gpu_vertices.data();
  gpu_graph._arcs = gpu_arcs.data();

  gpu_graph._num_vertices = num_vertex;
  gpu_graph._num_arcs = num_arc;
  gpu_graph._flatten_slew_data = flatten_slew_data.data();
  gpu_graph._flatten_at_data = flatten_at_data.data();
  gpu_graph._flatten_node_cap_data = flatten_node_cap_data.data();
  gpu_graph._flatten_node_delay_data = flatten_node_delay_data.data();
  gpu_graph._flatten_node_impulse_data = flatten_node_impulse_data.data();
  gpu_graph._flatten_arc_delay_data = flatten_arc_delay_data.data();

  return gpu_graph;
}

/**
 * @brief convert gpu analysis mode for cpu.
 *
 * @param gpu_analysis_mode
 * @return auto
 */
auto convert_analysis_mode(GPU_Analysis_Mode gpu_analysis_mode) {
  return gpu_analysis_mode == GPU_Analysis_Mode::kMax ? AnalysisMode::kMax
                                                      : AnalysisMode::kMin;
};

/**
 * @brief convert gpu trans type for cpu.
 *
 * @param gpu_trans_type
 * @return auto
 */
auto convert_trans_type(GPU_Trans_Type gpu_trans_type) {
  return gpu_trans_type == GPU_Trans_Type::kRise ? TransType::kRise
                                                 : TransType::kFall;
};

/**
 * @brief update sta slew data.
 *
 * @param the_sta_graph
 * @param the_cpu_graph
 */
void update_sta_slew_data(StaGraph* the_sta_graph, GPU_Graph& the_cpu_graph) {
  auto& the_sta_vertexes = the_sta_graph->get_vertexes();
  auto* the_cpu_graph_vertexes = the_cpu_graph._vertices;
  // iterate each vertex in gpu graph.
  for (unsigned vertex_index = 0; vertex_index < the_cpu_graph._num_vertices;
       ++vertex_index) {
    auto& current_vertex = the_cpu_graph_vertexes[vertex_index];
    auto& current_sta_vertex = the_sta_vertexes[vertex_index];

    // update vertex slew
    for (unsigned slew_index = 0;
         slew_index < current_vertex._slew_data._num_fwd_data; ++slew_index) {
      unsigned vertex_slew_pos =
          current_vertex._slew_data._start_pos + slew_index;
      auto slew_fwd_data = the_cpu_graph._flatten_slew_data[vertex_slew_pos];
      unsigned src_vertex_id = slew_fwd_data._src_vertex_id;
      auto& src_sta_vertex = the_sta_vertexes[src_vertex_id];
      unsigned src_data_index = slew_fwd_data._src_data_index;
      auto src_slew_fwd_data = the_cpu_graph._flatten_slew_data[src_data_index];

      // get src and current slew data.
      auto src_sta_slew_data = src_sta_vertex->getSlewData(
          convert_analysis_mode(src_slew_fwd_data._analysis_mode),
          convert_trans_type(src_slew_fwd_data._trans_type), nullptr);

      auto current_sta_slew_data = current_sta_vertex->getSlewData(
          convert_analysis_mode(slew_fwd_data._analysis_mode),
          convert_trans_type(slew_fwd_data._trans_type), nullptr);

      current_sta_slew_data->set_slew(slew_fwd_data._data_value);
      current_sta_slew_data->set_bwd(src_sta_slew_data);
    }
  }
}

/**
 * @brief update path delay data.
 *
 * @param the_sta_graph
 * @param the_cpu_graph
 */
void update_sta_at_data(StaGraph* the_sta_graph, GPU_Graph& the_cpu_graph) {
  auto& the_sta_vertexes = the_sta_graph->get_vertexes();
  auto* the_cpu_graph_vertexes = the_cpu_graph._vertices;
  // iterate each vertex in gpu graph.
  for (unsigned vertex_index = 0; vertex_index < the_cpu_graph._num_vertices;
       ++vertex_index) {
    auto& current_vertex = the_cpu_graph_vertexes[vertex_index];
    auto& current_sta_vertex = the_sta_vertexes[vertex_index];

    // update vertex at
    for (unsigned at_index = 0;
         at_index < current_vertex._at_data._num_fwd_data; ++at_index) {
      unsigned vertex_slew_pos = current_vertex._at_data._start_pos + at_index;
      auto at_fwd_data = the_cpu_graph._flatten_at_data[vertex_slew_pos];
      unsigned src_vertex_id = at_fwd_data._src_vertex_id;
      auto& src_sta_vertex = the_sta_vertexes[src_vertex_id];
      unsigned src_data_index = at_fwd_data._src_data_index;
      auto src_at_fwd_data = the_cpu_graph._flatten_at_data[src_data_index];

      // get src and current slew data.
      auto* src_sta_at_data = src_sta_vertex->getSlewData(
          convert_analysis_mode(src_at_fwd_data._analysis_mode),
          convert_trans_type(src_at_fwd_data._trans_type), nullptr);

      auto* current_sta_at_data = current_sta_vertex->getSlewData(
          convert_analysis_mode(src_at_fwd_data._analysis_mode),
          convert_trans_type(src_at_fwd_data._trans_type), nullptr);

      current_sta_at_data->set_slew(at_fwd_data._data_value);
      current_sta_at_data->set_bwd(src_sta_at_data);
    }
  }
}

/**
 * @brief update sta arc delay data.
 *
 * @param the_sta_graph
 * @param the_cpu_graph
 */
void update_sta_arc_delay_data(StaGraph* the_sta_graph,
                               GPU_Graph& the_cpu_graph) {
  auto& the_sta_arcs = the_sta_graph->get_arcs();
  auto* the_cpu_graph_arcs = the_cpu_graph._arcs;

  // iterate each arc in gpu graph.
  for (unsigned arc_index = 0; arc_index < the_cpu_graph._num_arcs;
       ++arc_index) {
    auto& current_arc = the_cpu_graph_arcs[arc_index];
    auto& current_sta_arc = the_sta_arcs[arc_index];

    // update arc delay.
    for (unsigned arc_index = 0;
         arc_index < current_arc._delay_values._num_fwd_data; ++arc_index) {
      unsigned arc_delay_pos = current_arc._delay_values._start_pos + arc_index;
      auto arc_delay_fwd_data =
          the_cpu_graph._flatten_arc_delay_data[arc_delay_pos];

      auto* arc_delay_data = current_sta_arc->getArcDelayData(
          convert_analysis_mode(arc_delay_fwd_data._analysis_mode),
          convert_trans_type(arc_delay_fwd_data._trans_type));
      arc_delay_data->set_arc_delay(arc_delay_fwd_data._data_value);
    }
  }
}

/**
 * @brief from the gpu graph data to update the sta graph.
 *
 * @param the_cpu_graph
 * @param the_sta_graph
 */
void update_sta_graph(GPU_Graph& the_cpu_graph, StaGraph* the_sta_graph) {
  update_sta_slew_data(the_sta_graph, the_cpu_graph);
  update_sta_at_data(the_sta_graph, the_cpu_graph);
  update_sta_arc_delay_data(the_sta_graph, the_cpu_graph);
}
    

/**
 * @brief The wrapper for gpu fwd propagation.
 * 
 * @param the_graph 
 * @return unsigned 
 */
unsigned StaGPUFwdPropagation::operator()(StaGraph* the_graph) {
  // prepare the gpu graph data in cpu.
  unsigned vertex_data_size;
  unsigned arc_data_size;
  std::map<StaArc*, unsigned> arc_to_index;
  auto host_graph =
      build_gpu_graph(the_graph, vertex_data_size, arc_data_size, arc_to_index);

  // prepare the lib data.
  auto* ista = getSta();
  ista->buildLibArcsGPU();
  std::vector<ista::Lib_Arc_GPU*> lib_arcs_gpu = ista->getLibArcsGPU();
  Lib_Data_GPU lib_data_gpu;
  build_lib_data_gpu(lib_data_gpu, lib_arcs_gpu);

    std::map<unsigned, std::vector<unsigned>> level_to_arcs;
  for (auto& [level, the_arcs] : _level_to_arcs) {
    std::vector<unsigned> arc_indexes;
    for (auto* the_arc : the_arcs) {
      arc_indexes.emplace_back(arc_to_index[the_arc]);
    }
  }

  // cpu the cuda gpu program.
  gpu_propagate_fwd(host_graph, vertex_data_size, arc_data_size, level_to_arcs,
                    lib_data_gpu);

  // update the sta graph.
  update_sta_graph(host_graph, the_graph);
  return 1;
}

}

