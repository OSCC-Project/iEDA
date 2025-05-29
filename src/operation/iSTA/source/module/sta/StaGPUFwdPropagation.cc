/**
 * @file StaGPUPropagation.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The host api for gpu propagation.
 * @version 0.1
 * @date 2025-02-02
 *
 */

#if CUDA_PROPAGATION

#include "StaGPUFwdPropagation.hh"
#include "ThreadPool/ThreadPool.h"
#include "Sta.hh"
#include "StaDataPropagationBFS.hh"
#include "propagation-cuda/fwd_propagation.cuh"
#include "propagation-cuda/lib_arc.cuh"


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
    std::vector<GPU_Fwd_Data<int64_t>>& flatten_at_data,
    std::map<StaPathDelayData*, unsigned>& at_to_index,
    std::map<unsigned, StaPathDelayData*>& index_to_at,
    std::map<StaClock*, unsigned>& clock_to_index) {
  // build at data.
  // the_vertex->initPathDelayData();
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

    auto* own_clock = path_delay_data->get_launch_clock_data()->get_prop_clock();
    unsigned own_clock_index = clock_to_index[own_clock];
    gpu_at_data._own_clock_index = own_clock_index;

    flatten_at_data.emplace_back(gpu_at_data);
    
    unsigned gpu_at_index = flatten_at_data.size() - 1;
    at_to_index[path_delay_data] = gpu_at_index;
    index_to_at[gpu_at_index] = path_delay_data;
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
    std::vector<GPU_Fwd_Data<float>>& flatten_node_cap_data) {
  gpu_vertex._node_cap_data._start_pos = flatten_node_cap_data.size();
  FOREACH_MODE_TRANS(mode, trans) {
    GPU_Fwd_Data<float> gpu_node_cap_data;
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
    std::vector<GPU_Fwd_Data<float>>& flatten_node_delay_data) {
  gpu_vertex._node_delay_data._start_pos = flatten_node_delay_data.size();
  FOREACH_MODE_TRANS(mode, trans) {
    GPU_Fwd_Data<float> gpu_node_delay_data;
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
  gpu_vertex._node_delay_data._num_fwd_data =
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
    std::vector<GPU_Fwd_Data<float>>& flatten_node_impulse_data) {
  gpu_vertex._node_impulse_data._start_pos = flatten_node_impulse_data.size();
  FOREACH_MODE_TRANS(mode, trans) {
    GPU_Fwd_Data<float> gpu_node_impulse_data;
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
  gpu_arc._delay_values._num_fwd_data =
      flatten_arc_delay_data.size() - gpu_arc._delay_values._start_pos;
}

/**
 * @brief build gpu graph for gpu speed computation.
 *
 * @param the_sta_graph
 * @param vertex_data_size the flatten vertex data size
 * @param arc_data_size the flatten arc data size
 * @param arc_to_index sta arc to gpu arc index
 * @param index_to_arc gpu arc index to sta arc
 * @return GPU_Graph
 */
GPU_Graph build_gpu_graph(StaGraph* the_sta_graph,
                          std::map<StaClock*, unsigned>& clock_to_index,
                          GPU_Flatten_Data& flatten_data,
                          std::vector<GPU_Vertex>& gpu_vertices,
                          std::vector<GPU_Arc>& gpu_arcs,
                          std::map<StaArc*, unsigned>& arc_to_index,
                          std::map<unsigned, StaArc*>& index_to_arc, 
                          std::map<StaPathDelayData*, unsigned>& at_to_index,
                          std::map<unsigned, StaPathDelayData*>& index_to_at) {
  GPU_Graph gpu_graph;
  
  // sort by level to improve data locality.
  the_sta_graph->sortVertexByLevel();
  // build gpu vertex
  StaVertex* the_vertex;
  std::map<StaVertex*, unsigned> vertex_to_id;

  auto build_vertex_data = [&flatten_data, &vertex_to_id, &gpu_vertices,
                            &at_to_index, &index_to_at,
                            &clock_to_index](auto* the_vertex) {
    GPU_Vertex gpu_vertex;

    build_gpu_vertex_slew_data(the_vertex, gpu_vertex,
                               flatten_data._flatten_slew_data);
    build_gpu_vertex_at_data(the_vertex, gpu_vertex,
                             flatten_data._flatten_at_data, at_to_index,
                             index_to_at, clock_to_index);
    build_gpu_vertex_node_cap_data(the_vertex, gpu_vertex,
                                   flatten_data._flatten_node_cap_data);
    build_gpu_vertex_node_delay_data(the_vertex, gpu_vertex,
                                     flatten_data._flatten_node_delay_data);
    build_gpu_vertex_node_impulse_data(the_vertex, gpu_vertex,
                                       flatten_data._flatten_node_impulse_data);
    vertex_to_id[the_vertex] = gpu_vertices.size();
    gpu_vertices.emplace_back(std::move(gpu_vertex));
  };

  FOREACH_VERTEX(the_sta_graph, the_vertex) {
    build_vertex_data(the_vertex);
    LOG_INFO_EVERY_N(10000) << "build gpu vertexes: " << gpu_vertices.size();
  }

  auto& main_to_assistant = the_sta_graph->get_main2assistant();
  for (auto& [main, assistant] : main_to_assistant) {
    build_vertex_data(assistant.get());
  }

  LOG_INFO << "build gpu vertexes num: " << gpu_vertices.size();

  // build gpu arc
  StaArc* the_arc;
  FOREACH_ARC(the_sta_graph, the_arc) {
    // skip not used arc.
    if (!the_arc->isDelayArc() && !the_arc->isCheckArc() &&
        !the_arc->isNetArc()) {
      continue;
    }

    // for back copy gpu data to arc delay data.
    the_arc->initArcDelayData();

    GPU_Arc gpu_arc;
    gpu_arc._src_vertex_id = vertex_to_id.at(the_arc->get_src());
    gpu_arc._snk_vertex_id = vertex_to_id.at(the_arc->get_snk());
    if (the_arc->isInstArc()) {
      gpu_arc._lib_data_arc_id =
          dynamic_cast<StaInstArc*>(the_arc)->get_lib_arc_id();
    } else {
      gpu_arc._lib_data_arc_id = -1;
    }

    gpu_arc._arc_type = the_arc->isNetArc()     ? GPU_Arc_Type::kNet
                        : the_arc->isCheckArc() ? GPU_Arc_Type::kInstCheckArc
                                                : GPU_Arc_Type::kInstDelayArc;
    gpu_arc._arc_trans_type =
        the_arc->isInstArc()
            ? the_arc->isPositiveArc()   ? GPU_Arc_Trans_Type::kPositive
              : the_arc->isNegativeArc() ? GPU_Arc_Trans_Type::kNegative
                                         : GPU_Arc_Trans_Type::kNonUnate
            : GPU_Arc_Trans_Type::kPositive;

    build_gpu_arc_delay_data(gpu_arc, flatten_data._flatten_arc_delay_data);

    gpu_arcs.emplace_back(std::move(gpu_arc));

    unsigned gpu_arc_index = gpu_arcs.size() - 1;
    arc_to_index[the_arc] = gpu_arc_index;
    index_to_arc[gpu_arc_index] = the_arc;

    LOG_INFO_EVERY_N(10000) << "build gpu arc: " << gpu_arcs.size();
  }

  LOG_INFO << "build gpu arcs num: " << gpu_arcs.size();

  // copy cpu data to gpu memory.
  gpu_graph._vertices = gpu_vertices.data();
  gpu_graph._arcs = gpu_arcs.data();

  gpu_graph._num_vertices = gpu_vertices.size();
  gpu_graph._num_arcs = gpu_arcs.size();
  gpu_graph._flatten_slew_data = flatten_data._flatten_slew_data.data();
  gpu_graph._flatten_at_data = flatten_data._flatten_at_data.data();
  gpu_graph._flatten_node_cap_data = flatten_data._flatten_node_cap_data.data();
  gpu_graph._flatten_node_delay_data =
      flatten_data._flatten_node_delay_data.data();
  gpu_graph._flatten_node_impulse_data =
      flatten_data._flatten_node_impulse_data.data();
  gpu_graph._flatten_arc_delay_data =
      flatten_data._flatten_arc_delay_data.data();

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
 * @param the_host_graph
 */
void update_sta_slew_data(StaGraph* the_sta_graph, GPU_Graph& the_host_graph) {
  LOG_INFO << "update to sta slew start.";
  LOG_INFO << "update num vertexes " << the_host_graph._num_vertices;

  auto& the_sta_vertexes = the_sta_graph->get_vertexes();
  auto the_sta_assistants = the_sta_graph->getAssistants();
  auto* the_host_graph_vertexes = the_host_graph._vertices;

  // iterate each vertex in gpu graph.
  for (unsigned vertex_index = 0; vertex_index < the_host_graph._num_vertices;
       ++vertex_index) {
    auto& current_vertex = the_host_graph_vertexes[vertex_index];
    StaVertex* current_sta_vertex = nullptr;
    if (vertex_index < the_sta_vertexes.size()) {
      current_sta_vertex = the_sta_vertexes[vertex_index].get();
    } else {
      current_sta_vertex = the_sta_assistants[vertex_index -
                                              the_sta_vertexes.size()];
    }

    // update vertex slew
    for (unsigned slew_index = 0;
         slew_index < current_vertex._slew_data._num_fwd_data; ++slew_index) {
      unsigned vertex_slew_pos =
          current_vertex._slew_data._start_pos + slew_index;
      auto slew_fwd_data = the_host_graph._flatten_slew_data[vertex_slew_pos];

      auto current_sta_slew_data = current_sta_vertex->getSlewData(
          convert_analysis_mode(slew_fwd_data._analysis_mode),
          convert_trans_type(slew_fwd_data._trans_type), nullptr);
      current_sta_slew_data->set_slew(slew_fwd_data._data_value);
 
    }
  }

  LOG_INFO << "update to sta slew end.";
}

/**
 * @brief update path delay data.
 *
 * @param the_sta_graph
 * @param the_host_graph
 */
void update_sta_at_data(StaGraph* /*the_sta_graph*/, GPU_Graph& the_host_graph,
                        std::map<unsigned, StaPathDelayData*>& index_to_at) {
  LOG_INFO << "update to sta at start.";
  LOG_INFO << "update num vertexes " << the_host_graph._num_vertices;

  for (unsigned at_index = 0; at_index < the_host_graph._num_at_data; ++at_index) {
    auto& current_at_data = the_host_graph._flatten_at_data[at_index];
    auto* path_delay_data = index_to_at[at_index];
    path_delay_data->set_arrive_time(current_at_data._data_value);
  }

  LOG_INFO << "update to sta at end.";
}

/**
 * @brief update sta arc delay data.
 *
 * @param the_sta_graph
 * @param the_host_graph
 */
void update_sta_arc_delay_data(StaGraph* /*the_sta_graph*/,
                               GPU_Graph& the_host_graph,
                               std::map<unsigned, StaArc*>& index_to_arc) {
  LOG_INFO << "update to arc delay start.";
  LOG_INFO << "update num arcs " << the_host_graph._num_arcs;

  auto* the_host_graph_arcs = the_host_graph._arcs;

  // iterate each arc in gpu graph.
  for (unsigned arc_index = 0; arc_index < the_host_graph._num_arcs;
       ++arc_index) {
    auto& current_arc = the_host_graph_arcs[arc_index];
    auto& current_sta_arc = index_to_arc[arc_index];
    // update arc delay.
    for (unsigned arc_data_index = 0;
         arc_data_index < current_arc._delay_values._num_fwd_data;
         ++arc_data_index) {
      unsigned arc_delay_pos =
          current_arc._delay_values._start_pos + arc_data_index;
      auto arc_delay_fwd_data =
          the_host_graph._flatten_arc_delay_data[arc_delay_pos];

      auto* arc_delay_data = current_sta_arc->getArcDelayData(
          convert_analysis_mode(arc_delay_fwd_data._analysis_mode),
          convert_trans_type(arc_delay_fwd_data._trans_type));

      arc_delay_data->set_arc_delay(arc_delay_fwd_data._data_value);
    }
  }

  LOG_INFO << "update to arc delay end.";
}

/**
 * @brief from the gpu graph data to update the sta graph.
 *
 * @param the_host_graph
 * @param the_sta_graph
 */
void update_sta_graph(GPU_Graph& the_host_graph, StaGraph* the_sta_graph,
                      std::map<unsigned, StaArc*>& index_to_arc,
                      std::map<unsigned, StaPathDelayData*>& index_to_at) {
  LOG_INFO << "update to sta graph start.";
  update_sta_slew_data(the_sta_graph, the_host_graph);
  update_sta_at_data(the_sta_graph, the_host_graph, index_to_at);
  update_sta_arc_delay_data(the_sta_graph, the_host_graph, index_to_arc);
  LOG_INFO << "update to sta graph end.";
}

/**
 * @brief prepare GPU data for fwd propagation.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaGPUFwdPropagation::prepareGPUData(StaGraph* the_graph) {
  LOG_INFO << "prepare gpu data start";
#if !CPU_SIM
  // prepare the lib data.
  LOG_INFO << "prepare lib data start";
  CPU_PROF_START(0);

  auto* ista = getSta();
  ista->buildLibArcsGPU();
  auto& lib_arcs_host = ista->get_lib_gpu_arcs();
  Lib_Data_GPU lib_data_gpu;
  std::vector<Lib_Table_GPU> lib_gpu_tables;
  std::vector<Lib_Table_GPU*> lib_gpu_tables_ptrs; 
  build_lib_data_gpu(lib_data_gpu, lib_gpu_tables, lib_gpu_tables_ptrs, lib_arcs_host);

  // save lib data
  ista->set_gpu_lib_data(std::move(lib_data_gpu));
  ista->set_lib_gpu_tables(std::move(lib_gpu_tables));
  ista->set_lib_gpu_table_ptr(std::move(lib_gpu_tables_ptrs));

  LOG_INFO << "prepare lib data end";
  CPU_PROF_END(0, "prepare lib data");

  // prepare the gpu graph data in cpu.
  LOG_INFO << "prepare gpu graph data start";
  CPU_PROF_START(1);
  unsigned num_vertex = the_graph->numVertex();
  unsigned num_arc = the_graph->numArc();

  unsigned vertex_data_size = c_gpu_num_vertex_data * num_vertex;
  unsigned arc_data_size = c_gpu_num_arc_delay * num_arc;

  std::map<StaArc*, unsigned> arc_to_index;
  std::map<unsigned, StaArc*> index_to_arc;

  std::map<StaPathDelayData*, unsigned> at_to_index;
  std::map<unsigned, StaPathDelayData*> index_to_at;

  auto clock_to_index = ista->getClockToIndex();

  // gpu graph vertex and arc data.
  std::vector<GPU_Vertex> gpu_vertices;
  gpu_vertices.reserve(num_vertex);
  std::vector<GPU_Arc> gpu_arcs;
  gpu_arcs.reserve(num_arc);

  // flatten data
  GPU_Flatten_Data flatten_data;
  flatten_data._flatten_slew_data.reserve(vertex_data_size);
  flatten_data._flatten_at_data.reserve(vertex_data_size);
  flatten_data._flatten_node_cap_data.reserve(vertex_data_size);
  flatten_data._flatten_node_delay_data.reserve(vertex_data_size);
  flatten_data._flatten_node_impulse_data.reserve(vertex_data_size);
  flatten_data._flatten_arc_delay_data.reserve(arc_data_size);

  auto the_host_graph =
      build_gpu_graph(the_graph, clock_to_index, flatten_data, gpu_vertices, gpu_arcs,
                      arc_to_index, index_to_arc, at_to_index, index_to_at);
  // set data size.
  the_host_graph._num_slew_data = flatten_data._flatten_slew_data.size();
  the_host_graph._num_at_data = flatten_data._flatten_at_data.size();
  the_host_graph._num_node_cap_data = flatten_data._flatten_node_cap_data.size();
  the_host_graph._num_node_delay_data = flatten_data._flatten_node_delay_data.size();
  the_host_graph._num_node_impulse_data = flatten_data._flatten_node_impulse_data.size();
  the_host_graph._num_arc_delay_data = flatten_data._flatten_arc_delay_data.size();

  // save the graph and data.
  ista->set_gpu_graph(std::move(the_host_graph));
  ista->set_gpu_vertices(std::move(gpu_vertices));
  ista->set_gpu_arcs(std::move(gpu_arcs));
  ista->set_flatten_data(std::move(flatten_data));

  LOG_INFO << "prepare gpu graph data end";
  CPU_PROF_END(1, "prepare gpu graph data");

  // prepare level arcs data.
  LOG_INFO << "prepare level arcs data start";
  CPU_PROF_START(2);

  std::map<unsigned, std::vector<unsigned>> level_to_arc_index;
  for (auto& [level, the_arcs] : _level_to_arcs) {
    std::vector<unsigned> arc_indexes;
    for (auto* the_arc : the_arcs) {
      arc_indexes.emplace_back(arc_to_index.at(the_arc));
    }
    level_to_arc_index[level] = std::move(arc_indexes);
  }

  _level_to_arc_index = std::move(level_to_arc_index);
  _index_to_arc = std::move(index_to_arc);

  LOG_INFO << "prepare level arcs data end";
  CPU_PROF_END(2, "prepare level arcs data");

  ista->set_arc_to_index(std::move(arc_to_index));
  ista->set_at_to_index(std::move(at_to_index));
  ista->set_index_to_at(std::move(index_to_at));
#endif

  LOG_INFO << "prepare gpu data end";

  return 1;
}

/**
 * @brief The wrapper for gpu fwd propagation.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaGPUFwdPropagation::operator()(StaGraph* the_graph) {
#if CPU_SIM
  // use cpu simulation the gpu fwd propagation.
  LOG_INFO << "dispatch arc task to cpu start";
  unsigned num_threads = getNumThreads();
  // create thread pool
  
  StaFwdPropagationBFS fwd_prop_bfs;
  for (auto& [level, the_arcs] : _level_to_arcs) {
    LOG_INFO << "propagate level " << level;
    {
      ThreadPool pool(num_threads);
      for (auto* the_arc : the_arcs) {
        pool.enqueue(
            [&fwd_prop_bfs](StaArc* the_arc) {
              return the_arc->exec(fwd_prop_bfs);
            },
            the_arc);
      }
    }
  }
  _level_to_arcs.clear();
  LOG_INFO << "dispatch arc task to cpu end";

#else
  LOG_INFO << "dispatch arc task to gpu start";

  {
    auto* ista = getSta();
    auto& the_host_graph = ista->get_gpu_graph();

    unsigned num_vertex = the_graph->numVertex();
    unsigned num_arc = the_graph->numArc();

    unsigned vertex_data_size = c_gpu_num_vertex_data * num_vertex;
    unsigned arc_data_size = c_gpu_num_arc_delay * num_arc;

    auto& lib_data_gpu = ista->get_gpu_lib_data();

    auto& index_to_at = ista->get_index_to_at();

    // call the cuda gpu program.
    gpu_propagate_fwd(the_host_graph, vertex_data_size, arc_data_size,
                      _level_to_arc_index, lib_data_gpu);

    // update result to the sta graph.
    update_sta_graph(the_host_graph, the_graph, _index_to_arc, index_to_at);
  }

  LOG_INFO << "dispatch arc task to gpu end";
#endif

  return 1;
}

}  // namespace ista

#endif