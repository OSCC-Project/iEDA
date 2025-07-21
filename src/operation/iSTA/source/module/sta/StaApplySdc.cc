// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of
// Sciences Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan
// PSL v2. You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
/**
 * @file StaApplySdc.cc
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The implemention of apply sdc to graph.
 * @version 0.1
 * @date 2021-03-01
 */

#include "StaApplySdc.hh"

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "Sta.hh"
#include "Type.hh"
#include "sdc/SdcClock.hh"
#include "sdc/SdcException.hh"
#include "sdc/SdcSetClockLatency.hh"
#include "sdc/SdcSetClockUncertainty.hh"
#include "sdc/SdcSetIODelay.hh"
#include "sdc/SdcSetInputTransition.hh"
#include "sdc/SdcSetLoad.hh"
#include "sdc/SdcTimingDRC.hh"
#include "sdc/SdcTimingDerate.hh"
#include "sta/StaBuildExceptionTag.hh"
#include "sta/StaClock.hh"
#include "sta/StaData.hh"

namespace ista {
/**
 * @brief Setup clocks to sta graph.
 *
 * @param sdc_clocks
 * @return unsigned success return 1, or return 0.
 */
unsigned StaApplySdc::setupClocks(StrMap<std::unique_ptr<SdcClock>>& sdc_clocks,
                                  StaGraph* the_graph) {
  Sta* ista = getSta();
  for (auto& [clock_name, sdc_clock] : sdc_clocks) {
    if (_prop_type == PropType::kApplySdcPostNormalClockProp) {
      if (sdc_clock->isGenerateClock() &&
          dynamic_cast<SdcGenerateCLock*>(sdc_clock.get())
              ->isNeedUpdateSourceClock()) {
        // only process generate clock which need update source clock.
        StaWaveForm wave_form;
        auto& edges = sdc_clock->get_edges();
        for (auto edge : edges) {
          wave_form.addWaveEdge(NS_TO_PS(edge));
        }
        auto* sta_clock = ista->findClock(clock_name);
        int period_ps = NS_TO_PS(sdc_clock->get_period());
        sta_clock->set_period(period_ps);
        sta_clock->set_wave_form(std::move(wave_form));
      }
    } else {
      std::unique_ptr<StaClock> sta_clock =
          std::make_unique<StaClock>(clock_name, StaClock::ClockType::kIdeal,
                                     NS_TO_PS(sdc_clock->get_period()));

      auto design_objs = sdc_clock->get_objs();
      for (auto* design_obj : design_objs) {
        auto the_vertex = the_graph->findVertex(design_obj);
        LOG_FATAL_IF(!the_vertex) << "The vertex is not exist.";
        (*the_vertex)->set_is_sdc_clock_pin();
        sta_clock->addVertex(*the_vertex);
      }

      StaWaveForm wave_form;
      auto& edges = sdc_clock->get_edges();
      for (auto edge : edges) {
        wave_form.addWaveEdge(NS_TO_PS(edge));
      }
      sta_clock->set_wave_form(std::move(wave_form));

      if (sdc_clock->isPropagatedClock()) {
        sta_clock->setPropagateClock();
      }

      if (_prop_type == PropType::kApplySdcPreProp) {
        if (sdc_clock->isGenerateClock() &&
            dynamic_cast<SdcGenerateCLock*>(sdc_clock.get())
                ->isNeedUpdateSourceClock()) {
          sta_clock->set_is_need_update_period_waveform(true);
        }
      }

      ista->addClock(std::move(sta_clock));
    }
  }

  return 1;
}

/**
 * @brief Apply set_input_transtion to the graph.
 *
 * @param io_constraint
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::setupInputTransition(
    const std::unique_ptr<SdcIOConstrain>& io_constraint, StaGraph* the_graph) {
  auto construct_slew_data = [](AnalysisMode delay_type, TransType trans_type,
                                StaVertex* own_vertex, double slew) {
    StaSlewData* slew_data =
        new StaSlewData(delay_type, trans_type, own_vertex, NS_TO_FS(slew));
    own_vertex->addData(slew_data);
  };

  auto* set_input_transition =
      dynamic_cast<SdcSetInputTransition*>(io_constraint.get());
  double slew = set_input_transition->get_transition_value();
  auto& objs = set_input_transition->get_objs();
  for (auto* obj : objs) {
    auto the_vertex = the_graph->findVertex(obj);
    if (the_vertex) {
      if (set_input_transition->isMax()) {
        if (set_input_transition->isRise()) {
          construct_slew_data(AnalysisMode::kMax, TransType::kRise, *the_vertex,
                              slew);
        }
        if (set_input_transition->isFall()) {
          construct_slew_data(AnalysisMode::kMax, TransType::kFall, *the_vertex,
                              slew);
        }
      }
      if (set_input_transition->isMin()) {
        if (set_input_transition->isRise()) {
          construct_slew_data(AnalysisMode::kMin, TransType::kRise, *the_vertex,
                              slew);
        }
        if (set_input_transition->isFall()) {
          construct_slew_data(AnalysisMode::kMin, TransType::kFall, *the_vertex,
                              slew);
        }
      }
    }
  }

  return 1;
}

/**
 * @brief Apply set_load to the graph.
 *
 * @param io_constraint
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::setupOutputLoad(
    const std::unique_ptr<SdcIOConstrain>& io_constraint,
    StaGraph* /* the_graph */) {
  auto* set_load = dynamic_cast<SdcSetLoad*>(io_constraint.get());
  double load = set_load->get_load_value();
  auto& objs = set_load->get_objs();
  for (auto* obj : objs) {
    LOG_FATAL_IF(!obj->isPort());
    auto* the_port = dynamic_cast<Port*>(obj);
    if (set_load->isMax()) {
      if (set_load->isRise()) {
        the_port->set_cap(AnalysisMode::kMax, TransType::kRise, load);
      }
      if (set_load->isFall()) {
        the_port->set_cap(AnalysisMode::kMax, TransType::kFall, load);
      }
    }
    if (set_load->isMin()) {
      if (set_load->isRise()) {
        the_port->set_cap(AnalysisMode::kMin, TransType::kRise, load);
      }
      if (set_load->isFall()) {
        the_port->set_cap(AnalysisMode::kMin, TransType::kFall, load);
      }
    }
  }

  return 1;
}

/**
 * @brief Apply set_input_delay/set_output_delay to the graph.
 *
 * @param io_constraint
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::setupIODelay(
    const std::unique_ptr<SdcIOConstrain>& io_constraint, StaGraph* the_graph) {
  auto* set_io_delay = dynamic_cast<SdcSetIODelay*>(io_constraint.get());

  auto& objs = set_io_delay->get_objs();

  auto* ista = getSta();
  for (auto* obj : objs) {
    auto the_vertex = the_graph->findVertex(obj);
    LOG_FATAL_IF(!the_vertex);
    if (obj->isInout() && set_io_delay->isSetOutputDelay()) {
      the_vertex = the_graph->getAssistant(the_vertex.value());
    }

    if (the_vertex) {
      ista->addIODelayConstrain(*the_vertex, set_io_delay);
    }

    // create input delay launch clock data for power analysis.
    if (set_io_delay->isSetInputDelay()) {
      auto* launch_clock = ista->findClock(set_io_delay->get_clock_name());
      DLOG_FATAL_IF(!launch_clock)
          << "launch clock " << set_io_delay->get_clock_name()
          << " is not found.";
      unsigned is_clock_fall = set_io_delay->isClockFall();
      auto construct_clock_data = [&the_vertex, launch_clock,
                                   is_clock_fall](auto analysis_mode) {
        auto clock_datas =
            !is_clock_fall
                ? (*the_vertex)->getClockData(analysis_mode, TransType::kRise)
                : (*the_vertex)->getClockData(analysis_mode, TransType::kFall);

        if (clock_datas.empty()) {
          StaClockData* launch_clock_data = nullptr;
          if (!is_clock_fall) {
            launch_clock_data = new StaClockData(
                analysis_mode, TransType::kRise, 0, *the_vertex, launch_clock);
            launch_clock_data->set_clock_wave_type(TransType::kRise);
            (*the_vertex)->addData(launch_clock_data);

          } else {
            launch_clock_data = new StaClockData(
                analysis_mode, TransType::kFall, 0, *the_vertex, launch_clock);
            launch_clock_data->set_clock_wave_type(TransType::kFall);
            (*the_vertex)->addData(launch_clock_data);
          }
        }
      };

      if (ista->isMaxAnalysis() && set_io_delay->isMax()) {
        construct_clock_data(AnalysisMode::kMax);
      }

      if (ista->isMinAnalysis() && set_io_delay->isMin()) {
        construct_clock_data(AnalysisMode::kMin);
      }
    }
  }
  return 1;
}

/**
 * @brief Setup IO constrain.
 *
 * @param sdc_io_constrain
 * @return unsigned success return 1, or return 0.
 */
unsigned StaApplySdc::setupIOConstrain(
    std::vector<std::unique_ptr<SdcIOConstrain>>& sdc_io_constraints,
    StaGraph* the_graph) {
  std::function<unsigned(const std::unique_ptr<SdcIOConstrain>& io_constraint,
                         StaGraph* the_graph)>
      f;

  std::map<std::string, decltype(f)> dispatch_funs = {
      {"set_input_transition",
       std::bind(&StaApplySdc::setupInputTransition, this,
                 std::placeholders::_1, std::placeholders::_2)},
      {"set_load", std::bind(&StaApplySdc::setupOutputLoad, this,
                             std::placeholders::_1, std::placeholders::_2)},
      {"set_input_delay",
       std::bind(&StaApplySdc::setupIODelay, this, std::placeholders::_1,
                 std::placeholders::_2)},
      {"set_output_delay",
       std::bind(&StaApplySdc::setupIODelay, this, std::placeholders::_1,
                 std::placeholders::_2)}};

  unsigned is_ok = 1;
  for (auto& io_constraint : sdc_io_constraints) {
    LOG_FATAL_IF(!dispatch_funs.contains(io_constraint->get_constrain_name()))
        << io_constraint->get_constrain_name() << " has not process func.";
    is_ok = dispatch_funs[io_constraint->get_constrain_name()](io_constraint,
                                                               the_graph);
    if (!is_ok) {
      break;
    }
  }

  return is_ok;
}

/**
 * @brief Setup the timing drc to the vertex.
 *
 * @param sdc_timing_drc
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::setupTimingDrc(
    std::vector<std::unique_ptr<SdcTimingDRC>>& sdc_timing_drcs,
    StaGraph* the_graph) {
  auto* ista = getSta();
  auto set_drc = []<typename T>(T t, auto* timing_drc, double drc_value) {
    if (timing_drc->isMaxCap()) {
      if (timing_drc->isRise()) {
        t->setMaxRiseCap(drc_value);
      }

      if (timing_drc->isFall()) {
        t->setMaxFallCap(drc_value);
      }
    } else if (timing_drc->isMaxTransition()) {
      if (timing_drc->isRise()) {
        t->setMaxRiseSlew(drc_value);
      }

      if (timing_drc->isFall()) {
        t->setMaxFallSlew(drc_value);
      }
    } else {
      t->setMaxFanout(drc_value);
    }
  };

  for (auto& timing_drc : sdc_timing_drcs) {
    auto& objs = timing_drc->get_objs();
    double drc_value = timing_drc->get_drc_val();

    if (objs.empty()) {
      // set drc value to the whole netlist.
      set_drc(ista, timing_drc.get(), drc_value);
    } else {
      for (auto obj : objs) {
        std::visit(
            overloaded{
                [the_graph, ista, &set_drc, &timing_drc,
                 drc_value](SdcCommandObj* sdc_obj) {
                  unsigned is_rise = timing_drc->isRise();
                  unsigned is_fall = timing_drc->isFall();

                  unsigned is_clock = timing_drc->isClockPath();
                  unsigned is_data = timing_drc->isDataPath();

                  auto* sdc_clock = dynamic_cast<SdcClock*>(sdc_obj);
                  LOG_FATAL_IF(!sdc_clock);
                  auto* sta_clock =
                      ista->findClock(sdc_clock->get_clock_name());

                  StaVertex* the_vertex;
                  FOREACH_VERTEX(the_graph, the_vertex) {
                    bool is_satified = false;
                    if (is_rise) {
                      if (is_clock) {
                        auto prop_clocks = the_vertex->getPropagatedClock(
                            AnalysisMode::kMaxMin, TransType::kRise, false);
                        is_satified = prop_clocks.contains(sta_clock);
                      }

                      if (is_data && !is_satified) {
                        auto prop_clocks = the_vertex->getPropagatedClock(
                            AnalysisMode::kMaxMin, TransType::kRise, true);
                        is_satified = prop_clocks.contains(sta_clock);
                      }
                    }

                    if (is_fall && !is_satified && the_vertex->is_start()) {
                      if (is_clock) {
                        auto prop_clocks = the_vertex->getPropagatedClock(
                            AnalysisMode::kMaxMin, TransType::kFall, false);
                        is_satified = prop_clocks.contains(sta_clock);
                      }

                      if (is_data && !is_satified) {
                        auto prop_clocks = the_vertex->getPropagatedClock(
                            AnalysisMode::kMaxMin, TransType::kFall, true);
                        is_satified = prop_clocks.contains(sta_clock);
                      }
                    }

                    if (is_satified) {
                      set_drc(the_vertex, timing_drc.get(), drc_value);
                    }
                  }
                },
                [the_graph, &set_drc, &timing_drc,
                 drc_value](DesignObject* design_obj) {
                  auto the_vertex = the_graph->findVertex(design_obj);
                  LOG_FATAL_IF(!the_vertex);
                  // set drc value to the vetex.
                  set_drc(*the_vertex, timing_drc.get(), drc_value);
                },
            },
            obj);
      }
    }
  }
  return 1;
}

/*define the index bit for ocv derate*/
#define max_min_bit 2
#define clock_data_bit 1
#define cell_net_bit 0

/**
 * @brief setup ocv derate to sta analysis.
 *
 * @param sdc_timing_derates
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::setupOcvDerate(
    std::vector<std::unique_ptr<SdcTimingDerate>>& sdc_timing_derates,
    StaGraph* the_graph) {
  StaDreateTable derate_table;
  derate_table.init();
  auto set_derate_table =
      [&derate_table](std::unique_ptr<SdcTimingDerate>& sdc_timing_derate) {
        auto process_cell_net =
            [&derate_table](std::unique_ptr<SdcTimingDerate>& sdc_timing_derate,
                            unsigned& index) {
              if (sdc_timing_derate->isCellDelay()) {
                derate_table[index] = sdc_timing_derate->get_derate_value();
              }
              if (sdc_timing_derate->isNetDelay()) {
                index |= (1 << cell_net_bit);
                derate_table[index] = sdc_timing_derate->get_derate_value();
              }
            };

        auto process_clock_data =
            [&derate_table, &process_cell_net](
                std::unique_ptr<SdcTimingDerate>& sdc_timing_derate,
                unsigned& index) {
              if (sdc_timing_derate->isClockDelay()) {
                process_cell_net(sdc_timing_derate, index);
              }
              if (sdc_timing_derate->isDataDelay()) {
                index |= (1 << clock_data_bit);
                process_cell_net(sdc_timing_derate, index);
              }
            };

        auto process_early_late =
            [&derate_table, &process_clock_data](
                std::unique_ptr<SdcTimingDerate>& sdc_timing_derate,
                unsigned& index) {
              if (sdc_timing_derate->isLateDelay()) {
                process_clock_data(sdc_timing_derate, index);
              }
              if (sdc_timing_derate->isEarlyDelay()) {
                index |= (1 << max_min_bit);
                process_clock_data(sdc_timing_derate, index);
              }
            };

        unsigned index = 0;
        process_early_late(sdc_timing_derate, index);
      };

  for (auto& sdc_timing_derate : sdc_timing_derates) {
    set_derate_table(sdc_timing_derate);
  }

  auto* ista = getSta();
  ista->set_derate_table(derate_table);

  return 1;
}

/**
 * @brief Get the exception obj name str of the graph.
 *
 * @param obj_vec origin obj vec, which
 * @param the_graph
 * @param analysis_mode
 * @param trans_type
 * @return std::vector<std::string> The vertex obj strs.
 */
std::vector<std::string> StaApplySdc::getExceptionObjs(
    std::vector<std::string>& obj_vec, StaGraph* the_graph,
    AnalysisMode analysis_mode, TransType trans_type, bool is_from) {
  std::vector<std::string> obj_strs;
  auto* design_nl = the_graph->get_nl();

  for (auto& obj : obj_vec) {
    auto collect_objs = FindObjOfSdc(obj, design_nl);

    for (auto& object : collect_objs) {
      std::visit(
          overloaded{
              [&obj_strs, the_graph, analysis_mode, trans_type,
               is_from](SdcCommandObj* sdc_obj) {
                // should be sdc clock
                auto* sdc_clock = dynamic_cast<SdcClock*>(sdc_obj);
                const char* clock_name = sdc_clock->get_clock_name();
                StaVertex* the_vertex;
                FOREACH_VERTEX(the_graph, the_vertex) {
                  if (the_vertex->isPropClock(clock_name, analysis_mode,
                                              trans_type)) {
                    if (is_from && the_vertex->is_start()) {
                      obj_strs.emplace_back(std::string(the_vertex->getName()));
                    } else if (!is_from && the_vertex->is_end()) {
                      obj_strs.emplace_back(std::string(the_vertex->getName()));
                    }
                  }
                }
              },
              [&obj_strs, the_graph](DesignObject* design_obj) {
                auto the_vertex = the_graph->findVertex(design_obj);
                LOG_FATAL_IF(!the_vertex) << "The vertex is not found.";
                obj_strs.emplace_back(std::string((*the_vertex)->getName()));
              },
          },
          object);
    }
  }
  return obj_strs;
}

/**
 * @brief apply the sdc exception to staArc.
 *
 * @param sdc_exceptions
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::setupException(
    std::vector<std::unique_ptr<SdcException>>& sdc_exceptions,
    StaGraph* the_graph) {
  for (auto& sdc_exception : sdc_exceptions) {
    auto& prop_froms = sdc_exception->get_prop_froms();
    auto from_vertexs = getExceptionObjs(
        prop_froms, the_graph, AnalysisMode::kMaxMin, TransType::kRise, true);
    prop_froms.swap(from_vertexs);

    auto& prop_tos = sdc_exception->get_prop_tos();
    auto to_vertexs = getExceptionObjs(
        prop_tos, the_graph, AnalysisMode::kMaxMin, TransType::kRise, false);
    prop_tos.swap(to_vertexs);

    auto& prop_throughs_list = sdc_exception->get_prop_throughs();
    for (auto& prop_throughs : prop_throughs_list) {
      auto through_vertexs =
          getExceptionObjs(prop_throughs, the_graph, AnalysisMode::kMaxMin,
                           TransType::kRise, false);
      prop_throughs.swap(through_vertexs);
    }

    if (sdc_exception->isMulticyclePath()) {
      StaBuildExceptionTag build_exception(
          StaPropagationTag::TagType::kMulticycle);
      build_exception.set_sdc_exception(sdc_exception.get());
      if (!build_exception(the_graph)) {
        LOG_FATAL << "build exception failed.";
      }
    }
  }

  return 1;
}

/**
 * @brief process clock uncertainty after propagation.
 *
 * @param sdc_clock_uncertaintys
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::processClockUncertainty(
    std::vector<std::unique_ptr<SdcSetClockUncertainty>>&
        sdc_clock_uncertaintys,
    StaGraph* the_graph) {
  unsigned is_ok = 1;

  auto get_uncertainty = [&sdc_clock_uncertaintys]() -> decltype(auto) {
    Multimap<SdcCollectionObj, SdcSetClockUncertainty*> obj2uncertainty;
    for (auto& sdc_clock_uncertainty : sdc_clock_uncertaintys) {
      auto& objs = sdc_clock_uncertainty->get_objs();
      for (auto obj : objs) {
        obj2uncertainty.insert(obj, sdc_clock_uncertainty.get());
      }
    }
    return obj2uncertainty;
  };

  auto* ista = getSta();

  auto apply_clock_uncetainty_to_obj = [ista, the_graph](auto* design_obj,
                                                         auto uncertainty) {
    StaVertex* end_vertex = ista->findVertex(design_obj);
    if (!end_vertex->is_end()) {
      return;
    }

    StaData* delay_data;
    FOREACH_DELAY_DATA(end_vertex, delay_data) {
      // set uncertainty.
      auto analysis_mode = delay_data->get_delay_type();
      auto seq_data_vec = ista->getSeqData(end_vertex, delay_data);
      if (analysis_mode == AnalysisMode::kMax) {
        if (uncertainty->isSetup()) {
          double uncertainty_value = uncertainty->getUncertaintyValueFs();
          for (auto* seq_data : seq_data_vec) {
            seq_data->set_uncertainty(uncertainty_value);
          }
        }
      } else {
        if (uncertainty->isHold()) {
          double uncertainty_value = uncertainty->getUncertaintyValueFs();
          for (auto* seq_data : seq_data_vec) {
            seq_data->set_uncertainty(uncertainty_value);
          }
        }
      }
    }
  };

  auto apply_clock_uncertainty_to_clk = [ista](auto* sdc_clk,
                                               auto* uncertainty) {
    auto cmp = [](StaPathData* left, StaPathData* right) -> bool {
      int left_slack = left->getSlack();
      int right_slack = right->getSlack();
      return left_slack > right_slack;
    };
    std::priority_queue<StaPathData*, std::vector<StaPathData*>, decltype(cmp)>
        seq_data_queue(cmp);

    auto& clk_groups = ista->get_clock_groups();
    for (auto& [clk, seq_path_group] : clk_groups) {
      if (sdc_clk->isAllClock() || (Str::equal(clk->get_clock_name(),
                     dynamic_cast<SdcClock*>(sdc_clk)->get_clock_name()))) {
        StaPathEnd* path_end;
        StaPathData* path_data;
        auto mode =
            uncertainty->isSetup() ? AnalysisMode::kMax : AnalysisMode::kMin;
        double uncertainty_value = uncertainty->getUncertaintyValueFs();

        FOREACH_PATH_GROUP_END(seq_path_group.get(), path_end)
        FOREACH_PATH_END_DATA(path_end, mode, path_data) {
          seq_data_queue.push(path_data);
        }

        while (!seq_data_queue.empty()) {
          auto* seq_path_data =
              dynamic_cast<StaSeqPathData*>(seq_data_queue.top());
          seq_path_data->set_uncertainty(uncertainty_value);
          seq_data_queue.pop();
        }
      }
    }
  };

  auto obj2uncertainty = get_uncertainty();

  for (auto [obj, uncertainty] : obj2uncertainty) {
    std::visit(overloaded{
                   [&apply_clock_uncertainty_to_clk,
                    uncertainty](SdcCommandObj* sdc_obj) {
                     apply_clock_uncertainty_to_clk(sdc_obj, uncertainty);
                   },
                   [&apply_clock_uncetainty_to_obj,
                    uncertainty](DesignObject* design_obj) {
                     apply_clock_uncetainty_to_obj(design_obj, uncertainty);
                   },
               },
               obj);
  }

  return is_ok;
}

/**
 * @brief Apply sdc constrain on sta graph.
 *
 * @param the_graph
 * @return unsigned
 */
unsigned StaApplySdc::operator()(StaGraph* the_graph) {
  LOG_INFO << "apply sdc start";

  Sta* ista = getSta();
  SdcConstrain* the_constrain = ista->getConstrain();

  unsigned is_ok = 0;
  if (_prop_type == PropType::kApplySdcPreProp) {
    auto& the_clocks = the_constrain->get_sdc_clocks();
    auto& the_io_constrain = the_constrain->get_sdc_io_constraints();
    auto& the_ocv_derate = the_constrain->get_sdc_timing_derates();

    is_ok = setupClocks(the_clocks, the_graph);
    is_ok &= setupIOConstrain(the_io_constrain, the_graph);
    is_ok &= setupOcvDerate(the_ocv_derate, the_graph);

  } else if (_prop_type == PropType::kApplySdcPostNormalClockProp) {
    auto& the_clocks = the_constrain->get_sdc_clocks();
    is_ok = setupClocks(the_clocks, the_graph);

  } else if (_prop_type == PropType::kApplySdcPostClockProp) {
    auto& the_sdc_exceptions = the_constrain->get_sdc_exceptions();
    is_ok = setupException(the_sdc_exceptions, the_graph);

  } else {
    auto& the_clock_uncertaintys = the_constrain->get_sdc_clock_uncertaintys();
    auto& the_timing_drcs = the_constrain->get_sdc_timing_drcs();

    is_ok = processClockUncertainty(the_clock_uncertaintys, the_graph);
    is_ok &= setupTimingDrc(the_timing_drcs, the_graph);
  }

  LOG_INFO << "apply sdc end";

  return is_ok;
}

}  // namespace ista
