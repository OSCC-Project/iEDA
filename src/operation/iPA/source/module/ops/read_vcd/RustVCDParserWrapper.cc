#include "RustVCDParserWrapper.hh"

#include "string/Str.hh"

namespace ipower {

unsigned RustVcdParserWrapper::readVcdFile(const char* vcd_file_path) {
  RustVcdReader vcd_reader;
  _vcd_file_ptr = vcd_reader.readVcdFile(vcd_file_path);
  _vcd_file = rust_convert_vcd_file(_vcd_file_ptr);

  return 1;
}

unsigned RustVcdParserWrapper::buildAnnotateDB(const char* top_instance_name) {
  std::function<RustVCDScope*(RustVCDScope*)> traverse_scope =
      [&traverse_scope,
       &top_instance_name](RustVCDScope* parent_scope) -> RustVCDScope* {
    auto children_scopes = parent_scope->children_scope;
    void* children_scope;
    FOREACH_VEC_ELEM(&children_scopes, void, children_scope) {
      void* children_scope_ptr = rust_convert_rc_ref_cell_scope(children_scope);
      RustVCDScope* cur_vcd_scope = rust_convert_vcd_scope(children_scope_ptr);
      if (ieda::Str::equal(cur_vcd_scope->name, top_instance_name)) {
        return cur_vcd_scope;
      }
    }

    FOREACH_VEC_ELEM(&children_scopes, void, children_scope) {
      void* children_scope_ptr = rust_convert_rc_ref_cell_scope(children_scope);
      RustVCDScope* cur_vcd_scope = rust_convert_vcd_scope(children_scope_ptr);
      auto* found_scope = traverse_scope(cur_vcd_scope);
      if (found_scope) {
        return found_scope;
      }
    }

    return nullptr;
  };

  auto* root_scope = static_cast<RustVCDScope*>(_vcd_file->scope_root);
  RustVCDScope* found_scope = nullptr;

  if (ieda::Str::equal(root_scope->name, top_instance_name)) {
    found_scope = root_scope;
  } else {
    found_scope = traverse_scope(root_scope);
  }
  LOG_FATAL_IF(!found_scope) << "not found the scope " << top_instance_name;

  _top_instance_scope = found_scope;

  // TODO(to shaozheqing),config the annotate simualtion time and time scale.
  /*set simualtion time for annotate database*/
  if (_end_time) {
    // User set timescale end time
    if (_begin_time) {
      // User set timescale begin time
      _annotate_db.set_simulation_duration(_end_time.value() -
                                           _begin_time.value());
    } else {
      _annotate_db.set_simulation_duration(_end_time.value());
    }
  } else {
    // User did not set timescale end time, use the simulation end time
    // FIXME(to shaozheqing) set simulation end time
    int64_t simulation_end_time = _vcd_file->end_time;
    _annotate_db.set_simulation_duration(simulation_end_time);
  }

  /*set timescale for annotate database*/
  auto time_scale = _vcd_file->time_resolution;
  auto scale_unit = _vcd_file->time_unit;
  _annotate_db.set_timescale(time_scale, scale_unit);

  /*build annotate database according the scope*/
  std::function<void(RustVCDScope*, AnnotateInstance*)>
      build_scope_instance_signal = [this, &build_scope_instance_signal](
                                        RustVCDScope* the_scope,
                                        AnnotateInstance* parent_instance) {
        auto the_scope_instance =
            std::make_unique<AnnotateInstance>(the_scope->name);
        auto* the_scope_instance_ptr = the_scope_instance.get();
        if (!parent_instance) {
          _annotate_db.set_top_instance(std::move(the_scope_instance));
        } else {
          parent_instance->addChildInstance(std::move(the_scope_instance));
        }

        // TODO add signal to struct
        auto scope_signals = the_scope->scope_signals;
        void* scope_signal;
        FOREACH_VEC_ELEM(&scope_signals, void, scope_signal) {
          void* signal_ptr = rust_convert_rc_ref_cell_signal(scope_signal);
          RustVCDSignal* signal = rust_convert_vcd_signal(signal_ptr);

          if (signal->signal_type != VCDVariableType::kVarWire) {
            continue;
          }
          if (signal->signal_size == 1) {
            // scalar signal
            auto annotate_signal =
                std::make_unique<AnnotateSignal>(signal->name);
            the_scope_instance_ptr->addSignal(std::move(annotate_signal));
          } else {
            // bus signal
            void* c_bus_index = signal->bus_index;
            Indexes* bus_index = rust_convert_signal_index(c_bus_index);
            int lindex = bus_index->lindex;
            int rindex = bus_index->rindex;
            for (auto i = rindex; i <= lindex; ++i) {
              std::string signal_name(signal->name);
              std::string name = signal_name + "[" + std::to_string(i) + "]";

              auto annotate_signal = std::make_unique<AnnotateSignal>(name);
              the_scope_instance_ptr->addSignal(std::move(annotate_signal));
            }
          }
        }

        auto children_scopes = the_scope->children_scope;
        void* child_scope;
        FOREACH_VEC_ELEM(&children_scopes, void, child_scope) {
          void* children_scope_ptr =
              rust_convert_rc_ref_cell_scope(child_scope);
          RustVCDScope* cur_child_scope =
              rust_convert_vcd_scope(children_scope_ptr);
          build_scope_instance_signal(cur_child_scope, the_scope_instance_ptr);
        }
      };

  build_scope_instance_signal(_top_instance_scope, nullptr);

  return 1;
}

unsigned RustVcdParserWrapper::calcScopeToggleAndSp(
    const char* top_instance_name) {
  RustTcAndSpResVecs* res_vecs =
      rust_calc_scope_tc_sp(top_instance_name, _vcd_file_ptr);
  auto signal_tc_vec = res_vecs->signal_tc_vec;
  auto signal_sp_vec = res_vecs->signal_duration_vec;

  /*set data to annotate db*/
  auto* top_instance = _annotate_db.get_top_instance();
  std::function<void(AnnotateInstance*)> traverse_instance =
      [&traverse_instance, this, &signal_tc_vec,
       &signal_sp_vec](auto* instance) {
        AnnotateSignal* signal;
        FOREACH_SIGNAL(instance, signal) {
          auto& signal_name = signal->get_signal_name();
          auto* record_data = signal->get_record_data();
          // set toggle
          void* signal_tc;
          FOREACH_VEC_ELEM(&signal_tc_vec, void, signal_tc) {
            RustSignalTC* cur_signal_tc = rust_convert_signal_tc(signal_tc);
            auto* cur_name = cur_signal_tc->signal_name;
            if (ieda::Str::equal(signal_name.c_str(), cur_name)) {
              auto cur_tc = cur_signal_tc->signal_tc;
              AnnotateToggle annotate_toggle;
              annotate_toggle.set_TC(cur_tc);
              (*record_data).set_toggle_record(std::move(annotate_toggle));
            }
          }
          // set sp
          void* signal_sp;
          FOREACH_VEC_ELEM(&signal_sp_vec, void, signal_sp) {
            RustSignalDuration* cur_signal_sp =
                rust_convert_signal_duration(signal_sp);
            auto* cur_name = cur_signal_sp->signal_name;
            if (ieda::Str::equal(signal_name.c_str(), cur_name)) {
              AnnotateTime annotate_time(
                  cur_signal_sp->bit_0_duration, cur_signal_sp->bit_1_duration,
                  cur_signal_sp->bit_x_duration, cur_signal_sp->bit_z_duration);
              (*record_data).set_time_record(std::move(annotate_time));
            }
          }
        }

        AnnotateInstance* child_instance;
        FOREACH_CHILD_INSTANCE(instance, child_instance) {
          traverse_instance(child_instance);
        }
      };

  traverse_instance(top_instance);

  return 1;
}
}  // namespace ipower