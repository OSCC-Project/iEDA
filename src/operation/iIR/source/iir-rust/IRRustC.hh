/**
 * @file IRRustC.hh
 * @author simin tao (taosm@pcl.ac.cn)
 * @brief The C wrapper for iIR Rust.
 * @version 0.1
 * @date 2024-04-11
 */
#pragma once

#include "RustCommon.hh"

extern "C" {

typedef struct RustMatrix {
  // val at (row,col)
  double data;
  uintptr_t row;
  uintptr_t col;
} RustMatrix;

typedef struct RustNetConductanceData {
  char *net_name;
  uintptr_t node_num;
  struct RustVec g_matrix_vec;
  // for free Rust ir net memory, record the ptr. 
  const void *ir_net_raw_ptr;
} RustNetConductanceData;

// iterator for access hash map
void *create_hashmap_iterator(void *hashmap);
bool hashmap_iterator_next(void *iterator, uintptr_t *out_key, double *out_value);
void destroy_hashmap_iterator(void *iterator);

void init_iir(void);

const void *read_spef(const char *c_power_net_spef);

const void *create_pg_node(const void *c_pg_netlist, const void *c_pg_node);

const void *create_pg_edge(const void *c_pg_netlist, const void *c_pg_edge);

const void *create_pg_netlist(const char *c_power_net_name);

const void *create_rc_data(const void *c_pg_netlist_ptr, uintptr_t len);

/**
 * Read instance power csv file.
 */
const void *read_inst_pwr_csv(const char *file_path);

/**
 * @brief Set the instance power data, not used the csv file.
 * 
 * @param c_instance_power_data 
 * @return void* 
 */
void *set_instance_power_data(struct RustVec c_instance_power_data);

double get_sum_resistance(const void *c_rc_data, const char *c_net_name);

struct RustNetConductanceData build_one_net_conductance_matrix_data(
    const void *c_rc_data, const char *c_net_name);

/**
 * Build RC matrix and current vector data.
 */
struct RustVec build_matrix_from_raw_data(const char *c_inst_power_path,
                                          const char *c_power_net_spef);

/**
 * Build one net instance current vector.
 */
void *build_one_net_instance_current_vector(const void *c_instance_power_data,
                                            const void *c_rc_data,
                                            const char *c_net_name);

struct RustVec get_bump_node_ids(const void *c_rc_data, const char *c_net_name);
struct RustVec get_instance_node_ids(const void *c_rc_data, const char *c_net_name);

const char *get_instance_name(const void *c_rc_data, const char *c_net_name, uintptr_t node_id);
}

namespace iir {

void BuildMatrixFromRawData(const char *c_inst_power_path,
                            const char *c_power_net_spef);

}