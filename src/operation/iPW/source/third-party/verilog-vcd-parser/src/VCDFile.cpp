
#include <iostream>

#include "VCDFile.hpp"

//! Instance a new VCD file container.
VCDFile::VCDFile() {}

//! Destructor
VCDFile::~VCDFile() {
  // Delete signals and scopes.

  for (VCDScope* scope : this->scopes) {
    for (VCDSignal* signal : scope->signals) {
      delete signal;
    }

    delete scope;
  }

  // Delete signal values.

  for (auto hash_val = this->val_map.begin(); hash_val != this->val_map.end();
       ++hash_val) {
    delete hash_val->second;
  }
}

/*!
@brief Add a new scope object to the VCD file
*/
void VCDFile::add_scope(VCDScope* s) { this->scopes.push_back(s); }

/*!
@brief Add a new signal object to the VCD file
*/
void VCDFile::add_signal(VCDSignal* s) {
  this->signals.push_back(s);

  // Add a timestream entry
  if (val_map.find(s->hash) == val_map.end()) {
    // Values will be populated later.
    val_map[s->hash] = new VCDSignalValues();
  }
}

/*!
 */
VCDScope* VCDFile::get_scope(VCDScopeName name) {
  for (auto& scope : scopes) {
    if (scope->name == name) {
      return scope;
    }
  }
  return nullptr;
}

/*!
@brief Add a new signal value to the VCD file, tagged by time.
*/
void VCDFile::add_signal_value(VCDTimedValue&& time_val, VCDSignalHash hash) {
  this->val_map[hash]->emplace_back(std::move(time_val));
}

/*!
 */
std::vector<VCDTime>* VCDFile::get_timestamps() { return &this->times; }

/*!
 */
std::vector<VCDScope*>* VCDFile::get_scopes() { return &this->scopes; }

/*!
 */
std::vector<VCDSignal*>* VCDFile::get_signals() { return &this->signals; }

/*!
 */
void VCDFile::add_timestamp(VCDTime time) { this->times.push_back(time); }

/*!
 */
VCDValue* VCDFile::get_signal_value_at(const VCDSignalHash& hash, VCDTime time,
                                       bool erase_prior) {
  auto find = val_map.find(hash);
  if (find == this->val_map.end()) {
    return nullptr;
  }

  VCDSignalValues* vals = find->second;

  if (vals->size() == 0) {
    return nullptr;
  }

  VCDSignalValues::iterator erase_until = vals->begin();

  VCDValue* tr = nullptr;

  for (auto it = vals->begin(); it != vals->end(); ++it) {
    if ((*it).time <= time) {
      erase_until = it;
      tr = &((*it).value);
    } else {
      break;
    }
  }

  if (erase_prior) {
    vals->erase(vals->begin(), erase_until);
  }

  return tr;
}

VCDSignalValues* VCDFile::get_signal_values(VCDSignalHash hash) {
  if (this->val_map.find(hash) == this->val_map.end()) {
    return nullptr;
  }

  return this->val_map[hash];
}
