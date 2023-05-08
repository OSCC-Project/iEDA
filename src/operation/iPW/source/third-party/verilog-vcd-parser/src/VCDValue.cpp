
#include "VCDValue.hpp"

// void* VCDValue::operator new(size_t size) {
//   auto* page_pool = ipower::MemPool::getPagePool(ipower::kVCDValueType);
//   uint64_t id = 0;
//   auto* vcd_value = page_pool->allocate<VCDValue>(ipower::kVCDValueType, id);
//   return vcd_value;
// }

// void VCDValue::operator delete(void* ptr) {
//   auto* page_pool = ipower::MemPool::getPagePool(ipower::kVCDValueType);
//   page_pool->free<VCDValue>(ipower::kVCDValueType, (VCDValue*)ptr);
// }
