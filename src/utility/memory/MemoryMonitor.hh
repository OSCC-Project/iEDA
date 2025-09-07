#ifndef MEMORY_MONITOR_H
#define MEMORY_MONITOR_H

#include <string>
#include <chrono>
#include <mutex>

class MemoryMonitor {
public:
    MemoryMonitor(const std::string& label, const std::string& logFile = "memory_monitor.log");
    ~MemoryMonitor();
    void report();
    
    static std::string getTimeStamp();
    
private:
    size_t getCurrentRSS();

    size_t getCurrentVirtual();
    
    size_t getPeakRSS();
    
    std::string formatMemory(size_t bytes);
    
    std::string label_;
    std::string logFile_;
    size_t start_memory_;
    size_t start_vm_;
    std::chrono::high_resolution_clock::time_point start_time_;
    static std::mutex file_mutex_; 
};

#endif // MEMORY_MONITOR_H
