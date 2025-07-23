#ifndef MEMORY_MONITOR_H
#define MEMORY_MONITOR_H

#include <string>
#include <chrono>
#include <mutex>

class MemoryMonitor {
public:
    // 构造函数：指定标签和输出文件名
    MemoryMonitor(const std::string& label, const std::string& logFile = "memory_monitor.log");
    
    // 析构函数：自动报告内存使用情况
    ~MemoryMonitor();
    
    // 手动报告内存使用情况
    void report();
    
    // 获取当前时间戳
    static std::string getTimeStamp();
    
private:
    // 获取当前进程的物理内存使用量(RSS)，单位为字节
    size_t getCurrentRSS();
    
    // 获取当前进程的虚拟内存使用量，单位为字节
    size_t getCurrentVirtual();
    
    // 获取进程的峰值物理内存使用量
    size_t getPeakRSS();
    
    // 格式化内存大小显示
    std::string formatMemory(size_t bytes);
    
    std::string label_;
    std::string logFile_;
    size_t start_memory_;
    size_t start_vm_;
    std::chrono::high_resolution_clock::time_point start_time_;
    static std::mutex file_mutex_; // 保证多线程安全
};

#endif // MEMORY_MONITOR_H
