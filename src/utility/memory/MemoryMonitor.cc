#include "MemoryMonitor.hh"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <unistd.h>
#include <sstream>

// 初始化静态成员
std::mutex MemoryMonitor::file_mutex_;

MemoryMonitor::MemoryMonitor(const std::string& label, const std::string& logFile) 
    : label_(label), logFile_(logFile) {
    
    start_memory_ = getCurrentRSS();
    start_vm_ = getCurrentVirtual();
    start_time_ = std::chrono::high_resolution_clock::now();
    
    // 打开日志文件（追加模式）
    std::lock_guard<std::mutex> lock(file_mutex_);
    std::ofstream log(logFile_, std::ios::app);
    if (log.is_open()) {
        log << getTimeStamp() << " [START] " << label_ 
            << " - Initial RSS: " << formatMemory(start_memory_)
            << ", VM: " << formatMemory(start_vm_) << std::endl;
        log.close();
    }
}

MemoryMonitor::~MemoryMonitor() {
    report();
}

void MemoryMonitor::report() {
    size_t current_memory = getCurrentRSS();
    size_t current_vm = getCurrentVirtual();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
    
    std::lock_guard<std::mutex> lock(file_mutex_);
    std::ofstream log(logFile_, std::ios::app);
    if (log.is_open()) {
        log << getTimeStamp() << " [REPORT] " << label_ << ":" << std::endl;
        log << "  Duration: " << duration << " ms" << std::endl;
        log << "  RSS Memory before: " << formatMemory(start_memory_) << std::endl;
        log << "  RSS Memory after: " << formatMemory(current_memory) << std::endl;
        log << "  RSS Memory change: " << formatMemory(current_memory - start_memory_) 
            << " (" << (current_memory > start_memory_ ? "+" : "") 
            << std::fixed << std::setprecision(2) 
            << (current_memory > start_memory_ ? 
               ((double)(current_memory - start_memory_) / start_memory_ * 100) : 
               ((double)(start_memory_ - current_memory) / start_memory_ * 100))
            << "%)" << std::endl;
        
        log << "  Virtual Memory before: " << formatMemory(start_vm_) << std::endl;
        log << "  Virtual Memory after: " << formatMemory(current_vm) << std::endl;
        log << "  Virtual Memory change: " << formatMemory(current_vm - start_vm_) 
            << " (" << (current_vm > start_vm_ ? "+" : "") 
            << std::fixed << std::setprecision(2) 
            << (current_vm > start_vm_ ? 
               ((double)(current_vm - start_vm_) / start_vm_ * 100) : 
               ((double)(start_vm_ - current_vm) / start_vm_ * 100))
            << "%)" << std::endl;
        
        log << "  Peak RSS Memory: " << formatMemory(getPeakRSS()) << std::endl;
        log << std::endl;
        log.close();
    }
}

std::string MemoryMonitor::getTimeStamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
        
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

size_t MemoryMonitor::getCurrentRSS() {
    std::ifstream statm("/proc/self/statm");
    long rss = 0;
    if (statm.is_open()) {
        long vm = 0;
        statm >> vm >> rss;
        statm.close();
    }
    return rss * sysconf(_SC_PAGESIZE); // 转换页数为字节数
}

size_t MemoryMonitor::getCurrentVirtual() {
    std::ifstream statm("/proc/self/statm");
    long vm = 0;
    if (statm.is_open()) {
        statm >> vm;
        statm.close();
    }
    return vm * sysconf(_SC_PAGESIZE); // 转换页数为字节数
}

size_t MemoryMonitor::getPeakRSS() {
    std::ifstream status("/proc/self/status");
    std::string line;
    size_t peak_rss = 0;
    
    if (status.is_open()) {
        while (std::getline(status, line)) {
            if (line.compare(0, 10, "VmHWM:") == 0) {
                std::istringstream iss(line.substr(10));
                size_t kb;
                iss >> kb;
                peak_rss = kb * 1024; // 转换KB为字节
                break;
            }
        }
        status.close();
    }
    return peak_rss;
}

std::string MemoryMonitor::formatMemory(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024 && unit < 4) {
        size /= 1024;
        unit++;
    }
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << units[unit] 
       << " (" << bytes << " bytes)";
    return ss.str();
}
