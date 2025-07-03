// ***************************************************************************************
// Copyright (c) 2023-2025 Peng Cheng Laboratory
// Copyright (c) 2023-2025 Institute of Computing Technology, Chinese Academy of Sciences
// Copyright (c) 2023-2025 Beijing Institute of Open Source Chip
//
// iEDA is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
// http://license.coscl.org.cn/MulanPSL2
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
//
// See the Mulan PSL v2 for more details.
// ***************************************************************************************
#include "NotificationUtility.h"

#include <curl/curl.h>
#include <json/json.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>

using json = nlohmann::json;

namespace ieda {

// Static member definitions
std::unique_ptr<NotificationUtility> NotificationUtility::_instance = nullptr;
std::mutex NotificationUtility::_instance_mutex;

// Callback function for libcurl to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t total_size = size * nmemb;
    userp->append(static_cast<char*>(contents), total_size);
    return total_size;
}

NotificationUtility& NotificationUtility::getInstance() {
    std::lock_guard<std::mutex> lock(_instance_mutex);
    if (!_instance) {
        _instance = std::unique_ptr<NotificationUtility>(new NotificationUtility());
    }
    return *_instance;
}

void NotificationUtility::destroyInstance() {
    std::lock_guard<std::mutex> lock(_instance_mutex);
    if (_instance) {
        _instance->waitForPendingNotifications(5000); // Wait up to 5 seconds
        _instance.reset();
    }
}

NotificationUtility::~NotificationUtility() {
    waitForPendingNotifications(5000);
    curl_global_cleanup();
}

bool NotificationUtility::initialize(const NotificationConfig& config) {
    std::lock_guard<std::mutex> lock(_config_mutex);
    
    // Initialize libcurl globally
    CURLcode curl_init_result = curl_global_init(CURL_GLOBAL_DEFAULT);
    if (curl_init_result != CURLE_OK) {
        std::cerr << "NotificationUtility: Failed to initialize libcurl: " 
                  << curl_easy_strerror(curl_init_result) << std::endl;
        return false;
    }
    
    _config = config;
    
    // Load auth token from environment if not provided
    if (_config.auth_token.empty()) {
        _config.auth_token = loadAuthTokenFromEnv();
    }
    
    _initialized = !_config.endpoint_url.empty();
    _enabled = _initialized;
    
    if (!_initialized) {
        std::cerr << "NotificationUtility: Endpoint URL is required for initialization" << std::endl;
    }
    
    return _initialized;
}

bool NotificationUtility::initialize(const std::string& endpoint_url) {
    NotificationConfig config;
    
    // Use provided URL or try to get from environment
    if (!endpoint_url.empty()) {
        config.endpoint_url = endpoint_url;
    } else {
        const char* env_url = std::getenv("IEDA_NOTIFICATION_URL");
        if (env_url) {
            config.endpoint_url = env_url;
        }
    }
    
    return initialize(config);
}

NotificationUtility::HttpResponse NotificationUtility::sendNotification(const NotificationPayload& payload) {
    if (!isEnabled() || !isInitialized()) {
        HttpResponse response;
        response.error_message = "NotificationUtility not initialized or disabled";
        return response;
    }
    
    std::string payload_json = payloadToJson(payload);
    
    if (_config.async_mode) {
        // Launch async notification
        std::lock_guard<std::mutex> lock(_futures_mutex);
        _pending_futures.emplace_back(
            std::async(std::launch::async, &NotificationUtility::asyncNotificationWorker, this, payload_json)
        );
        
        // Clean up completed futures
        _pending_futures.erase(
            std::remove_if(_pending_futures.begin(), _pending_futures.end(),
                [](const std::future<void>& f) {
                    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                }),
            _pending_futures.end()
        );
        
        HttpResponse response;
        response.success = true;
        response.response_body = "Async notification queued";
        return response;
    } else {
        return performHttpRequest(payload_json);
    }
}

NotificationUtility::HttpResponse NotificationUtility::sendIterationNotification(
    const std::string& algorithm_name,
    int iteration,
    int total_iterations,
    const std::string& status,
    const std::map<std::string, std::string>& metrics) {
    
    NotificationPayload payload;
    payload.algorithm_name = algorithm_name;
    payload.stage = "iteration";
    payload.iteration_number = iteration;
    payload.total_iterations = total_iterations;
    payload.status = status;
    payload.metrics = metrics;
    payload.timestamp = getCurrentTimestamp();
    
    return sendNotification(payload);
}

bool NotificationUtility::isInitialized() const {
    std::lock_guard<std::mutex> lock(_config_mutex);
    return _initialized;
}

void NotificationUtility::updateConfig(const NotificationConfig& config) {
    std::lock_guard<std::mutex> lock(_config_mutex);
    _config = config;
}

const NotificationUtility::NotificationConfig& NotificationUtility::getConfig() const {
    std::lock_guard<std::mutex> lock(_config_mutex);
    return _config;
}

void NotificationUtility::setEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(_config_mutex);
    _enabled = enabled;
}

bool NotificationUtility::isEnabled() const {
    std::lock_guard<std::mutex> lock(_config_mutex);
    return _enabled;
}

bool NotificationUtility::waitForPendingNotifications(int timeout_ms) {
    std::lock_guard<std::mutex> lock(_futures_mutex);
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (auto& future : _pending_futures) {
        if (timeout_ms > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();
            
            if (elapsed >= timeout_ms) {
                return false; // Timeout reached
            }
            
            auto remaining_time = std::chrono::milliseconds(timeout_ms - elapsed);
            if (future.wait_for(remaining_time) != std::future_status::ready) {
                return false; // Timeout on this future
            }
        } else {
            future.wait(); // Wait indefinitely
        }
    }
    
    _pending_futures.clear();
    return true;
}

NotificationUtility::HttpResponse NotificationUtility::performHttpRequest(const std::string& payload_json) {
    HttpResponse response;
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        response.error_message = "Failed to initialize CURL";
        return response;
    }
    
    std::string response_body;
    struct curl_slist* headers = nullptr;
    
    try {
        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL, _config.endpoint_url.c_str());
        
        // Set POST method
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        
        // Set payload
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_json.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, payload_json.length());
        
        // Set headers
        std::string content_type_header = "Content-Type: " + _config.content_type;
        headers = curl_slist_append(headers, content_type_header.c_str());
        
        if (!_config.auth_token.empty()) {
            std::string auth_header = "Authorization: Bearer " + _config.auth_token;
            headers = curl_slist_append(headers, auth_header.c_str());
        }
        
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        
        // Set response callback
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
        
        // Set timeout
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, _config.timeout_seconds);
        
        // SSL verification
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, _config.enable_ssl_verification ? 1L : 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, _config.enable_ssl_verification ? 2L : 0L);
        
        // Perform request with retries
        CURLcode res = CURLE_FAILED_INIT;
        for (int retry = 0; retry <= _config.max_retries; ++retry) {
            res = curl_easy_perform(curl);
            if (res == CURLE_OK) {
                break;
            }
            
            if (retry < _config.max_retries) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (retry + 1))); // Exponential backoff
            }
        }
        
        if (res == CURLE_OK) {
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.response_code);
            response.response_body = response_body;
            response.success = (response.response_code >= 200 && response.response_code < 300);
            
            if (!response.success) {
                response.error_message = "HTTP error: " + std::to_string(response.response_code);
            }
        } else {
            response.error_message = "CURL error: " + std::string(curl_easy_strerror(res));
        }
        
    } catch (const std::exception& e) {
        response.error_message = "Exception during HTTP request: " + std::string(e.what());
    }
    
    // Cleanup
    if (headers) {
        curl_slist_free_all(headers);
    }
    curl_easy_cleanup(curl);
    
    return response;
}

std::string NotificationUtility::payloadToJson(const NotificationPayload& payload) {
    json j;

    j["algorithm_name"] = payload.algorithm_name;
    j["stage"] = payload.stage;
    j["iteration_number"] = payload.iteration_number;
    j["total_iterations"] = payload.total_iterations;
    j["status"] = payload.status;
    j["timestamp"] = payload.timestamp;

    // Add metrics
    if (!payload.metrics.empty()) {
        j["metrics"] = payload.metrics;
    }

    // Add metadata
    if (!payload.metadata.empty()) {
        j["metadata"] = payload.metadata;
    }

    // Add progress percentage
    if (payload.total_iterations > 0) {
        double progress = static_cast<double>(payload.iteration_number) / payload.total_iterations * 100.0;
        j["progress_percentage"] = progress;
    }

    return j.dump();
}

std::string NotificationUtility::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';

    return ss.str();
}

std::string NotificationUtility::loadAuthTokenFromEnv() {
    const char* token = std::getenv("ID_SECRET");
    return token ? std::string(token) : std::string();
}

void NotificationUtility::asyncNotificationWorker(const std::string& payload_json) {
    try {
        HttpResponse response = performHttpRequest(payload_json);

        // Log result (optional - could be made configurable)
        if (!response.success) {
            std::cerr << "NotificationUtility: Async notification failed: "
                      << response.error_message << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "NotificationUtility: Exception in async worker: "
                  << e.what() << std::endl;
    }
}

} // namespace ieda
