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
#include <thread>

using json = nlohmann::json;

namespace ieda {

// Static member definitions
std::unique_ptr<NotificationUtility> NotificationUtility::_instance = nullptr;
std::mutex NotificationUtility::_instance_mutex;



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
        const char* env_url = std::getenv("IEDA_ECOS_NOTIFICATION_URL");
        if (env_url) {
            config.endpoint_url = env_url;
        }
    }

    // Try to load task context from environment variables
    const char* env_task_id = std::getenv("ECOS_TASK_ID");
    const char* env_project_id = std::getenv("ECOS_PROJECT_ID");
    const char* env_task_type = std::getenv("ECOS_TASK_TYPE");

    if (env_task_id) config.task_id = env_task_id;
    if (env_project_id) config.project_id = env_project_id;
    if (env_task_type) config.task_type = env_task_type;

    return initialize(config);
}

bool NotificationUtility::initialize(const std::string& endpoint_url,
                                    const std::string& task_id,
                                    const std::string& project_id,
                                    const std::string& task_type) {
    NotificationConfig config;
    config.endpoint_url = endpoint_url;
    config.task_id = task_id;
    config.project_id = project_id;
    config.task_type = task_type;

    return initialize(config);
}

NotificationUtility::HttpResponse NotificationUtility::sendNotification(const std::string& tool_name, 
                                                                         const std::map<std::string, std::string>& metadata) {
    NotificationPayload payload;
    payload.tool_name = tool_name;
    payload.metadata = metadata;
    payload.timestamp = getCurrentTimestamp();
    
    return sendNotification(payload);
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
        return response;
    } else {
        return performHttpRequest(payload_json);
    }
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
    
    struct curl_slist* headers = nullptr;
    
    try {
        // Build URL with query parameters for task context
        std::string url = _config.endpoint_url;
        if (!_config.task_id.empty() && !_config.project_id.empty()) {
            char separator = (url.find('?') != std::string::npos) ? '&' : '?';
            url = _config.endpoint_url + separator + "task_id=" + _config.task_id + "&project_id=" + _config.project_id;
            if (!_config.task_type.empty()) {
                url += "&task_type=" + _config.task_type;
            }
        }

        // Set URL
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        
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

    j["tool_name"] = payload.tool_name;
    j["timestamp"] = payload.timestamp;

    // Add metadata
    if (!payload.metadata.empty()) {
        j["metadata"] = payload.metadata;
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
    const char* token = std::getenv("IEDA_ECOS_NOTIFICATION_SECRET");
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

void NotificationUtility::setTaskContext(const std::string& task_id,
                                        const std::string& project_id,
                                        const std::string& task_type) {
    std::lock_guard<std::mutex> lock(_config_mutex);
    _config.task_id = task_id;
    _config.project_id = project_id;
    _config.task_type = task_type;
}

std::tuple<std::string, std::string, std::string> NotificationUtility::getTaskContext() const {
    std::lock_guard<std::mutex> lock(_config_mutex);
    return std::make_tuple(_config.task_id, _config.project_id, _config.task_type);
}

} // namespace ieda
