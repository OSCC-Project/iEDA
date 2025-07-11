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
#pragma once

#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <future>
#include <functional>
#include <tuple>

namespace ieda {

/**
 * @brief Universal notification utility for sending HTTP requests to external systems
 * 
 * This utility provides a thread-safe way to send HTTP POST notifications to configurable
 * endpoints when algorithms complete iterations. It supports custom payload data, authentication,
 * and both synchronous and asynchronous operation modes.
 */
class NotificationUtility {
public:
    /**
     * @brief Response structure for HTTP requests
     */
    struct HttpResponse {
        long response_code;
        std::string error_message;
        bool success;

        HttpResponse() : response_code(0), success(false) {}
    };

    /**
     * @brief Configuration structure for notification settings
     */
    struct NotificationConfig {
        std::string endpoint_url;           // Target URL for notifications
        std::string auth_token;             // Authentication token (from IEDA_ECOS_NOTIFICATION_SECRET env var)
        std::string content_type;           // Content-Type header (default: application/json)
        int timeout_seconds;                // Request timeout in seconds
        int max_retries;                    // Maximum number of retry attempts
        bool enable_ssl_verification;       // Enable SSL certificate verification
        bool async_mode;                    // Send notifications asynchronously
        std::string task_id;                // Task ID for cloud integration
        std::string project_id;             // Project ID for cloud integration
        std::string task_type;              // Task type for cloud integration

        NotificationConfig()
            : content_type("application/json")
            , timeout_seconds(30)
            , max_retries(3)
            , enable_ssl_verification(true)
            , async_mode(true) {}
    };

    /**
     * @brief Generic notification payload data structure
     */
    struct NotificationPayload {
        std::string tool_name;              // Name of the tool/algorithm
        std::map<std::string, std::string> metadata; // Generic metadata map
        std::string timestamp;              // ISO 8601 timestamp
        
        NotificationPayload() = default;

    };

public:
    /**
     * @brief Get singleton instance of NotificationUtility
     * @return Reference to the singleton instance
     */
    static NotificationUtility& getInstance();

    /**
     * @brief Destroy singleton instance
     */
    static void destroyInstance();

    /**
     * @brief Initialize the notification utility with configuration
     * @param config Notification configuration
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const NotificationConfig& config);

    /**
     * @brief Initialize with environment variables and default settings
     * @param endpoint_url Target URL for notifications
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const std::string& endpoint_url = "");

    /**
     * @brief Initialize with task context for cloud integration
     * @param endpoint_url Target URL for notifications
     * @param task_id Task ID for cloud integration
     * @param project_id Project ID for cloud integration
     * @param task_type Task type for cloud integration
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const std::string& endpoint_url,
                   const std::string& task_id,
                   const std::string& project_id,
                   const std::string& task_type = "");

    /**
     * @brief Send generic notification with tool name and metadata
     * @param tool_name Name of the tool/algorithm
     * @param metadata Optional metadata map
     * @return HttpResponse containing result (for sync mode) or empty response (for async mode)
     */
    HttpResponse sendNotification(const std::string& tool_name, 
                                 const std::map<std::string, std::string>& metadata = {});

    /**
     * @brief Send notification with custom payload
     * @param payload Notification payload data
     * @return HttpResponse containing result (for sync mode) or empty response (for async mode)
     */
    HttpResponse sendNotification(const NotificationPayload& payload);



    /**
     * @brief Check if notification utility is properly initialized
     * @return true if initialized, false otherwise
     */
    bool isInitialized() const;

    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void updateConfig(const NotificationConfig& config);

    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const NotificationConfig& getConfig() const;

    /**
     * @brief Enable or disable notifications
     * @param enabled true to enable, false to disable
     */
    void setEnabled(bool enabled);

    /**
     * @brief Check if notifications are enabled
     * @return true if enabled, false otherwise
     */
    bool isEnabled() const;

    /**
     * @brief Wait for all pending async notifications to complete
     * @param timeout_ms Maximum time to wait in milliseconds (0 = wait indefinitely)
     * @return true if all notifications completed, false if timeout
     */
    bool waitForPendingNotifications(int timeout_ms = 0);

    /**
     * @brief Set task context for cloud integration
     * @param task_id Task ID
     * @param project_id Project ID
     * @param task_type Task type (optional)
     */
    void setTaskContext(const std::string& task_id,
                       const std::string& project_id,
                       const std::string& task_type = "");

    /**
     * @brief Get current task context
     * @return Tuple of (task_id, project_id, task_type)
     */
    std::tuple<std::string, std::string, std::string> getTaskContext() const;

    ~NotificationUtility();

private:
    NotificationUtility() = default;

    // Disable copy constructor and assignment operator
    NotificationUtility(const NotificationUtility&) = delete;
    NotificationUtility& operator=(const NotificationUtility&) = delete;

    /**
     * @brief Perform the actual HTTP POST request
     * @param payload_json JSON payload string
     * @return HttpResponse containing result
     */
    HttpResponse performHttpRequest(const std::string& payload_json);

    /**
     * @brief Convert payload to JSON string
     * @param payload Notification payload
     * @return JSON string representation
     */
    std::string payloadToJson(const NotificationPayload& payload);

    /**
     * @brief Get current timestamp in ISO 8601 format
     * @return Timestamp string
     */
    std::string getCurrentTimestamp();

    /**
     * @brief Load authentication token from environment variable
     * @return Authentication token or empty string if not found
     */
    std::string loadAuthTokenFromEnv();

    /**
     * @brief Async notification worker function
     * @param payload_json JSON payload to send
     */
    void asyncNotificationWorker(const std::string& payload_json);

private:
    static std::unique_ptr<NotificationUtility> _instance;
    static std::mutex _instance_mutex;

    mutable std::mutex _config_mutex;
    NotificationConfig _config;
    bool _initialized;
    bool _enabled;
    
    // For async operations
    std::vector<std::future<void>> _pending_futures;
    mutable std::mutex _futures_mutex;
};

} // namespace ieda
