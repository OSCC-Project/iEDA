# NotificationUtility - Universal HTTP Notification System

The NotificationUtility provides a thread-safe way to send HTTP POST notifications to external systems when algorithms complete iterations. It's designed to notify external monitoring systems, dashboards, or webhooks about the progress of long-running algorithms like the DetailedRouter.

## Features

- **Thread-safe**: Can be safely used in multi-threaded environments
- **Asynchronous**: Supports both synchronous and asynchronous notification modes
- **Configurable**: Flexible configuration via environment variables or programmatic setup
- **Retry logic**: Built-in retry mechanism with exponential backoff
- **Authentication**: Support for Bearer token authentication
- **JSON payload**: Structured JSON payload with algorithm metrics
- **Error handling**: Comprehensive error handling and logging

## Configuration

### Environment Variables

- `IEDA_NOTIFICATION_URL`: Target endpoint URL for notifications
- `ID_SECRET`: Authentication token (used as Bearer token)

### Programmatic Configuration

```cpp
#include "NotificationUtility.h"

// Get singleton instance
auto& notifier = ieda::NotificationUtility::getInstance();

// Configure with custom settings
ieda::NotificationUtility::NotificationConfig config;
config.endpoint_url = "https://your-webhook-endpoint.com/notifications";
config.auth_token = "your-bearer-token";
config.timeout_seconds = 30;
config.max_retries = 3;
config.async_mode = true;
config.enable_ssl_verification = true;

notifier.initialize(config);
```

## Usage Examples

### Basic Usage

```cpp
#include "NotificationUtility.h"

// Initialize with environment variables
auto& notifier = ieda::NotificationUtility::getInstance();
notifier.initialize(); // Uses IEDA_NOTIFICATION_URL and ID_SECRET env vars

// Send iteration notification
auto response = notifier.sendIterationNotification(
    "DetailedRouter",    // algorithm name
    5,                   // current iteration
    10,                  // total iterations
    "running",           // status
    {                    // metrics map
        {"violations", "42"},
        {"wire_length", "1234.56"},
        {"elapsed_time", "2.5s"}
    }
);

if (response.success) {
    std::cout << "Notification sent successfully" << std::endl;
} else {
    std::cerr << "Failed to send notification: " << response.error_message << std::endl;
}
```

### Custom Payload

```cpp
#include "NotificationUtility.h"

auto& notifier = ieda::NotificationUtility::getInstance();

// Create custom payload
ieda::NotificationUtility::NotificationPayload payload;
payload.algorithm_name = "MyAlgorithm";
payload.stage = "optimization";
payload.iteration_number = 3;
payload.total_iterations = 20;
payload.status = "running";
payload.metrics["cost"] = "123.45";
payload.metrics["temperature"] = "0.85";
payload.metadata["version"] = "1.0.0";
payload.metadata["config"] = "high_performance";

auto response = notifier.sendNotification(payload);
```

### Integration in Algorithm Code

```cpp
void MyAlgorithm::runIterations() {
    auto& notifier = ieda::NotificationUtility::getInstance();
    
    // Initialize once
    if (!notifier.isInitialized()) {
        notifier.initialize();
    }
    
    for (int iter = 1; iter <= total_iterations; ++iter) {
        // Run algorithm iteration
        performIteration(iter);
        
        // Collect metrics
        std::map<std::string, std::string> metrics;
        metrics["violations"] = std::to_string(getViolationCount());
        metrics["cost"] = std::to_string(getCurrentCost());
        metrics["elapsed_time"] = getElapsedTime();
        
        // Send notification
        if (notifier.isEnabled()) {
            std::string status = (iter == total_iterations) ? "completed" : "running";
            notifier.sendIterationNotification(
                "MyAlgorithm", iter, total_iterations, status, metrics
            );
        }
    }
    
    // Wait for pending async notifications before exit
    notifier.waitForPendingNotifications(5000); // 5 second timeout
}
```

## JSON Payload Format

The notification utility sends JSON payloads with the following structure:

```json
{
    "algorithm_name": "DetailedRouter",
    "stage": "iteration",
    "iteration_number": 5,
    "total_iterations": 10,
    "status": "running",
    "progress_percentage": 50.0,
    "timestamp": "2025-06-28T10:30:45.123Z",
    "metrics": {
        "route_violations": "42",
        "total_wire_length": "1234.56",
        "total_via_num": "789",
        "elapsed_time": "2.5s"
    },
    "metadata": {
        "version": "1.0.0",
        "config": "default"
    }
}
```

## HTTP Headers

The utility automatically sets the following headers:

- `Content-Type`: `application/json` (configurable)
- `Authorization`: `Bearer <token>` (if auth_token is provided)

## Error Handling

The utility provides comprehensive error handling:

- Network timeouts
- HTTP error responses (4xx, 5xx)
- SSL certificate verification failures
- JSON serialization errors
- CURL library errors

All errors are logged and returned in the `HttpResponse` structure.

## Thread Safety

The NotificationUtility is designed to be thread-safe:

- Singleton pattern with mutex protection
- Thread-safe configuration updates
- Async notifications use separate threads
- Proper cleanup of completed futures

## Performance Considerations

- **Async Mode**: Use async mode (`config.async_mode = true`) for better performance in time-critical algorithms
- **Timeout**: Set appropriate timeout values to avoid blocking
- **Retries**: Configure retry count based on network reliability
- **SSL**: Disable SSL verification only in development environments

## Dependencies

- **libcurl**: HTTP client library
- **nlohmann/json**: JSON serialization (included in third_party)
- **C++20**: Modern C++ features

## Building

The notification utility is automatically built when building the iEDA project. Ensure libcurl is installed:

```bash
# Ubuntu/Debian
sudo apt-get install libcurl4-openssl-dev

# CentOS/RHEL
sudo yum install libcurl-devel

# macOS
brew install curl
```

## Webhook Endpoint Requirements

Your webhook endpoint should:

1. Accept HTTP POST requests
2. Handle `application/json` content type
3. Return HTTP 2xx status codes for success
4. Optionally validate the Bearer token
5. Be able to handle the JSON payload structure shown above

Example webhook endpoint (Python Flask):

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/notifications', methods=['POST'])
def handle_notification():
    # Validate authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Process notification
    data = request.get_json()
    print(f"Received notification from {data['algorithm_name']}")
    print(f"Iteration {data['iteration_number']}/{data['total_iterations']}")
    print(f"Status: {data['status']}")
    print(f"Metrics: {data['metrics']}")
    
    return jsonify({'status': 'received'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Troubleshooting

### Common Issues

1. **"Failed to initialize libcurl"**: Ensure libcurl is properly installed
2. **"Endpoint URL is required"**: Set `IEDA_NOTIFICATION_URL` environment variable
3. **SSL verification errors**: Set `config.enable_ssl_verification = false` for self-signed certificates
4. **Timeout errors**: Increase `config.timeout_seconds` value
5. **Authentication failures**: Verify `ID_SECRET` environment variable is set correctly

### Debug Mode

Enable debug logging by setting the log level appropriately in your application.

### Testing

Test your webhook endpoint using curl:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{"algorithm_name":"test","iteration_number":1,"total_iterations":1,"status":"completed"}' \
  https://your-webhook-endpoint.com/notifications
```
