#include <condition_variable>
#include <iostream>
#include <thread>
#include <mutex>
#include <string>

#define LOG(x) std::cout << x << "\n";

void work(std::mutex& mtx, std::condition_variable& cv, bool& ready, bool& processed, std::string& data) {
    LOG("Worker thread waiting for data from main thread");
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]{ return ready; });
    LOG("Worker thread received data from main thread");
    data += "[processed]";
    processed = true;
    LOG("Worker thread processed data from main thread");
    lock.unlock();
    cv.notify_one();
}

int main() {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    bool processed = false;
    std::string data;
    
    std::thread worker(work, std::ref(mtx), std::ref(cv), std::ref(ready), std::ref(processed), std::ref(data));
    
    data = "[unprocessed]; ";
    LOG("Main thread sending unprocessed data to worker thread: " << data);
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one();
    LOG("Main thread sent data to worker thread and now waiting for processing");
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return processed; });
    }
    worker.join();
    LOG("Processed data: " << data);
    return 0;
}