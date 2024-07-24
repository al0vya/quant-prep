#include <queue>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>

#define LOG(x) std::cout << x << "\n";

void work(std::mutex& mtx, std::condition_variable& cv, bool& ready, bool& processed, std::string& data) {
    LOG("Worker thread waiting for data from main thread");
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&]{ return ready; });
    LOG("Worker thread working...");
    std::this_thread::sleep_for(std::chrono::seconds(5));
    data += "[processed]";
    processed = true;
    LOG("Worker thread processed data from main thread");
    lock.unlock();
    cv.notify_one();
}

void send_data() {
    std::mutex mtx;
    std::condition_variable cv;
    bool ready = false;
    bool processed = false;
    std::string data;
    
    std::thread worker(work, std::ref(mtx), std::ref(cv), std::ref(ready), std::ref(processed), std::ref(data));
    
    LOG("Main thread working...");
    std::this_thread::sleep_for(std::chrono::seconds(5));
    data = "[unprocessed]; ";
    LOG("Main thread sending unprocessed data to worker thread: " << data);
    {
        std::lock_guard<std::mutex> lock(mtx);
        ready = true;
    }
    cv.notify_one();
    LOG("Main thread waiting for worker thread...");
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]{ return processed; });
    }
    worker.join();
    LOG("Processed data: " << data);
}

void wait_for_assignment() {
    int value = 100;
    bool notified = false;
    std::mutex mtx;
    std::condition_variable cv;
    
    std::thread assigner([&] {
        LOG("Assigner thread is working, current value is " << value);
        std::this_thread::sleep_for(std::chrono::seconds(5));
        value = 20;
        notified = true;
        cv.notify_one();
        LOG("Assigner thread has notified the worker thread.");
    });
    
    std::thread reporter([&] {
        LOG("Reporter thread waiting to be notified.");
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return notified; } );
        LOG("Reporter thread has been notified: new value is " << value);
    });
    
    assigner.join();
    reporter.join();
}

void produce_and_consume() {
    int count = 0;
    int N = 500;
    bool finished = false;
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<int> items;
    
    std::thread producer([&] {
        for (int i = 0; i < N; i++) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                items.push(i);
                count++;
            }
            cv.notify_one();
        }
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            finished = true;
        }
        cv.notify_one();
    });
    
    std::thread consumer([&] {
        while (true) {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock,[&] { return !items.empty() || finished; });
            
            if (items.empty() && finished) {
                break;
            }
            
            while (!items.empty()) {
                items.pop();
                count--;
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    LOG("Final count: " << count);
}

int main() {
    //send_data();
    //wait_for_assignment();
    produce_and_consume();
    return 0;
}