#include <queue>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>

#define LOG(x) std::cout << x << "\n";

void func(int x) {
    std::cout << "Value for thread is " << x << "\n";
}

void oneThread() {
    std::thread t(func, 100);
    t.join();
}

void square(const int x, int& total, std::mutex& mtx) {
    int tmp = x * x;
    std::cout << "tmp: " << tmp << "\n";
    std::lock_guard<std::mutex> lock(mtx);
    total += tmp;
}

void sumOfSquares() {
    int total = 0;
    std::vector<std::thread> threads;
    int N = 20;
    std::mutex mtx;
    for (int i = 0; i < N; i++) {
        threads.push_back(std::thread(square, i, std::ref(total), std::ref(mtx)));
    }
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "Total is " << total << "\n";
}

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
            
            while (!items.empty()) {
                items.pop();
                count--;
            }
            
            if (items.empty() && finished) {
                break;
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    LOG("Final count: " << count);
}

template<typename T, const int MAX_ITEMS>
class ProducerConsumer {
public:
    ProducerConsumer() : m_head(0), m_tail(0), m_count(0) {}
    
    // [. . x x .] -> produce(x) -> [. . x x x]
    //      ^   ^                    ^   ^
    //      t   h                    h   t
    void produce(T item) {
        if (m_count == MAX_ITEMS) {
            return;
        }
        
        m_items[head] = item;
        m_count++;
        m_head = (m_head + 1) % MAX_ITEMS;
    }
    
    T consume() {
        if (m_count == 0) {
            return T();
        }
        
        T popped = m_items[m_tail];
        m_count--;
        m_tail = (m_tail + 1) % MAX_ITEMS;
        return popped;
    }

private:
    int m_head;
    int m_tail;
    int m_count;
    T m_items[MAX_ITEMS];
};

class HeapClass {
public:
    HeapClass() : m_size(0) {
        LOG("HeapClass default constructed!")
        m_data = nullptr;
    }
    
    HeapClass(int size) : m_size(size) {
        LOG("HeapClass constructed!")
        m_data = new int[m_size];
    }
    
    HeapClass(const HeapClass&) = delete;
    
    HeapClass(HeapClass&& other) {
        LOG("HeapClass move constructed!")
        m_size = other.m_size;
        m_data = other.m_data;
        other.m_size = 0;
        other.m_data = nullptr;
    }
    
    HeapClass& operator=(HeapClass&& other) {
        if (this != &other) {
            LOG("HeapClass move assigned!")
            delete[] m_data;
            m_size = other.m_size;
            m_data = other.m_data;
            other.m_size = 0;
            other.m_data = nullptr;
        }
        
        return *this;
    }
    
    ~HeapClass() {
        LOG("HeapClass destructed!")
        delete[] m_data;
    }
    
private:
    int m_size;
    int* m_data;
};

void use_producer_consumer() {
    ProducerConsumer<int, 5> pc;
    HeapClass hc1(10); // constructed
    HeapClass hc2(std::move(hc1)); // move constructed
    HeapClass hc3; // default constructed
    hc2 = std::move(hc3); // move assigned
    // hc1, hc2, hc3, all three destructed
}

int main() {
    //oneThread();
    //sumOfSquares();
    //send_data();
    //wait_for_assignment();
    //produce_and_consume();
    use_producer_consumer();
    return 0;
}