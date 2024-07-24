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
    void produce(std::unique_ptr<T> item) {
        /*if (m_count == MAX_ITEMS) {
            LOG("Buffer is full, cannot produce more items until some are consumed.")
            return;
        }*/
        std::unique_lock<std::mutex> lock(m_mtx);
        m_cv_producer.wait(lock, [&] { return m_count < MAX_ITEMS; });
        m_items[m_head] = std::move(item);
        m_count++;
        m_head = (m_head + 1) % MAX_ITEMS;
        lock.unlock();
        m_cv_consumer.notify_one();
    }
    
    std::unique_ptr<T> consume() {
        /*if (m_count == 0) {
            LOG("Buffer is empty, cannot consume any more items until more are produced.")
            return nullptr;
        }*/
        std::unique_lock<std::mutex> lock(m_mtx);
        m_cv_consumer.wait(lock, [&] { return m_count > 0; });
        std::unique_ptr<T> consumed = std::move(m_items[m_tail]);
        m_count--;
        m_tail = (m_tail + 1) % MAX_ITEMS;
        lock.unlock();
        m_cv_producer.notify_one();
        return consumed;
    }

private:
    int m_head;
    int m_tail;
    int m_count;
    std::unique_ptr<T> m_items[MAX_ITEMS];
    std::mutex m_mtx;
    std::condition_variable m_cv_producer;
    std::condition_variable m_cv_consumer;
};

class HeapClass {
public:
    HeapClass() : m_size(10) {
        LOG("HeapClass default constructed with size 10!")
        m_data = new int[m_size];
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
        if (m_data != nullptr) {
            LOG("HeapClass destructed!")
            delete[] m_data;
        }
    }
    
private:
    int m_size;
    int* m_data;
};

void use_producer_consumer() {
    /*{
        HeapClass hc1(10); // constructed
        HeapClass hc2(std::move(hc1)); // move constructed
        HeapClass hc3; // default constructed
        hc2 = std::move(hc3); // move assigned
        // hc1, hc2, hc3, all three destructed
    }*/
    
    LOG("Constructing a ProducerConsumer with MAX_ITEMS = 5")
    const int N = 5;
    ProducerConsumer<HeapClass, N> pc;
    
    LOG("Producing 4 items")
    std::thread producer([&] {
        for (int i = 0; i < N - 1; i++) {
            pc.produce(std::make_unique<HeapClass>(HeapClass(i)));
        }
    });
    
    LOG("Consuming 3 items")
    std::thread consumer([&] {
        for (int i = 0; i < N - 2; i++) {
            LOG("Consuming")
            auto consumed = pc.consume();
        }
    });
    
    producer.join();
    consumer.join();
    
    LOG("Production-consumption finished.")
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