#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <atomic>

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

int main() {
    oneThread();
    sumOfSquares();
    return 0;
}