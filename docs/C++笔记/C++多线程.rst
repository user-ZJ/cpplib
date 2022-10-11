C++多线程
==============

pthread(POSIX标准)
-----------------------

.. code-block:: cpp

    #include <pthread.h>
    // 创建线程。成功时返回0，失败是返回其他值
    // thread 保存新创建线程ID的变量地址值
    // attr 用于传递线程属性的参数，传递NULL时，创建默认属性的线程
    // start_routine  线程的入口函数地址值（函数指针）
    // arg 通过第三个参数传递调用函数时包含传递参数信息的变量地址值
    int pthread_create(pthread_t *restrict thread,const pthread_attr_t * restrict attr,
                    void *(*start_routine)(void *),void *restrict arg);
    // 线程执行完成后返回。成功时返回0，失败是返回其他值
    // thread  线程ID
    // status 保存线程的执行函数返回值的指针变量地址
    int pthread_join(pthread_t thread,void **status);


C++11多线程
----------------------

* feature:一个句柄，通过它你可以从一个共享的单对象缓冲区中 get()一个值，可能需要等待某个 promise 将该值放入缓冲区。
* promise:一个句柄，通过它你可以将一个值 put() 到一个共享的单对象缓冲区，可能会唤醒某个等待 future 的 thread 。
* packaged_task:一个类，它使得设置一个函数在线程上异步执行变得容易，由 future 来接受 promise 返回的结果。
* async():一个函数，可以启动一个任务并在另一个 thread 上执行。async将代码包装在 packaged_task 中，并管理 future 及其传输结果的 promise的设置。

示例
```````````
.. code-block:: cpp

    #include <iostream>
    #include <thread>
    std::thread::id main_thread_id = std::this_thread::get_id();
    void hello()  
    {
        std::cout << "Hello Concurrent World\n";
        if (main_thread_id == std::this_thread::get_id())
            std::cout << "This is the main thread.\n";
        else
            std::cout << "This is not the main thread.\n";
    }
    void pause_thread(int n) {
        std::this_thread::sleep_for(std::chrono::seconds(n));
        std::cout << "pause of " << n << " seconds ended\n";
    }
    int main() {
        std::thread t(hello);
        std::cout << t.hardware_concurrency() << std::endl;//可以并发执行多少个(不准确)
        std::cout << "native_handle " << t.native_handle() << std::endl;//可以并发执行多少个(不准确)
        t.join();
        std::thread a(hello);
        a.detach();
        std::thread threads[5];                         // 默认构造线程

        std::cout << "Spawning 5 threads...\n";
        for (int i = 0; i < 5; ++i)
            threads[i] = std::thread(pause_thread, i + 1);   // move-assign threads
        std::cout << "Done spawning threads. Now waiting for them to join:\n";
        for (auto &thread : threads)
            thread.join();
        std::cout << "All threads joined!\n";
    }

.. code-block:: shell

    g++ -std=c++11 test.cpp -lpthread


C++11创建线程的三种方式
```````````````````````````
::

    线程创建

    std::thread thObj(<CALLBACK>);

    其中callback接收 **函数指针** ， **函数对象** 和 **lambda函数**

1. 函数指针

.. code-block:: cpp

    #include <iostream>
    #include <thread>

    void thread_function()
    {
        for(int i = 0; i < 10000; i++);
            std::cout<<"thread function Executing"<<std::endl;
    }
    int main()  
    {

        std::thread threadObj(thread_function);
        for(int i = 0; i < 10000; i++);
            std::cout<<"Display From MainThread"<<std::endl;
        threadObj.join();    
        std::cout<<"Exit of Main function"<<std::endl;
        return 0;
    }

.. note:: 

    使用 **成员函数** 创建线程，需要传递类的一个对象作为参数

    .. code-block:: cpp

        #include <thread>
        #include <iostream>

        class bar {
        public:
        void foo(int x) {
            std::cout << "hello from member function" << std::endl;
        }
        };

        int main()
        {
            bar obj;
            std::thread t(&bar::foo, obj,0);
            t.join();
        }

    如果是在类的成员函数中处理thread，传入 this 即可，如：

    .. code-block:: cpp

        std::thread spawn() {
            return std::thread(&blub::test, this);
        }

2. 函数对象

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    class DisplayThread
    {
    public:
        void operator(int N)()     
        {
            for(int i = 0; i < N; i++)
                std::cout<<"Display Thread Executing"<<std::endl;
        }
    };

    int main()  
    {
        std::thread threadObj( (DisplayThread()),10000 );
        for(int i = 0; i < 10000; i++)
            std::cout<<"Display From Main Thread "<<std::endl;
        std::cout<<"Waiting For Thread to complete"<<std::endl;
        threadObj.join();
        std::cout<<"Exiting from Main Thread"<<std::endl;
        return 0;
    }

3. lambda函数

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    int main()  
    {
        std::thread threadObj([](int N){
                for(int i = 0; i < N; i++)
                    std::cout<<"Display Thread Executing"<<std::endl;
                },10000);

        for(int i = 0; i < 10000; i++)
            std::cout<<"Display From Main Thread"<<std::endl;

        threadObj.join();
        std::cout<<"Exiting from Main Thread"<<std::endl;
        return 0;
    }


获取线程ID
```````````````````
.. code-block:: cpp

    //通过线程对象获取线程ID
    std::thread::get_id();
    //在线程内部获取线程id
    std::this_thread::get_id();

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    void thread_function()
    {
        std::cout<<"Inside Thread :: ID  = "<<std::this_thread::get_id()<<std::endl;    
    }
    int main()  
    {
        std::thread threadObj1(thread_function);
        std::thread threadObj2(thread_function);

        if(threadObj1.get_id() != threadObj2.get_id())
            std::cout<<"Both Threads have different IDs"<<std::endl;

            std::cout<<"From Main Thread :: ID of Thread 1 = "<<threadObj1.get_id()<<std::endl;    
        std::cout<<"From Main Thread :: ID of Thread 2 = "<<threadObj2.get_id()<<std::endl;    

        threadObj1.join();    
        threadObj2.join();    
        return 0;
    }


join和detach
```````````````````
* join：等待线程执行结束，再执行join后的代码
* detach：分离的线程也称为守护程序/后台线程。调用detach()之后，std::thread对象不再与实际的执行线程关联。

在std::thread的析构函数中，如果std::thread对象如果还处于joinable的状态，那么会调用std::terminate()立刻退出这个程序。
如果主线程还有代码没有执行完则会导致程序异常退出。

ps：join和detach会将std::thread对象状态置为unjoinable的状态

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    void thread_function()
    {
        std::cout<<"Inside Thread :: ID  = "<<std::this_thread::get_id()<<std::endl;    
    }
    int main()  
    {
        {
            std::thread threadObj1(thread_function);
            std::cout<<"From Main Thread :: ID of Thread 1 = "<<threadObj1.get_id()<<std::endl;      
        }
        //执行报错，在退出{}作用域时会销毁threadObj1对象，发现threadObj1.joinable()为true
        //所以调用std::terminate()来终止程序。
        std::cout<<"Main Thread exit !!!"<<std::endl; 
        return 0;
    }


当程序终止（即`main`返回）时，不会等待在后台执行的其余detach的线程；
相反，它们的执行被挂起，并且它们的线程本地对象被破坏。这意味着 *不会解开那些线程的堆栈，* 因此不会执行某些析构函数。
相当程序崩溃或被kill一样；操作系统会释放文件等的锁定，但是可能损坏共享内存，或者有文件写到一半等操作。

使用RESOURCE ACQUISITION IS INITIALIZATION (RAII)可以防止忘记调用join或detach

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    class ThreadRAII
    {
        std::thread & m_thread;
        public:
            ThreadRAII(std::thread  & threadObj) : m_thread(threadObj)
            {

            }
            ~ThreadRAII()
            {
                // Check if thread is joinable then detach the thread
                if(m_thread.joinable())
                {
                    m_thread.detach();
                }
            }
    };
    void thread_function()
    {
        for(int i = 0; i < 10000; i++);
            std::cout<<"thread_function Executing"<<std::endl;
    }

    int main()  
    {
        std::thread threadObj(thread_function);

        // If we comment this Line, then program will crash
        ThreadRAII wrapperObj(threadObj);
        return 0;
    }


参数传递
```````````````

默认情况下，所有参数都复制到新线程的内部存储中

普通参数传递
:::::::::::::::

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <thread>
    void threadCallback(int x, std::string str)
    {
        std::cout<<"Passed Number = "<<x<<std::endl;
        std::cout<<"Passed String = "<<str<<std::endl;
    }
    int main()  
    {
        int x = 10;
        std::string str = "Sample String";
        std::thread threadObj(threadCallback, x, str);
        threadObj.join();
        return 0;
    }

.. note:: 
    
  1. 不要将变量的地址从本地堆栈传递到线程的回调函数。
     因为线程1中的局部变量可能超出作用范围，但线程2仍在尝试通过其地址访问它。在这种情况下，访问无效地址可能会导致意外行为。
  2. 将堆指针传递给线程时要小心。因为某些线程可能会在新线程尝试访问该内存之前删除该内存。在这种情况下，访问无效地址可能会导致意外行为。

引用传递(std::ref)
::::::::::::::::::::::::

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    void threadCallback(int const & x)
    {
        int & y = const_cast<int &>(x);
        y++;
        std::cout<<"Inside Thread x = "<<x<<std::endl;
    }
    int main()
    {
        int x = 9;
        std::cout<<"In Main Thread : Before Thread Start x = "<<x<<std::endl;
        std::thread threadObj(threadCallback,std::ref(x));
        threadObj.join();
        std::cout<<"In Main Thread : After Thread Joins x = "<<x<<std::endl;
        return 0;
    }

获取线程返回值
```````````````````````
**std::future**，是一个类模板，它存储着一个未来的值。
一个 **std::future** 对象里存储着一个在未来会被赋值的变量，
这个变量可以通过 **std::future** 提供的成员函数 **std::future::get()** 来得到。
如果在这个变量被赋值之前就有别的线程试图通过 **std::future::get()** 获取这个变量，那么这个线程将会被阻塞到这个变量可以获取为止

**std::promise**同样也是一个类模板，它的对象 **承诺** 会在未来设置变量(这个变量也就是**std::future**中的变量)。
每一个 **std::promise** 对象都有一个与之关联的 **std::future** 对象。
当 **std::promise** 设置值的时候，这个值就会赋给 **std::future** 中的对象了。

.. code-block:: cpp

    #include<iostream>    //std::cout std::endl
    #include<thread>      //std::thread
    #include<future>      //std::future std::promise
    #include<utility>     //std::ref
    #include<chrono>      //std::chrono::seconds

    void initiazer(std::promise<int> &promiseObj){
        std::cout << "Inside thread: " << std::this_thread::get_id() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        promiseObj.set_value(35);
    }

    int main(){
        std::promise<int> promiseObj;
        std::future<int> futureObj = promiseObj.get_future();
        std::thread th(initiazer, std::ref(promiseObj));

        std::cout << futureObj.get() << std::endl;

        th.join();
        return 0;
    }

std::async
```````````````
**std::async()** 是一个函数模板，接收callback(函数，函数对象，lambda函数)作为参数， 有可能异步执行callback

.. code-block:: cpp

    template <class Fn, class... Args>
    future<typename result_of<Fn(Args...)>::type> async (launch policy, Fn&& fn, Args&&... args);

**std::async** 返回  **std::future<T>,** 存储  **std::async()** 执行的函数的返回值. 函数参数接在函数后面

policy：控制std::async的行为，包括：

* **std::launch::async**：它保证了异步行为，即传递的函数将在单独的线程中执行
* **std :: launch :: deferred**：非异步行为，即当其他线程将来调用get()以访问共享状态时，将调用Function
* **std :: launch :: async | std :: launch :: deferred**:它是默认行为。使用此启动策略，
  它可以异步运行或不异步运行，具体取决于系统上的负载。但是我们无法控制它

.. code-block:: cpp

    #include <iostream>
    #include <string>
    #include <chrono>
    #include <thread>
    #include <future>
    using namespace std::chrono;
    std::string fetchDataFromDB(std::string recvdData)
    {
        // Make sure that function takes 5 seconds to complete
        std::this_thread::sleep_for(seconds(5));
        //Do stuff like creating DB Connection and fetching Data
        return "DB_" + recvdData;
    }
    std::string fetchDataFromFile(std::string recvdData)
    {
        // Make sure that function takes 5 seconds to complete
        std::this_thread::sleep_for(seconds(5));
        //Do stuff like fetching Data File
        return "File_" + recvdData;
    }
    int main()
    {
        // Get Start Time
        system_clock::time_point start = system_clock::now();
        std::future<std::string> resultFromDB = std::async(std::launch::async, fetchDataFromDB, "Data");
        //Fetch Data from File
        std::string fileData = fetchDataFromFile("Data");
        //Fetch Data from DB
        // Will block till data is available in future<std::string> object.
        std::string dbData = resultFromDB.get();
        // Get End Time
        auto end = system_clock::now();
        auto diff = duration_cast < std::chrono::seconds > (end - start).count();
        std::cout << "Total Time Taken = " << diff << " Seconds" << std::endl;
        //Combine The Data
        std::string data = dbData + " :: " + fileData;
        //Printing the combined Data
        std::cout << "Data = " << data << std::endl;
        return 0;
    }


线程间通信
-----------------

线程间通信有两种方式：

1. [全局变量](https://thispointer.com//c11-multithreading-part-6-need-of-event-handling/)  
   缺点：等待线程会不停的查询全局变量，每次查询的时候会反复加锁/解锁
2. 条件变量(condition_variable)
   它使当前线程阻塞，直到信号通知条件变量或发生虚假唤醒为止。

.. code-block:: cpp

    #include<iostream>
    #include<thread>
    #include<vector>
    #include<mutex>
    class Wallet
    {
        int mMoney;
        std::mutex mutex;
    public:
        Wallet() :mMoney(0){}
        int getMoney()   {     return mMoney; }
        void addMoney(int money)
        {
            std::lock_guard<std::mutex> lockGuard(mutex);
            for(int i = 0; i < money; ++i)
            {
                mMoney++;
            }
        }
    };
    int testMultithreadedWallet()
    {
        Wallet walletObject;
        std::vector<std::thread> threads;
        for(int i = 0; i < 5; ++i){
            threads.push_back(std::thread(&Wallet::addMoney, &walletObject, 1000));
        }
        for(int i = 0; i < threads.size() ; i++)
        {
            threads.at(i).join();
        }
        return walletObject.getMoney();
    }
    int main()
    {
        int val = 0;
        for(int k = 0; k < 1000; k++)
        {
            if((val = testMultithreadedWallet()) != 5000)
            {
                std::cout << "Error at count = "<<k<<"  Money in Wallet = "<<val << std::endl;
                //break;
            }
        }
        return 0;
    }

锁(mutex)
`````````````````

Mutex，互斥量，就是互斥访问的量。只在多线程编程中起作用，在单线程程序中是没有什么用处。
从c++11开始，c++提供了std::mutex类型，对于多线程的加锁操作提供了很好的支持。

互斥量（Mutex）和二元信号量很类似，资源仅同时允许一个线程访问，
但和信号量不同的是，信号量在整个系统可以被任意线程获取并释放，也就是说，同一个信号量可以被系统中的一个线程获取之后由另一个线程释放。
而互斥量则要求哪个线程获取了互斥量，哪个线程就要负责释放这个锁，其他线程越俎代庖去释放互斥量是无效的。

**c++11中有4种锁类型**：

- std::mutex，最基本的 Mutex 类。
- std::recursive_mutex，递归 Mutex 类。
- std::time_mutex，定时 Mutex 类。
- std::recursive_timed_mutex，定时递归 Mutex 类。


std::mutex
::::::::::::::::::::

std::mutex 是C++11 中最基本的互斥量，std::mutex 对象提供了独占所有权的特性——即不支持递归地对 std::mutex 对象上锁，
而 std::recursive_lock 则可以递归地对互斥量对象上锁

构造函数:std::mutex不允许拷贝构造，也不允许 move 拷贝，最初产生的 mutex 对象是处于 unlocked 状态的。

lock():调用线程将锁住该互斥量。如果当前互斥量被其他线程锁住，则当前的调用线程被阻塞住；
如果当前互斥量被当前调用线程锁住，则会产生死锁(deadlock)

unlock():解锁，释放对互斥量的所有权。

try_lock():尝试锁住互斥量。如果当前互斥量被其他线程锁住，则当前调用线程返回 false，而并不会被阻塞掉；
如果当前互斥量被当前调用线程锁住，则会产生死锁(deadlock)    

.. code-block:: cpp

    #include <iostream>  // std::cout
    #include <thread>   // std::thread
    #include <mutex>   // std::mutex

    volatile int counter(0); // non-atomic counter
    std::mutex mtx;   // locks access to counter

    void attempt_10k_increases() {
    for (int i=0; i<10000; ++i) {
    if (mtx.try_lock()) { // only increase if currently not locked:
    ++counter;
    mtx.unlock();
    }
    }
    }

    int main (int argc, const char* argv[]) {
    std::thread threads[10];
    for (int i=0; i<10; ++i)
    threads[i] = std::thread(attempt_10k_increases);

    for (auto& th : threads) th.join();
    std::cout << counter << " successful increases of the counter.\n";

    return 0;
    }


std::recursive_mutex
::::::::::::::::::::::::::

和std::mutex不同的是，std::recursive_mutex 允许 **同一个线程** 对互斥量 **多次上锁** （即递归上锁），
来获得对互斥量对象的多层所有权，std::recursive_mutex 释放互斥量时需要调用与该锁层次深度相同次数的 unlock()，
可理解为 lock() 次数和 unlock() 次数相同，除此之外，std::recursive_mutex 的特性和 std::mutex 大致相同

std::time_mutex
:::::::::::::::::::::::::

std::time_mutex 比 std::mutex 多了两个成员函数，try_lock_for()，try_lock_until()。

try_lock_for 函数接受一个时间范围，表示在这一段时间范围之内线程如果没有获得锁则被阻塞住,
如果超时（即在指定时间内还是没有获得锁），则返回 false

try_lock_until 函数则接受一个时间点作为参数，在指定时间点未到来之前线程如果没有获得锁则被阻塞住，
如果超时（即在指定时间内还是没有获得锁），则返回 false。

.. code-block:: cpp

    #include <iostream>  // std::cout
    #include <chrono>   // std::chrono::milliseconds
    #include <thread>   // std::thread
    #include <mutex>   // std::timed_mutex

    std::timed_mutex mtx;
    void fireworks() {
    // waiting to get a lock: each thread prints "-" every 200ms:
    while (!mtx.try_lock_for(std::chrono::milliseconds(200))) {
    std::cout << "-";
    }
    // got a lock! - wait for 1s, then this thread prints "*"
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "*\n";
    mtx.unlock();
    }
    int main ()
    {
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
    threads[i] = std::thread(fireworks);

    for (auto& th : threads) th.join();
    return 0;
    }

std::recursive_timed_mutex
:::::::::::::::::::::::::::::::::::::

和 std:recursive_mutex 与 std::mutex 的关系一样

std::shared_mutex
::::::::::::::::::::::::::::::

shared_mutex 拥有二个访问级别：

- 共享 - 多个线程能共享同一互斥的所有权；
- 独占性 - 仅一个线程能占有互斥。

**只有一个线程可以占有写模式的读写锁，但是可以有多个线程占有读模式的读写锁。**读写锁也叫做“共享-独占锁”，
当读写锁以读模式锁住时，它是以共享模式锁住的；当它以写模式锁住时，它是以独占模式锁住的**。

- 当读写锁处于写加锁状态时，在其解锁之前，所有尝试对其加锁的线程都会被阻塞；
- 当读写锁处于读加锁状态时，所有试图以读模式对其加锁的线程都可以得到访问权，但是如果想以写模式对其加锁，线程将阻塞。
  这样也有问题，如果读者很多，那么写者将会长时间等待，如果有线程尝试以写模式加锁，
  那么后续的读线程将会被阻塞，这样可以避免锁长期被读者占有。

**排他性锁定**

lock/try_lock:锁定互斥。若另一线程已锁定互斥，则lock的调用线程将阻塞执行，直至获得锁。
若已以任何模式（共享或排他性）占有 mutex 的线程调用 lock ，则行为未定义。
也就是说， **已经获得读模式锁或者写模式锁的线程再次调用lock的话，行为是未定义的。**

unlock:解锁互斥。互斥必须为当前执行线程所锁定，否则行为未定义。
如果当前线程不拥有该互斥还去调用unlock，那么就不知道去unlock谁，行为是未定义的。

**共享锁定**

lock_shared/try_lock_shared:相比mutex，shared_mutex还拥有lock_shared函数。
该函数获得互斥的共享所有权。若另一线程以排他性所有权保有互斥，则lock_shared的调用者将阻塞执行，直到能取得共享所有权。
**若多于实现定义最大数量的共享所有者已以共享模式锁定互斥，则 lock_shared 阻塞执行**，直至共享所有者的数量减少。
所有者的最大数量保证至少为 10000。

unlock_shared:将互斥从调用方线程的共享所有权释放。当前执行线程必须以共享模式锁定互斥，否则行为未定义

锁存在的问题
:::::::::::::::::::::

虽然std::mutex可以对多线程编程中的共享变量提供保护，但是直接使用std::mutex的情况并不多。因为仅使用std::mutex有时候会发生死锁。

考虑这样一个情况：假设线程1上锁成功，线程2上锁等待。但是线程1上锁成功后，抛出异常并退出，没有来得及释放锁，
导致线程2“永久的等待下去”，此时就发生了死锁

.. code-block:: 

    #include <iostream>
    #include <thread>
    #include <vector>
    #include <mutex>
    #include <chrono>
    #include <stdexcept>

    int counter = 0;
    std::mutex mtx; // 保护counter

    void increase_proxy(int time, int id) {
        for (int i = 0; i < time; i++) {
            mtx.lock();
            // 线程1上锁成功后，抛出异常：未释放锁
            if (id == 1) {
                throw std::runtime_error("throw excption....");
            }
            // 当前线程休眠1毫秒
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            counter++;
            mtx.unlock();
        }
    }
    void increase(int time, int id) {
        try {
            increase_proxy(time, id);
        }
        catch (const std::exception& e){
            std::cout << "id:" << id << ", " << e.what() << std::endl;
        }
    }
    int main(int argc, char** argv) {
        std::thread t1(increase, 10000, 1);
        std::thread t2(increase, 10000, 2);
        t1.join();
        t2.join();
        std::cout << "counter:" << counter << std::endl;
        return 0;
    }

为了避免出现以上这种情况，一般使用lock_guard或unique_lock两个类对mutex进行管理

锁管理
`````````````````
std::lock_guard
:::::::::::::::::::::::::::

lock_guard 对象通常用于管理某个锁(Lock)对象；

在 lock_guard 对象构造时，传入的 Mutex 对象(即它所管理的 Mutex 对象)会被当前线程锁住。
在lock_guard 对象被析构时，它所管理的 Mutex 对象会自动解锁，由于不需要程序员手动调用 lock 和 unlock 对 Mutex 进行上锁和解锁操作，
因此这也是最简单安全的上锁和解锁方式，尤其是在程序抛出异常后先前已被上锁的 Mutex 对象可以正确进行解锁操作，
极大地简化了程序员编写与 Mutex 相关的异常处理代码

值得注意的是，lock_guard 对象并不负责管理 Mutex 对象的生命周期，
lock_guard 对象只是简化了 Mutex 对象的上锁和解锁操作，方便线程对互斥量上锁

构造函数:lock_guard 对象的拷贝构造和移动构造(move construction)均被禁用

.. code-block:: cpp

    explicit lock_guard (mutex_type& m);  //lock_guard 对象管理 Mutex 对象 m，并在构造时对 m 进行上锁（调用 m.lock()）
    lock_guard (mutex_type& m, adopt_lock_t tag); //lock_guard 对象管理 Mutex 对象 m,m 已被当前线程锁住
    //tag有三个可选项
    //std::adopt_lock  表明当前线程已经获得了锁，此后 mtx 对象的解锁操作交由 lock_guard 对象 lck 来管理，
    //                  在 lck 的生命周期结束之后，mtx 对象会自动解锁。
    //std::defer_lock  表明当前线程没有获得锁，后续需要去申请锁
    //std::try_to_lock  表示创建对象的时候尝试去申请锁
    lock_guard (const lock_guard&) = delete;  //拷贝构造被禁用


.. code-block:: cpp

    #include <iostream>    // std::cout
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::lock_guard
    #include <stdexcept>   // std::logic_error

    std::mutex mtx;
    void print_even (int x) {
    if (x%2==0) std::cout << x << " is even\n";
    else throw (std::logic_error("not even"));
    }

    void print_thread_id (int id) {
    try {
    // using a local lock_guard to lock mtx guarantees unlocking on destruction / exception:
    std::lock_guard<std::mutex> lck (mtx);
    print_even(id);
    }
    catch (std::logic_error&) {
    std::cout << "[exception caught]\n";
    }
    } 
    int main ()
    {
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
    threads[i] = std::thread(print_thread_id,i+1);
    for (auto& th : threads) th.join();
    return 0;
    }


std::unique_lock
:::::::::::::::::::::::::::

lock_guard 最大的缺点也是简单，没有给程序员提供足够的灵活度。unique_lock，与 lock_guard 类相似，
也很方便线程对互斥量上锁，但它提供了更好的上锁和解锁控制。

unique_lock 对象以独占所有权的方式（ unique owership）管理 mutex 对象的上锁和解锁操作，所谓独占所有权，
就是没有其他的 unique_lock 对象同时拥有某个 mutex 对象的所有权

std::unique_lock 对象也能保证在其自身析构时它所管理的 Mutex 对象能够被正确地解锁（即使没有显式地调用 unlock 函数）。
因此，和 lock_guard 一样，这也是一种简单而又安全的上锁和解锁方式，尤其是在程序抛出异常后先前已被上锁的 Mutex 对象可以正确进行解锁操作，极大地简化了程序员编写与 Mutex 相关的异常处理代码。

值得注意的是，unique_lock 对象同样也不负责管理 Mutex 对象的生命周期，unique_lock 对象只是简化了 Mutex 对象的上锁和解锁操作，
方便线程对互斥量上锁

构造函数:

.. code-block:: cpp

    unique_lock() noexcept;  //新创建的 unique_lock 对象不管理任何 Mutex 对象
    explicit unique_lock(mutex_type& m);  //新创建的unique_lock对象管理Mutex对象m,并尝试调用m.lock()对 Mutex对象进行上锁
    //新创建的unique_lock对象管理Mutex对象 m，并尝试调用m.try_lock()对 Mutex对象进行上锁
    unique_lock(mutex_type& m, try_to_lock_t tag);
    //新创建的 unique_lock 对象管理 Mutex 对象 m，但是在初始化的时候并不锁住 Mutex 对象
    unique_lock(mutex_type& m, defer_lock_t tag) noexcept;
    //新创建的 unique_lock 对象管理 Mutex 对象 m， m 应该是一个已经被当前线程锁住的 Mutex 对象。
    unique_lock(mutex_type& m, adopt_lock_t tag);
    //新创建的 unique_lock 对象管理 Mutex 对象 m，并试图通过调用 m.try_lock_for(rel_time) 来锁住 Mutex 对象一段时间。
    template <class Rep, class Period>
    unique_lock(mutex_type& m, const chrono::duration<Rep,Period>& rel_time);
    //新创建的 unique_lock 对象管理 Mutex 对象m，并试图通过调用 m.try_lock_until(abs_time)来在某个时间点之前锁住Mutex对象。
    template <class Clock, class Duration>
    unique_lock(mutex_type& m, const chrono::time_point<Clock,Duration>& abs_time);
    unique_lock(const unique_lock&) = delete; //拷贝构造 [被禁用]
    unique_lock(unique_lock&& x); //移动(move)构造
    unique_lock& operator= (unique_lock&& x) noexcept;   //移动赋值
    unique_lock& operator= (const unique_lock&) = delete; //普通赋值[被禁用]

.. code-block:: cpp

    #include <iostream>    // std::cout
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::lock, std::unique_lock
                // std::adopt_lock, std::defer_lock
    std::mutex foo,bar;
    void task_a () {
    std::lock (foo,bar);     // simultaneous lock (prevents deadlock)
    std::unique_lock<std::mutex> lck1 (foo,std::adopt_lock);
    std::unique_lock<std::mutex> lck2 (bar,std::adopt_lock);
    std::cout << "task a\n";
    // (unlocked automatically on destruction of lck1 and lck2)
    }
    void task_b () {
    // foo.lock(); bar.lock(); // replaced by:
    std::unique_lock<std::mutex> lck1, lck2;
    lck1 = std::unique_lock<std::mutex>(bar,std::defer_lock); // move-assigned
    lck2 = std::unique_lock<std::mutex>(foo,std::defer_lock);
    std::lock (lck1,lck2);    // simultaneous lock (prevents deadlock)
    std::cout << "task b\n";
    // (unlocked automatically on destruction of lck1 and lck2)
    }
    int main ()
    {
    std::thread th1 (task_a);
    std::thread th2 (task_b);
    th1.join();
    th2.join();
    return 0;
    }

成员函数:

::

    上锁/解锁操作：lock，try_lock，try_lock_for，try_lock_until 和 unlock
    修改操作：移动赋值(move assignment)(前面已经介绍过了)，
            交换(swap)（与另一个 std::unique_lock 对象交换它们所管理的 Mutex 对象的所有权），
            释放(release)（返回指向它所管理的 Mutex 对象的指针，并释放所有权）
    获取属性操作：owns_lock（返回当前 std::unique_lock 对象是否获得了锁）、
                operator bool()（与 owns_lock 功能相同，返回当前 std::unique_lock 对象是否获得了锁）、
                mutex（返回当前 std::unique_lock 对象所管理的 Mutex 对象的指针）。


**std::unique_lock::lock/std::unique_lock::unlock**

.. code-block:: cpp

    #include <iostream>    // std::cout
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::unique_lock, std::defer_lock
    std::mutex mtx;      // mutex for critical section
    void print_thread_id (int id) {
    std::unique_lock<std::mutex> lck (mtx,std::defer_lock);
    // critical section (exclusive access to std::cout signaled by locking lck):
    lck.lock();
    std::cout << "thread #" << id << '\n';
    lck.unlock();
    }
    int main ()
    {
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
    threads[i] = std::thread(print_thread_id,i+1);

    for (auto& th : threads) th.join();

    return 0;
    }


**std::unique_lock::try_lock**

.. code-block:: cpp

    #include <iostream>    // std::cout
    #include <vector>     // std::vector
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::unique_lock, std::defer_lock

    std::mutex mtx;      // mutex for critical section

    void print_star () {
    std::unique_lock<std::mutex> lck(mtx,std::defer_lock);
    // print '*' if successfully locked, 'x' otherwise: 
    if (lck.try_lock())
    std::cout << '*';
    else         
    std::cout << 'x';
    }

    int main ()
    {
    std::vector<std::thread> threads;
    for (int i=0; i<500; ++i)
    threads.emplace_back(print_star);

    for (auto& x: threads) x.join();

    return 0;
    }

**std::unique_lock::try_lock_for**

.. code-block:: cpp

    #include <iostream>    // std::cout
    #include <chrono>     // std::chrono::milliseconds
    #include <thread>     // std::thread
    #include <mutex>     // std::timed_mutex, std::unique_lock, std::defer_lock

    std::timed_mutex mtx;

    void fireworks () {
    std::unique_lock<std::timed_mutex> lck(mtx,std::defer_lock);
    // waiting to get a lock: each thread prints "-" every 200ms:
    while (!lck.try_lock_for(std::chrono::milliseconds(200))) {
    std::cout << "-";
    }
    // got a lock! - wait for 1s, then this thread prints "*"
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout << "*\n";
    }

    int main ()
    {
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
    threads[i] = std::thread(fireworks);

    for (auto& th : threads) th.join();

    return 0;
    }

**std::unique_lock::release**

.. code-block:: cpp

    //返回指向它所管理的 Mutex 对象的指针，并释放所有权。
    #include <iostream>    // std::cout
    #include <vector>     // std::vector
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::unique_lock
    std::mutex mtx;
    int count = 0;
    void print_count_and_unlock (std::mutex* p_mtx) {
    std::cout << "count: " << count << '\n';
    p_mtx->unlock();
    }
    void task() {
    std::unique_lock<std::mutex> lck(mtx);
    ++count;
    print_count_and_unlock(lck.release());
    }
    int main ()
    {
    std::vector<std::thread> threads;
    for (int i=0; i<10; ++i)
    threads.emplace_back(task);
    for (auto& x: threads) x.join();
    return 0;
    }

**std::unique_lock::owns_lock**

.. code-block:: cpp

    //返回当前 std::unique_lock 对象是否获得了锁
    #include <iostream>    // std::cout
    #include <vector>     // std::vector
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::unique_lock, std::try_to_lock
    std::mutex mtx;      // mutex for critical section
    void print_star () {
    std::unique_lock<std::mutex> lck(mtx,std::try_to_lock);
    // print '*' if successfully locked, 'x' otherwise: 
    if (lck.owns_lock())
    std::cout << '*';
    else         
    std::cout << 'x';
    } 
    int main ()
    {
    std::vector<std::thread> threads;
    for (int i=0; i<500; ++i)
    threads.emplace_back(print_star);
    for (auto& x: threads) x.join();
    return 0;
    }

**std::unique_lock::operator bool()**

.. code-block:: cpp

    //与 owns_lock 功能相同，返回当前 std::unique_lock 对象是否获得了锁。
    #include <iostream>    // std::cout
    #include <vector>     // std::vector
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::unique_lock, std::try_to_lock
    std::mutex mtx;      // mutex for critical section
    void print_star () {
    std::unique_lock<std::mutex> lck(mtx,std::try_to_lock);
    // print '*' if successfully locked, 'x' otherwise: 
    if (lck)
    std::cout << '*';
    else         
    std::cout << 'x';
    }
    int main ()
    {
    std::vector<std::thread> threads;
    for (int i=0; i<500; ++i)
    threads.emplace_back(print_star);
    for (auto& x: threads) x.join();
    return 0;
    }

**std::unique_lock::mutex**

.. code-block:: cpp

    //返回当前 std::unique_lock 对象所管理的 Mutex 对象的指针。
    #include <iostream>    // std::cout
    #include <thread>     // std::thread
    #include <mutex>     // std::mutex, std::unique_lock, std::defer_lock
    class MyMutex : public std::mutex {
    int _id;
    public:
    MyMutex (int id) : _id(id) {}
    int id() {return _id;}
    };
    MyMutex mtx (101);
    void print_ids (int id) {
    std::unique_lock<MyMutex> lck (mtx);
    std::cout << "thread #" << id << " locked mutex " << lck.mutex()->id() << '\n';
    }
    int main ()
    {
    std::thread threads[10];
    // spawn 10 threads:
    for (int i=0; i<10; ++i)
    threads[i] = std::thread(print_ids,i+1);
    for (auto& th : threads) th.join();
    return 0;
    }

std::shared_lock
:::::::::::::::::::::::

类 shared_lock 是通用 **共享互斥所有权包装器（unique_lock则是独占互斥所有权包装器）** ，允许延迟锁定、定时锁定和锁所有权的转移。
**锁定 shared_lock ，会以共享模式锁定关联的共享互斥** （`std::unique_lock` 可用于以排他性模式锁定）

方法和unique_lock一样，用法也相同

.. code-block:: cpp

    #include <iostream>
    #include <mutex>    //unique_lock
    #include <shared_mutex> //shared_mutex shared_lock
    #include <thread>
    std::mutex mtx;
    class ThreadSaferCounter
    {
    private:
        mutable std::shared_mutex mutex_;
        unsigned int value_ = 0;
    public:
        ThreadSaferCounter(/* args */) {};
        ~ThreadSaferCounter() {};

        unsigned int get() const {
            //读者, 获取共享锁, 使用shared_lock
            std::shared_lock<std::shared_mutex> lck(mutex_);//执行mutex_.lock_shared();
            return value_;  //lck 析构, 执行mutex_.unlock_shared();
        }

        unsigned int increment() {
            //写者, 获取独占锁, 使用unique_lock
            std::unique_lock<std::shared_mutex> lck(mutex_);//执行mutex_.lock();
            value_++;   //lck 析构, 执行mutex_.unlock();
            return value_;
        }

        void reset() {
            //写者, 获取独占锁, 使用unique_lock
            std::unique_lock<std::shared_mutex> lck(mutex_);//执行mutex_.lock();
            value_ = 0;   //lck 析构, 执行mutex_.unlock();
        }
    };
    ThreadSaferCounter counter;
    void reader(int id){
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::unique_lock<std::mutex> ulck(mtx);//cout也需要锁去保护, 否则输出乱序
            std::cout << "reader #" << id << " get value " << counter.get() << "\n";
        }    
    }

    void writer(int id){
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::unique_lock<std::mutex> ulck(mtx);//cout也需要锁去保护, 否则输出乱序
            std::cout << "writer #" << id << " write value " << counter.increment() << "\n";
        }
    }

    int main()
    {
        std::thread rth[10];
        std::thread wth[10];
        for(int i=0; i<10; i++){
            rth[i] = std::thread(reader, i+1);
        }
        for(int i=0; i<10; i++){
            wth[i] = std::thread(writer, i+1);
        }

        for(int i=0; i<10; i++){
            rth[i].join();
        }
        for(int i=0; i<10; i++){
            wth[i].join();
        }
        return 0;
    }



条件变量()condition_variable
```````````````````````````````````````

**wait()**:它使当前线程阻塞，直到信号通知条件变量或发生虚假唤醒为止。
它以原子方式释放附加的互斥锁，阻塞当前线程，并将其添加到等待当前条件变量对象的线程列表中。
当某些线程在同一条件变量对象上调用notify_one() 或notify_all() 时，该线程将被解除阻塞。
它也可能会被虚假地解除阻塞，因此，每次解除阻塞后，都需要再次检查条件。
如果不满足条件，则再次自动释放附加的互斥锁，阻塞当前线程，并将其添加到等待当前条件变量对象的线程列表中。

.. code-block:: cpp

    void wait (unique_lock<mutex>& lck);
    template <class Predicate>
    void wait (unique_lock<mutex>& lck, Predicate pred);
    //第一种形式只有一个参数unique_lock<mutex>&，调用wait时，若参数互斥量lck被锁定，则wait会阻塞。
    //第二种形式除了unique_lock<mutex>&参数外，第二个参数pred，即函数指针。
    // 当函数运行到该wait()函数时，若互斥量lck被锁定，且pred()返回值为false，则wait阻塞，
    // 必须同时满足，否则不会阻塞。其等同于下面的形式：
    while (!pred()) wait(lck);

**notify_one（）**：如果有多个线程在同一条件变量对象上等待，则notify_one解除阻塞其中一个正在等待的线程

**notify_all（）**：如果有多个线程在同一条件变量对象上等待，则notify_all取消阻止所有正在等待的线程。

.. code-block:: cpp

    #include <iostream>
    #include <thread>
    #include <functional>
    #include <mutex>
    #include <condition_variable>
    class Application
    {
    std::mutex m_mutex;
    std::condition_variable m_condVar;
    bool m_bDataLoaded;
    public:
    Application()
    {
        m_bDataLoaded = false;
    }
    void loadData()
    {
    // Make This Thread sleep for 1 Second
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::cout<<"Loading Data from XML"<<std::endl;
    // Lock The Data structure
    std::lock_guard<std::mutex> guard(m_mutex);
    // Set the flag to true, means data is loaded
    m_bDataLoaded = true;
    // Notify the condition variable
    m_condVar.notify_one();
    }
    bool isDataLoaded()
    {
        return m_bDataLoaded;
    }
    void mainTask()
    {
        std::cout<<"Do Some Handshaking"<<std::endl;
        // Acquire the lock
        std::unique_lock<std::mutex> mlock(m_mutex);
        // Start waiting for the Condition Variable to get signaled
        // Wait() will internally release the lock and make the thread to block
        // As soon as condition variable get signaled, resume the thread and
        // again acquire the lock. Then check if condition is met or not
        // If condition is met then continue else again go in wait.
        m_condVar.wait(mlock, std::bind(&Application::isDataLoaded, this));
        std::cout<<"Do Processing On loaded Data"<<std::endl;
    }
    };
    int main()
    {
    Application app;
    std::thread thread_1(&Application::mainTask, &app);
    std::thread thread_2(&Application::loadData, &app);
    thread_2.join();
    thread_1.join();
    return 0;
    }

mutex函数
`````````````````

- std::try_lock，尝试同时对多个互斥量上锁。
- std::lock，可以同时对多个互斥量上锁。
- std::call_once，如果多个线程需要同时调用某个函数，call_once 可以保证多个线程对该函数只调用一次

线程同步的四项原则
`````````````````````

1. 首要原则是尽量最低限度的共享对象，减少需要同步的场合。
   一个对象能不暴露给别的线程就不要暴露；如果要暴露，优先考虑immutable对象；实在不行才暴露可修改的对象，并用同步措施来保护它
2. 其次是使用高级的并发编程构建，如TaskQueue，Product-Consumer Queue,ConutDownLatch等等
3. 最后不得已使用底层同步原语（primitives）时，只使用非递归的互斥器和条件变量，慎用读写锁
4. 除了使用atomic整数之外，不自己编写lock-free代码，也不要用“内核级”同步原语。不凭空猜测哪种做法性能会更好，比如spin lock vs mutex

CountDownLatch
`````````````````````

倒计时（CountDownLatch）是一种常用且易用的同步手段。它主要有两种用途：

1. 主线程发起多个子线程，等这些子线程各自都完成一定的任务之后，主线程才继续执行。通常用于主线程等待多个子线程完成初始化。
2. 主线程发起多个子线程，子线程都等待主线程，主线程完成一些其他任务之后通知所有子线程开始执行。通常用于多个子线程等待主线程发出“起跑”命令。
   
.. code-block:: cpp

   #include <mutex>
   #include <condition_variable>
   
   class CountDownLatch {
   public:
       CountDownLatch(uint32_t count) : m_count(count) {}
   
       void countDown() noexcept {
           std::lock_guard<std::mutex> guard(m_mutex);
           if (0 == m_count) {
               return;
           }
           --m_count;
           if (0 == m_count) {
               m_cv.notify_all();
           }
       }
   
       void await() noexcept {
           std::unique_lock<std::mutex> lock(m_mutex);
           m_cv.wait(lock, [this] { return 0 == m_count; });
       }
   
   private:
       std::mutex m_mutex;
       std::condition_variable m_cv;
       uint32_t m_count;
   };


参考
----------------------

https://blog.csdn.net/acaiwlj/article/details/49818965

https://www.jb51.net/article/179681.htm

https://www.cnblogs.com/pluviophile/p/cpp11-future.html

https://www.cnblogs.com/chen-cs/p/13065948.html

https://thispointer.com//c-11-multithreading-part-1-three-different-ways-to-create-threads/

线程池：https://zhuanlan.zhihu.com/p/367309864

https://github.com/progschj/ThreadPool

https://www.cnblogs.com/lzpong/p/6397997.html