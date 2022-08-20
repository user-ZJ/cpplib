C++使用笔记
===============

继承和派生
--------------------

::
    
    继承和派生是同一概念。
    派生类名::派生类名(参数表):基类名1(参数表),基类名2(参数表)
    {
        本类成员初始化赋值语句
    }

    派生类名::派生类名(参数表):基类名1(参数表),基类名2(参数表),新增成员对象的初始化
    {
        本类成员初始化赋值语句
    }

时间统计
-----------------

.. code-block:: cpp

    #include<chrono>
    auto begin_t = std::chrono::steady_clock::now();
    auto finish_t = std::chrono::steady_clock::now();
    double timecost = std::chrono::duration<double, std::milli>(finish_t - begin_t).count();

