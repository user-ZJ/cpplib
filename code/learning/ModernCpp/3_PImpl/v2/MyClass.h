#include<memory>

class MyClass {
public:
	
	MyClass();
	~MyClass();


	MyClass(const MyClass& other);
	MyClass& operator=(const MyClass& rhs);
	MyClass(MyClass &&other) noexcept;
	MyClass &operator=(MyClass &&rhs) noexcept;

	void process();


private:
	// 前置声明
	class Impl;

	// 实现类指针
	std::unique_ptr<Impl> pimpl;
};
