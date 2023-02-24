

class MyClass {
public:
	
	MyClass();
	~MyClass();


	MyClass(const MyClass& other);
	MyClass& operator=(const MyClass& rhs);

	void process();


private:
	// 前置声明
	class Impl;

	// 实现类指针
	Impl* pimpl;//Impl不完整类型
};
