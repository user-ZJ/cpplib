#include "MyClass.h"
#include <iostream>
using namespace std;

class MyClass::Impl {
public:
	virtual void invoke()
	{
		cout << "invoke" << endl;
	}

	virtual ~Impl(){
		cout<<"Impl dtor"<<endl;
	}

    Impl(){
		cout<<"Impl ctor"<<endl;
	}
	
	string text;
	int data;
};


MyClass::MyClass()
	: pimpl(new Impl())
{
	
}

MyClass::~MyClass()
{
    delete pimpl;
}

// 赋值与拷贝

MyClass::MyClass(const MyClass& other)
	: pimpl(new Impl(*other.pimpl))
{
}

MyClass& MyClass::operator=(const MyClass& rhs)
{
	if(this==&rhs)
	{
		return *this;
	}

	//copy & swap 
	MyClass temp(rhs);
	swap(this->pimpl, temp.pimpl); 
	return *this;
}





void MyClass::process()
{
	return pimpl->invoke();
}
