#include <iostream>
#include <string>
#include <memory>
using namespace std;

class Library
{
public:
	void run() //template method
	{
		step1();
		while (!step2()) //多态调用
			step3();
		step4(); //多态调用
		step5();
	}

	virtual ~Library(){}

protected: //或者private 
	void step1() 
	{
		cout << "Library.step1()"<<endl;
	}

	 void step3() 
	{
		cout << "Library.step3()"<<endl;
	}

	 void step5()
	{
		cout << "Library.step5()"<<endl;
	}
	int number{0};
	
private: //NVI: Non-Virtual Interface
	virtual bool step2() = 0;
	virtual int step4() = 0;

};

//============================

class App : public Library
{

private:
	bool step2() override
	{
		//Library::step2();//静态绑定

        cout<<"App.step2()"<<endl;
        number++;
		return number>=4;
	}
	
	int step4() override
	{
        cout<<"App.step4() : "<<number<<endl;
		return number;
	}

};

int main()
{
	auto pLib=make_unique<App>(); 
	
	pLib->run();


	return 0;
}