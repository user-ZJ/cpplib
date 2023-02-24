#include <iostream>
using namespace std;


/*
class Widget {
public:

	Widget(string& n, vector<int>& c);
    Widget(string&& n, vector<int>&& c);
    Widget(string& n, vector<int>&& c);
    Widget(string&& n, vector<int>& c) ;
private:
	std::string name;
	std::vector<int> coordinates;
};



class Widget {
public:
	template<typename T1, typename T2>
	Widget(T1&& n, T2&& c) 		// n 和 c可以绑定任何值 
	: name(std::forward<T1>(n)), 	// 将 n 转发给string构造 
	coordinates(std::forward<T2>(c)) // 将 c 转发给vector构造 
	{}
private:
	std::string name;
	std::vector<int> coordinates;
}; 

*/






class Widget{
public:	
    Widget(){}
    Widget(const Widget& rhs){ cout<<"copy ctor"<<endl;}	
    Widget(Widget&& rhs){ cout<<"move ctor"<<endl; }	    
    Widget& operator=(Widget&& rhs)	{	
        cout<<"move assignment"<<endl;	
        return *this; 			
    }
	Widget& operator=(const Widget& rhs){
        cout<<"copy assignment"<<endl;
        return *this;
    }	
};

void process(Widget& param)
{
    cout<<"process left value"<<endl;
} 
void process(Widget&& param)
{
    cout<<"process right value"<<endl;
}




template<typename T>
void invoke(T&& param){ 
        cout<<"left or right? "<<endl;
        process(std::forward<T>(param));  
}


template<>
void invoke<Widget&> (Widget& param)
{
    cout<<"left or right? "<<endl;
    process(std::forward<Widget&>(param)); 
}
template<>
void invoke<Widget>(Widget&& param)
{
    cout<<"left or right? "<<endl;
    process(std::forward<Widget>(param)); 
}


int main()
{
    Widget w;

    invoke(w); 
    invoke(std::move(w)); 

}         


/*
//移除引用traits
template <class _Tp> 
struct  remove_reference        
{
    typedef  _Tp type;
};

template <class _Tp> 
struct  remove_reference<_Tp&>  
{
    typedef  _Tp type;
};

template <class _Tp> 
struct  remove_reference<_Tp&&> 
{
    typedef  _Tp type;
};


template <class _Tp>
inline constexpr _Tp&&
forward(typename remove_reference<_Tp>::type& __t) noexcept {
  return static_cast<_Tp&&>(__t);
}

template <class _Tp>
inline constexpr _Tp&&
forward(typename remove_reference<_Tp>::type&& __t) noexcept {
  static_assert(!is_lvalue_reference<_Tp>::value, "cannot forward an rvalue as an lvalue");
  return static_cast<_Tp&&>(__t);
}
*/
