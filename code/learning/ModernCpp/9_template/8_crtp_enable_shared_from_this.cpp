#include <iostream>
#include <memory> 
using namespace std;
template<typename T>
class enable_shared_from_this
{
    private:
      weak_ptr<T>  weak_this;

    protected:
      constexpr enable_shared_from_this() noexcept { }

      enable_shared_from_this(const enable_shared_from_this&) noexcept { }

      enable_shared_from_this&
      operator=(const enable_shared_from_this&) noexcept
      { return *this; }

      ~enable_shared_from_this() { }

    public:
      shared_ptr<T> shared_from_this() //获取一个子类的共享指针
      { return shared_ptr<T>(this->weak_this); }

      shared_ptr<const T> shared_from_this() const
      { return shared_ptr<const T>(this->weak_this); }

};


// 扩展接口： 父类里面注入子类信息
class Widget : public std::enable_shared_from_this<Widget> {
public:
    
    std::shared_ptr<Widget> getWidget() {

         return shared_from_this(); // OK
    }

    void invoke(){
        // process(shared_from_this());
    }


    void print(){
        cout<<"print"<<endl;
    }

    ~Widget()
    {
        cout<<"~Widget()"<<endl;
    }

    //工厂函数
    static std::shared_ptr<Widget> create() {
        return std::shared_ptr<Widget>(new Widget());
    }

private:
    Widget()=default;
};

int main(){
    return 0;
}
