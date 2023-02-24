#include <iostream>
#include <vector>

using namespace std;

template <typename T>
concept Document = std::semiregular<T> &&
requires(T t) 
{
    t.process() ;
};

template <typename T>
concept DocumentExt = Document<T> &&
requires(T t) 
{
    t.extend() ;
};

template <typename T>
concept DocumentImp = Document<T> &&
requires(T t) 
{
    t.process_imp() ;
};

// template <typename T>
// concept TxtDocument = DocumentExt<T> &&
// DocumentImp<T> &&
// requires(T t) 
// {
//     t.read_text() ;

// };


template <typename T>
concept TxtDocument = std::semiregular<T> &&
requires(T t) 
{
    t.process() ;
    t.extend() ;
    t.process_imp() ;
    t.read_text() ;
};




class Markdown{
public:
    void process()  {
        cout<<"process"<<endl;
    }

    void extend()  {
        cout<<"extend"<<endl;
    }

    void process_imp()  {
        cout<<"process_imp"<<endl;
    }

    void read_text()  {
        cout<<"read_text"<<endl;
    }
};


template<TxtDocument T>
void invoke(){
    T doc;
    doc.process();
    doc.extend();
    doc.process_imp();
    doc.read_text();

}

int main(){

    invoke<Markdown>();
    return 0;
}