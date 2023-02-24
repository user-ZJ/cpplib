#include <iostream>
#include <vector>

using namespace std;

template <typename T>
concept Document = requires(T t) 
{
    t.process() ;
};



class Markdown
{
public:
    void process()  {
        cout<<"process Markdown"<<endl;
    }

 
};


class Html
{
public:
    void process()  {
        cout<<"process Html"<<endl;
    }

 
};


template<Document T>
void invoke( T& doc){

    doc.process();
 

}

int main(){

    Html doc;
    invoke(doc);
}