#include <iostream>
#include <memory>
using namespace std;


class Document{
public:
    virtual void process()=0;
    virtual ~Document()=default;
};




class Markdown: public Document{
public:
    void process() override {
        //...
    }
};

class HTML: public Document{
public:
    void process() override {
        //...
    }
};


void invoke(Document& doc){

   
    doc.process();
}


int main(){

     auto pDoc=make_unique<Markdown>();

    invoke(*pDoc);
}