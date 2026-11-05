

#include <memory>
#include <vector>
#include <iostream>
using namespace std;



struct Node
{
    int data;
    // Node* next;
    std::unique_ptr<Node> next;

    ~Node()
    {
        cout<<"dtor: "<<data<<endl;

    }
};

struct List
{

    std::unique_ptr<Node> head;
 
    void push(int data)
    {
        head = std::unique_ptr<Node>(new Node{data, std::move(head)});
    }

    //避免递归结构带来栈溢出
    // ~List(){
    //     while(head)
    //     {
    //         head=std::move(head->next);
    //     }
    // }

};

int main()
{
    
    List mylist;

    for (int i = 1; i < 100000; i++)
    {
        mylist.push(i);
    }


}