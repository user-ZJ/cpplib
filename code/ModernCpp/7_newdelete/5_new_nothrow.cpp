#include <iostream>
#include <vector>
#include <new>
using namespace std;

class Point{
public:
  int x;
  int y;
  int z;
};

int main(){

    try{

        for(;;){
            int* p=new int[100000];

        }
    }
    catch(bad_alloc){
        cerr<<"out of memory";
    }

    

    {

        for(;;){
            int* p=new(nothrow) int[100000];
            if(p==nullptr){

            }
        }

    }

}
