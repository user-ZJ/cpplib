
#include <functional>
#include <algorithm>
#include <vector>
#include <iostream>
#include <string>
using namespace std;
using namespace std::placeholders;

class Language {
  private:
    string name;
  public:
    Language (const string& n) : name(n) {
    }
    void print () const {
        cout << name << " ";
    }
    void show (const string& postfix) const {
        cout  << name << postfix;
    }
    //...
};

int main()
{
    vector<Language> coll
            = { Language("C++"), Language("Rust"), Language("GO") };

     //Language l("C++");
     //l.print();// print(&l);
   
    for_each (coll.begin(), coll.end(),
              bind(&Language::print,_1));
    cout << endl;

    for_each( coll.begin(), coll.end(),
              mem_fn(&Language::print));
    cout << endl;

    for_each (coll.begin(), coll.end(),
              bind(&Language::show,_1," * "));
    cout << endl;

    for_each (coll.begin(), coll.end(),
              bind(mem_fn(&Language::show), _1, " -> "));
              
    cout << endl;
    cout<<"\nlambda: "<<endl;

    for_each(coll.begin(), coll.end(),
            [](const Language& lang) { lang.print(); });
    cout << endl;
    for_each(coll.begin(), coll.end(),
            [](const Language& lang) { lang.show(" -> "); });
    cout << endl;
    


}
