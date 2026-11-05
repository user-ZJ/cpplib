int* glob;
void process(int* q){
  // …
  glob = q; 
}

void g1(){
  int* p = new int{7};
  process(p);
  delete p;
 
  // … 
  *glob = 9; 
}

void g2(){
  int* p = new int{7};
  process(p);
  delete p; 
  
  // …
  delete glob; 
}

void g3(){
  int x = 7;
  int* p = &x;
  process(p);
  // …
  delete glob;
}

int main(){
  return 0;
}





