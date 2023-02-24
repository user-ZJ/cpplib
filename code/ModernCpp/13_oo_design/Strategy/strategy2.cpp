class TaxStrategy{
public:
    virtual double Calculate(const Context& context)=0;
    virtual ~TaxStrategy(){}
};

class CNTax : public TaxStrategy{
public:
    double Calculate(const Context& context) override 
    {  
        //*********** 
    }
};

class USTax : public TaxStrategy{
public:
    double Calculate(const Context& context) override
    { 
        //***********
    }
};

class DETax : public TaxStrategy{
public:
    double Calculate(const Context& context) override
    { 
        //***********
    }
};


//扩展....
class FRTax: public TaxStrategy{
public:
    double Calculate(const Context& context) override
    { 
        //***********
    }
};



class SalesOrder{
    unique_ptr<TaxStrategy> strategy;
public:
     double CalculateTax(){
        //...
        Context context();
        double val = 
            strategy->Calculate(context); //
        //...
    }
};




