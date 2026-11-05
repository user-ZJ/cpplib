enum class TaxBase {
	CN_Tax,
	US_Tax,
	DE_Tax,
    FR_Tax // 变更

};

class SalesOrder{
    TaxBase tax;
public:
    double CalculateTax(){
        //...
        
        if (tax == TaxBase::CN_Tax){
            //CN***********
        }
        else if (tax == TaxBase::US_Tax){
            //US***********
        }
        else if (tax == TaxBase::DE_Tax){
            //DE***********
        }
        //变更....
        else if(tax==TaxBase::FR_Tax)
        {
            
        }
        

        //....
        return 0;
     }
    
};

int main(){
    
}
