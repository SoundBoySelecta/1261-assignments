         
val nums = (1 to 9).toList
val oddNums = nums.filter( (x: Int) => x%2 != 0)
//for (n <- oddNums) {println(n)}]              
def fact(n : Int) : Int =
{
	var f = 1
  	
  	for (i <- 1 to n)
  		{
  			f = f * i ;
  		}
		
  	return f;
 
}                                                

val factlist = oddNums.map(x => fact(x))
for(num <- factlist)
{
	println(num)
}         
                                                  
 