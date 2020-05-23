object Exercise2_and_3
{
	def main(args : Array[ String ]) 
	{
		def test_data_type(num : Any): Any = num match 
		{
			case y : Int => println(factorial(y))
			case z : Double => println(factorial(z))
			case _ => "NA"
		}
	
		def factorial(num : Double): BigInt =
		{
			val new_num = num.toInt;
			if (new_num == 0)
			{
				return 1;
			} 
			else 
			{
				return new_num * factorial(new_num-1);
			}	
		}


	println (test_data_type(9))
	println (test_data_type(5))
	println (test_data_type(3.8))
	println (test_data_type("i"))

	}	
}