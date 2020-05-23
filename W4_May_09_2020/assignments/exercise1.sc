object Exercise_1 
{
	def main ( args : Array[ String ])
	{
		val x : Double = 42.354562;
		val y : Int = 3;
		println(f"x with 2 decimal places: $x%5.2f");
		println(f"y with 3 preceding zeros: $y%04d");			
	}	
}