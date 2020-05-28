object W5_Ex3
{
	def main(args : Array[ String ]) : Unit =
	{
		def max(x : Int, y : Int) : Int = 
			{
				if (x > y) 
				{
					return x
				}
				else
				{
					return y
				}
			}
		def get_max(x: Int, y : Int, f: (Int, Int) => Int) : Int =
		{
			f(x, y)
		} 
		val result = get_max(293, 406, max)
		println(result)	
	}	
}

