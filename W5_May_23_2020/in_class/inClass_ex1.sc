object W5_Inclass_Ex1
{
	def main(args : Array[ String ]) : Unit =
	{
		def greeting(x: String) : String = 
			{
				("Hello " + x + ", How u Doing")
			}               

		def frame(x: String, f: String => String) : String =
			{ 
				f(x)
			}                                                 
		val result2 = frame("Hal", greeting) 
		println(result2)
	}	

}