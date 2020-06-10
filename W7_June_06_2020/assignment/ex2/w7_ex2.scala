// Databricks notebook source
// MAGIC %scala
// MAGIC val nums= (1 to 45).toList
// MAGIC 
// MAGIC val cond_1 = nums.filter( (x : Int) => x%4 == 0).sum
// MAGIC println(cond_1)
// MAGIC 
// MAGIC val cond_2 = nums.filter( (x : Int) => x%3 == 0).filter( (x : Int) => x < 20).map( (x: Int) => {x * x}).sum
// MAGIC println(cond_2)
// MAGIC 
// MAGIC def max(x : Int, y : Int) : Int = 
// MAGIC 	{
// MAGIC 		if (x > y) 
// MAGIC 			{
// MAGIC 				return x
// MAGIC 			}
// MAGIC 		else
// MAGIC 			{
// MAGIC 				return y
// MAGIC 			}
// MAGIC 	}
// MAGIC 
// MAGIC def get_max(x: Int, y : Int, f: (Int, Int) => Int) : Int =
// MAGIC 	{
// MAGIC 		f(x, y)
// MAGIC 	} 
// MAGIC val result = get_max(293, 406, max)
// MAGIC println(result)	
