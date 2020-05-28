val nums= (1 to 45).toList

val cond_1 = nums.filter( (x : Int) => x%4 == 0).sum
println(cond_1)

val cond_2 = nums.filter( (x : Int) => x%3 == 0).filter( (x : Int) => x < 20).map( (x: Int) => {x * x}).sum
println(cond_2)

