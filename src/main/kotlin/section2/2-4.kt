package section2

import org.nd4j.linalg.factory.Nd4j
import util.*

fun main(args: Array<String>) {
    println("${xor(0,0)}")
    println("${xor(1,0)}")
    println("${xor(0,1)}")
    println("${xor(1,1)}")
}

private fun xor(x1: Int, x2: Int): Int {
    val s1 = nand(x1, x2)
    val s2 = or(x1, x2)
    return and(s1, s2)
}

private fun and(x1: Int, x2: Int): Int {
    val x = Nd4j.create(doubleArrayOf(x1.toDouble(),x2.toDouble()))
    val w = Nd4j.create(doubleArrayOf(0.5,0.5))
    val b = -0.7
    return if((x*w).sumNumber().toDouble() + b <= 0) 0 else 1
}

private fun nand(x1: Int, x2: Int): Int {
    val x = Nd4j.create(doubleArrayOf(x1.toDouble(),x2.toDouble()))
    val w = Nd4j.create(doubleArrayOf(-0.5,-0.5))
    val b = 0.7
    return if((x*w).sumNumber().toDouble() + b <= 0) 0 else 1
}

private fun or(x1: Int, x2: Int): Int {
    val x = Nd4j.create(doubleArrayOf(x1.toDouble(),x2.toDouble()))
    val w = Nd4j.create(doubleArrayOf(0.5,0.5))
    val b = -0.2
    return if((x*w).sumNumber().toDouble() + b <= 0) 0 else 1
}