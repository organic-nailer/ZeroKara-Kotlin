package section2

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

operator fun INDArray.plus(other: INDArray): INDArray = this.add(other)
operator fun INDArray.minus(other: INDArray): INDArray = this.sub(other)
operator fun INDArray.times(other: INDArray): INDArray = this.mul(other)
operator fun INDArray.div(other: INDArray): INDArray = this.div(other)

operator fun INDArray.plus(other: Number): INDArray = this.add(other)
operator fun INDArray.minus(other: Number): INDArray = this.sub(other)
operator fun INDArray.times(other: Number): INDArray = this.mul(other)
operator fun INDArray.div(other: Number): INDArray = this.div(other)

fun main(args: Array<String>) {
    println("${or(0,0)}")
    println("${or(1,0)}")
    println("${or(0,1)}")
    println("${or(1,1)}")
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
