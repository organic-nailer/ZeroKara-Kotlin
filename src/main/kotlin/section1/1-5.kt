package section1

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.shape.Broadcast
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing

operator fun INDArray.plus(other: INDArray): INDArray = this.add(other)
operator fun INDArray.minus(other: INDArray): INDArray = this.sub(other)
operator fun INDArray.times(other: INDArray): INDArray = this.mul(other)
operator fun INDArray.div(other: INDArray): INDArray = this.div(other)

operator fun INDArray.plus(other: Number): INDArray = this.add(other)
operator fun INDArray.minus(other: Number): INDArray = this.sub(other)
operator fun INDArray.times(other: Number): INDArray = this.mul(other)
operator fun INDArray.div(other: Number): INDArray = this.div(other)

fun main(args: Array<String>) {
    section1_5_2()
    section1_5_3()
    section1_5_4()
    section1_5_5()
    section1_5_6()
}

private fun section1_5_2() {
    println("section 1.5.2")
    val x = Nd4j.create(doubleArrayOf(1.0,2.0,3.0))
    println("$x")
}

private fun section1_5_3() {
    println("section 1.5.3")
    val x = Nd4j.create(doubleArrayOf(1.0,2.0,3.0))
    val y = Nd4j.create(doubleArrayOf(2.0,4.0,6.0))
    println("${x+y}")
    println("${x-y}")
    println("${x*y}")
    println("${x/y}")
    println("${x/2.0}")
}

private fun section1_5_4() {
    println("section 1.5.4")
    val A = Nd4j.create(arrayOf(
            doubleArrayOf(1.0,2.0),
            doubleArrayOf(3.0,4.0)
    ))
    println("$A")
    println("${A.shape()}")
    val B = Nd4j.create(arrayOf(
            doubleArrayOf(3.0,0.0),
            doubleArrayOf(0.0,6.0)
    ))
    println("${A+B}")
    println("${A*B}")
    println("${A*10}")
}

private fun section1_5_5() {
    println("section 1.5.5")
    val A = Nd4j.create(arrayOf(
            doubleArrayOf(1.0,2.0),
            doubleArrayOf(3.0,4.0)
    ))
    val B = Nd4j.create(doubleArrayOf(10.0,20.0))
    println("${A.mulRowVector(B)}")
}

private fun section1_5_6() {
    println("section 1.5.6")
    val X = Nd4j.create(arrayOf(
            doubleArrayOf(51.0,55.0),
            doubleArrayOf(14.0,19.0),
            doubleArrayOf(0.0,4.0)
    ))
    println("$X")
    println("${X.getRow(0)}")
    println("${X.getDouble(0,1)}")
    for(i in 0 until X.rows()) {
        println("${X.getRow(i)}")
    }
    val X2 = X.reshape(1, X.length())
    println("$X2")
    println("${X2.getColumns(0,2,4)}")
    println("${X2.gt(15)}") // X2 > 15
    println("${X2.data().asDouble().filter { it > 15 }}")
}
