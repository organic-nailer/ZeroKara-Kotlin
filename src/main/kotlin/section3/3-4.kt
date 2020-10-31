package section3

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

//3層ニューラルネットワークの実装
fun main(args: Array<String>) {
    section3_4_2()
}

private fun section3_4_2() {
    println("section3.4.2")
    val x = Nd4j.create(doubleArrayOf(1.0, 0.5))
    val w1 = Nd4j.create(arrayOf(
            doubleArrayOf(0.1, 0.3, 0.5),
            doubleArrayOf(0.2, 0.4, 0.6)
    ))
    val b1 = Nd4j.create(doubleArrayOf(0.1, 0.2, 0.3))
    println(w1.shape().contentToString())
    println(x.shape().contentToString())
    println(b1.shape().contentToString())
    //0層->1層
    val a1 = (x dot w1) + b1
    println("$a1")
    val z1 = Transforms.sigmoid(a1)
    println("z1=$z1")
    //1層->2層
    val w2 = Nd4j.create(arrayOf(
            doubleArrayOf(0.1, 0.4),
            doubleArrayOf(0.2, 0.5),
            doubleArrayOf(0.3, 0.6)
    ))
    val b2 = Nd4j.create(doubleArrayOf(0.1, 0.2))
    val a2 = (z1 dot w2) + b2
    val z2 = Transforms.sigmoid(a2)
    println("z2=$z2")
    //2層->出力層
    val w3 = Nd4j.create(arrayOf(
            doubleArrayOf(0.1,0.3),
            doubleArrayOf(0.2,0.4)
    ))
    val b3 = Nd4j.create(doubleArrayOf(0.1,0.2))
    val a3 = (z2 dot w3) + b3
    val y = a3
    println("y=$y")
}
