package section3

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

fun main(args: Array<String>) {
    val a = Nd4j.create(doubleArrayOf(0.3,2.9,4.0))
    println("${softmax(a)}")
}

private fun softmax(a: INDArray): INDArray {
    val maxA = a.maxNumber()
    val expA = Transforms.exp(a - maxA)
    val sumA = expA.sumNumber()
    return expA / sumA
}
