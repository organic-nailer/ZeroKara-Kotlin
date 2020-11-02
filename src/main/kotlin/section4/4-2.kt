package section4

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

fun main(args: Array<String>) {
    section4_2_1()
    section4_2_2()
}

private fun section4_2_1() {
    println("section4.2.1")
    val t = Nd4j.create(doubleArrayOf(0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0))
    val y1 = Nd4j.create(doubleArrayOf(0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0))
    val y2 = Nd4j.create(doubleArrayOf(0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0))

    println("誤差=${meanSquaredError(y1,t)}")
    println("誤差=${meanSquaredError(y2,t)}")
}
//二乗和誤差
private fun meanSquaredError(y: INDArray, t: INDArray): Double {
    return 0.5 * Transforms.pow(y-t, 2).sumNumber().toDouble()
}

private fun section4_2_2() {
    println("section4.2.2")
    val t = Nd4j.create(doubleArrayOf(0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0))
    val y1 = Nd4j.create(doubleArrayOf(0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0))
    val y2 = Nd4j.create(doubleArrayOf(0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0))

    println("誤差=${crossEntropyError(y1,t)}")
    println("誤差=${crossEntropyError(y2,t)}")
}
//交差エントロピー誤差
private fun crossEntropyError(y: INDArray, t: INDArray): Double {
    val delta = 0.0000001
    return -(t * Transforms.log(y + delta)).sumNumber().toDouble()
}
