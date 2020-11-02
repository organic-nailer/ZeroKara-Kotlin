package section4

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

fun main() {
    section4_4_2()
}

private fun section4_4_0() {
    println("${numericalGradient({ x -> function2ND(x) }, Nd4j.create(doubleArrayOf(3.0, 4.0)))}")
    println("${numericalGradient({ x -> function2ND(x) }, Nd4j.create(doubleArrayOf(0.0, 2.0)))}")
    println("${numericalGradient({ x -> function2ND(x) }, Nd4j.create(doubleArrayOf(3.0, 0.0)))}")
}

private fun function2ND(x: INDArray): Double {
    return Transforms.pow(x, 2).sumNumber().toDouble()
}

private fun numericalGradient(f: (INDArray) -> Double, x: INDArray): INDArray {
    val h = 1e-4
    val grad = Nd4j.zerosLike(x)

    for(i in 0 until x.length()) {
        val tmp = x.getDouble(i)
        x.putScalar(i, tmp + h)
        val fxh1 = f(x)

        x.putScalar(i, tmp - h)
        val fxh2 = f(x)

        grad.putScalar(i, (fxh1 - fxh2) / (2 * h))
        x.putScalar(i, tmp)
    }
    return grad
}


private fun section4_4_1() {
    println("${gradientDescent({ x -> function2ND(x) }, 0.1, Nd4j.create(doubleArrayOf(-3.0, 4.0)))}")
    println("${gradientDescent({ x -> function2ND(x) }, 1e-10, Nd4j.create(doubleArrayOf(-3.0, 4.0)))}")
    println("${gradientDescent({ x -> function2ND(x) }, 10.0, Nd4j.create(doubleArrayOf(-3.0, 4.0)))}")
}

private fun gradientDescent(f: (INDArray) -> Double, leaningRate: Double, initial: INDArray, step: Int = 100): INDArray {
    var x = initial

    for(i in 0 until step) {
        val grad = numericalGradient(f, x)
        x -= leaningRate * grad
    }
    return x
}


private fun section4_4_2() {
    val net = SimpleNet()
    println("w=${net.w}")
    val x = Nd4j.create(arrayOf(doubleArrayOf(0.6,0.9)))
    val p = net.predict(x)
    println("p=$p")
    println("maxIndex=${p.argMax()}")
    val t = Nd4j.create(doubleArrayOf(0.0,0.0,1.0))
    println("loss=${net.loss(x,t)}")
    val dw = numericalGradient({ net.loss(x, t) }, net.w)
    println("dw=$dw")
}

class SimpleNet {
    val w = Nd4j.rand(DataType.DOUBLE, 2, 3)

    fun predict(x: INDArray): INDArray = x dot w

    fun loss(x: INDArray, t: INDArray): Double {
        val z = predict(x)
        val y = Transforms.softmax(z)
        return crossEntropyError(y, t)
    }

    private fun crossEntropyError(y: INDArray, t: INDArray): Double {
        val delta = 0.0000001
        return -(t * Transforms.log(y + delta)).sumNumber().toDouble()
    }
}
