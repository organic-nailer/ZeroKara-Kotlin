package section5

import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

fun main() {
    section5_5_1()
}

private fun section5_5_1() {
    val x = Nd4j.create(arrayOf(
            doubleArrayOf(1.0, -0.5),
            doubleArrayOf(-2.0, 3.0)
    ))

    println("$x")

    val relu = ReluLayer()

    println("${relu.forward(x)}")
}

class ReluLayer {
    var mask: INDArray? = null

    fun forward(x: INDArray): INDArray {
        mask = x.gt(0)
        println("$mask")
        return x.putWhereWithMask(mask, Nd4j.zerosLike(x))
    }

    fun backward(dOut: INDArray): INDArray {
        return dOut.putWhereWithMask(mask, Nd4j.zerosLike(dOut))
    }
}

class SigmoidLayer {
    var out: INDArray? = null

    fun forward(x: INDArray): INDArray {
        out = 1 / (1 + Transforms.exp(-x))
        return out!!
    }

    fun backward(dOut: INDArray): INDArray {
        return dOut * (1 - out!!) * out!!
    }
}
