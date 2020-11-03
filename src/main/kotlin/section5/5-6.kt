package section5

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

class AffineLayer(val w: INDArray, val b: INDArray) {
    var x: INDArray? = null
    var dW: INDArray? = null
    var dB: INDArray? = null

    fun forward(inX: INDArray): INDArray {
        x = inX
        return (x!! dot w) + b
    }

    fun backward(dOut: INDArray): INDArray {
        dW = x!!.transpose() dot dOut
        dB = dOut.sum(0)
        return dOut dot w.transpose()
    }
}

class SoftmaxWithLossLayer {
    var loss: Double? = null
    var y: INDArray? = null
    var t: INDArray? = null

    fun forward(inX: INDArray, inT: INDArray): Double {
        t = inT
        y = softmax(inX)
        loss = crossEntropyError(y!!, t!!)

        return loss!!
    }

    fun backward(): INDArray {
        val batchSize = t!!.shape()[0]
        return (y!! - t!!) / batchSize
    }

    private fun softmax(a: INDArray): INDArray {
        val maxA = a.maxNumber()
        val expA = Transforms.exp(a - maxA)
        val sumA = expA.sumNumber()
        return expA / sumA
    }

    private fun crossEntropyError(y: INDArray, t: INDArray): Double {
        val delta = 0.0000001
        return -(t * Transforms.log(y + delta)).sumNumber().toDouble()
    }
}
