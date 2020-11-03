package section5

import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import section4.TwoLayerNet
import util.dot
import util.minus
import util.plus
import util.times

fun main() {
    section5_7_4()
}

private fun section5_7_3() {
    val batchSize = 3

    val network = TwoLayerNet2(784, 50, 10)
    val trainData = EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, batchSize, false, true, true, 0)

    val batch = trainData.next()
    val xBatch = batch.features
    val tBatch = batch.labels

    val gradNumerical = network.getNumericalGradient(xBatch, tBatch)
    val gradBackprop = network.gradient(xBatch, tBatch)

    println("weight1: ${Transforms.abs(gradNumerical.dWeight1 - gradBackprop.dWeight1).sumNumber().toDouble() / gradNumerical.dWeight1.length().toDouble()}")
    println("bias1: ${Transforms.abs(gradNumerical.dBias1 - gradBackprop.dBias1).sumNumber().toDouble() / gradNumerical.dBias1.length().toDouble()}")
    println("weight2: ${Transforms.abs(gradNumerical.dWeight2 - gradBackprop.dWeight2).sumNumber().toDouble() / gradNumerical.dWeight2.length().toDouble()}")
    println("bias2: ${Transforms.abs(gradNumerical.dBias2 - gradBackprop.dBias2).sumNumber().toDouble() / gradNumerical.dBias2.length().toDouble()}")
}

private fun section5_7_4() {
    val batchSize = 100
    val learningRate = 0.1
    val lossList = mutableListOf<Double>()
    val trainAccList = mutableListOf<Double>()
    val testAccList = mutableListOf<Double>()

    val network = TwoLayerNet2(784, 50, 10)

    var index = 0

    val testData = EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, batchSize, false, false, true, 0).next()

    for(epoch in 0 until 3) {
        val trainData = EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, batchSize, false, true, true, 0)

        for(batch in trainData) {
            val bImage = batch.features
            val bLabel = batch.labels

            val grad = network.gradient(bImage, bLabel)
            //println("grad=$grad")

            network.wb1.weight -= learningRate * grad.dWeight1
            network.wb1.bias -= learningRate * grad.dBias1
            network.wb2.weight -= learningRate * grad.dWeight2
            network.wb2.bias -= learningRate * grad.dBias2

            val loss = network.loss(bImage, bLabel)
            lossList.add(loss)
            println("${index++}: $loss")

            if(!trainData.hasNext()) {
                val trainAcc = network.accuracy(bImage, bLabel)
                val testAcc = network.accuracy(testData.features, testData.labels)
                trainAccList.add(trainAcc)
                testAccList.add(testAcc)
                println("$trainAcc : $testAcc")
            }
        }
    }

    println("train: $trainAccList")
    println("test : $testAccList")

    //println("$lossList")
    //println("${lossList.size}")

    val data = mapOf<String, Any>(
            "x" to Nd4j.arange(0.0, lossList.size.toDouble()).data().asDouble(),
            "y" to lossList.toDoubleArray()
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
    }
    val p = ggplot(data) + geom + ggtitle("loss")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    //println("plotSpec=$plotSpec")
    //println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "loss")
}

class TwoLayerNet2(
        inputSize: Int, hiddenSize: Int, outputSize: Int,
        weightInitStd: Double = 0.01) {
    val wb1 = WeightAndBias(
            weightInitStd * Nd4j.rand(inputSize, hiddenSize),
            Nd4j.zeros(hiddenSize)
    )
    val wb2 = WeightAndBias(
            weightInitStd * Nd4j.rand(hiddenSize, outputSize),
            Nd4j.zeros(outputSize)
    )
//    var weight1 = weightInitStd * Nd4j.rand(inputSize, hiddenSize)
//    var bias1 = Nd4j.zeros(hiddenSize)
//    var weight2 = weightInitStd * Nd4j.rand(hiddenSize, outputSize)
//    var bias2 = Nd4j.zeros(outputSize)

    private val layers = listOf<NeuralLayer>(
            AffineLayer(wb1),
            ReluLayer(),
            AffineLayer(wb2)
    )
    private val lastLayer: NeuralLastLayer = SoftmaxWithLossLayer()

    private fun predict(x: INDArray): INDArray {
        var tmp = x
        layers.forEach {
            tmp = it.forward(tmp)
        }
        return tmp
    }

    fun loss(x: INDArray, t: INDArray): Double {
        return lastLayer.forward(predict(x), t)
    }

    fun accuracy(x: INDArray, t: INDArray): Double {
        val yMax = predict(x).argMax(1)
        val tMax = t.argMax(1) //注意
        return tMax.eq(yMax).castTo(DataType.DOUBLE).sumNumber().toDouble() / x.shape()[0]
    }

    fun getNumericalGradient(x: INDArray, t: INDArray): TwoLayerNet.GradientData {
        return TwoLayerNet.GradientData(
                dWeight1 = numericalGradient({ loss(x, t) }, wb1.weight),
                dBias1 = numericalGradient({ loss(x, t) }, wb1.bias),
                dWeight2 = numericalGradient({ loss(x, t) }, wb2.weight),
                dBias2 = numericalGradient({ loss(x, t) }, wb2.bias)
        )
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

    fun gradient(x: INDArray, t: INDArray): GradientData {
        //forward
        loss(x, t)

        //backward
        var dOut = lastLayer.backward()
        layers.asReversed().forEach {
            dOut = it.backward(dOut)
        }

        //ガバガバ実装
        return GradientData(
                dWeight1 = (layers[0] as AffineLayer).dW!!,
                dBias1 = (layers[0] as AffineLayer).dB!!,
                dWeight2 = (layers[2] as AffineLayer).dW!!,
                dBias2 = (layers[2] as AffineLayer).dB!!
        )
    }

    data class GradientData(
            val dWeight1: INDArray, val dBias1: INDArray,
            val dWeight2: INDArray, val dBias2: INDArray
    )

    data class WeightAndBias(
            var weight: INDArray,
            var bias: INDArray
    )

    interface NeuralLayer {
        fun forward(x: INDArray): INDArray
        fun backward(dOut: INDArray): INDArray
    }

    interface NeuralLastLayer {
        fun forward(inX: INDArray, inT: INDArray): Double
        fun backward(): INDArray
    }

    class ReluLayer: NeuralLayer {
        var mask: INDArray? = null

        override fun forward(x: INDArray): INDArray {
            mask = x.gt(0)
            return x.putWhereWithMask(mask, Nd4j.zerosLike(x))
        }

        override fun backward(dOut: INDArray): INDArray {
            return dOut.putWhereWithMask(mask, Nd4j.zerosLike(dOut))
        }
    }

    class AffineLayer(val data: WeightAndBias): NeuralLayer {
        var x: INDArray? = null
        var dW: INDArray? = null
        var dB: INDArray? = null

        override fun forward(inX: INDArray): INDArray {
            x = inX
            return (x!! dot data.weight) + data.bias
        }

        override fun backward(dOut: INDArray): INDArray {
            dW = x!!.transpose() dot dOut
            dB = dOut.sum(0)
            return dOut dot data.weight.transpose()
        }
    }

    class SoftmaxWithLossLayer: NeuralLastLayer {
        var loss: Double? = null
        var y: INDArray? = null
        var t: INDArray? = null

        override fun forward(inX: INDArray, inT: INDArray): Double {
            t = inT
            y = Transforms.softmax(inX)
            loss = crossEntropyError(y!!, t!!)

            return loss!!
        }

        override fun backward(): INDArray {
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
            val batchSize = y.shape()[0]
            //return -(t * Transforms.log(y))

            val delta = 0.0000001
            return -(t * Transforms.log(y + delta)).sumNumber().toDouble() / batchSize
        }
    }
}
