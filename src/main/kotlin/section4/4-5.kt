package section4

import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle
import org.deeplearning4j.datasets.fetchers.EmnistDataFetcher
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

fun main() {
    section4_5_2()
}

private fun section4_5_1() {
    val net = TwoLayerNet(784, 100, 10)
    println("w1=${net.weight1.shape().contentToString()}")
    println("b1=${net.bias1.shape().contentToString()}")
    println("w2=${net.weight2.shape().contentToString()}")
    println("b2=${net.bias2.shape().contentToString()}")
}

private fun section4_5_2() {
    val batchSize = 100
    val learningRate = 0.1
    val lossList = mutableListOf<Double>()
    val trainAccList = mutableListOf<Double>()
    val testAccList = mutableListOf<Double>()

    val network = TwoLayerNet(784, 50, 10)

    var index = 0

    val testData = EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, batchSize, false, false, true, 0).next()

    for(epoch in 0 until 5) {
        val trainData = EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, batchSize, false, true, true, 0)

        for(batch in trainData) {
            val bImage = batch.features
            val bLabel = batch.labels

            val grad = network.getGradientFast(bImage, bLabel)
            //println("grad=$grad")

            network.weight1 -= learningRate * grad.dWeight1
            network.bias1 -= learningRate * grad.dBias1
            network.weight2 -= learningRate * grad.dWeight2
            network.bias2 -= learningRate * grad.dBias2

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

class TwoLayerNet(
    inputSize: Int, hiddenSize: Int, outputSize: Int,
    weightInitStd: Double = 0.01) {
    var weight1 = weightInitStd * Nd4j.rand(inputSize, hiddenSize)
    var bias1 = Nd4j.zeros(hiddenSize)
    var weight2 = weightInitStd * Nd4j.rand(hiddenSize, outputSize)
    var bias2 = Nd4j.zeros(outputSize)

    fun predict(x: INDArray): INDArray {
        val a1 = x dot weight1 + bias1
        val z1 = Transforms.sigmoid(a1)
        val a2 = z1 dot weight2 + bias2
        return Transforms.softmax(a2)
    }

    fun loss(x: INDArray, t: INDArray): Double {
        return crossEntropyError(predict(x), t)
    }

    fun accuracy(x: INDArray, t: INDArray): Double {
        val yMax = predict(x).argMax(1)
        val tMax = t.argMax(1)
        return tMax.eq(yMax).castTo(DataType.DOUBLE).sumNumber().toDouble() / x.shape()[0]
    }

    //遅すぎて無理
    fun getGradient(x: INDArray, t: INDArray): GradientData {
        //val lossConst = loss(x, t)
        return GradientData(
            dWeight1 = numericalGradient({ loss(x, t) }, weight1),
            dBias1 = numericalGradient({ loss(x, t) }, bias1),
            dWeight2 = numericalGradient({ loss(x, t) }, weight2),
            dBias2 = numericalGradient({ loss(x, t) }, bias2)
        )
    }

    fun getGradientFast(x: INDArray, t: INDArray): GradientData {
        val a1 = x dot weight1 + bias1
        val z1 = Transforms.sigmoid(a1)
        val a2 = z1 dot weight2 + bias2
        val y = Transforms.sigmoid(a2)

        val dy = (y - t) / x.shape()[0]

        val dz1 = dy dot weight2.transpose()
        val da1 = sigmoidGrad(a1) * dz1

        return GradientData(
            dWeight1 = x.transpose() dot da1,
            dBias1 = da1.sum(0),
            dWeight2 = z1.transpose() dot dy,
            dBias2 = dy.sum(0)
        )
    }

    private fun sigmoidGrad(x: INDArray): INDArray {
        return (1 - Transforms.sigmoid(x)) * Transforms.sigmoid(x)
    }

    private fun crossEntropyError(y: INDArray, t: INDArray): Double {
        val delta = 0.0000001
        return -(t * Transforms.log(y + delta)).sumNumber().toDouble()
    }

    private fun numericalGradient(f: (INDArray) -> Double, x: INDArray): INDArray {
        val h = 1e-4
        val grad = Nd4j.zerosLike(x)

        println("x=$x, ${x.length()}, ${x.shape().contentToString()}")
        for(i in 0 until x.length()) {
            val tmp = x.getDouble(i)
            x.putScalar(i, tmp + h)
            val fxh1 = f(x)

            x.putScalar(i, tmp - h)
            val fxh2 = f(x)

            //println("fxh1=$fxh1, fxh2=$fxh2")
            grad.putScalar(i, (fxh1 - fxh2) / (2 * h))
            x.putScalar(i, tmp)
        }
        println("xgrad:")
        println("$grad")
        return grad
    }

    data class GradientData(
        val dWeight1: INDArray, val dBias1: INDArray,
        val dWeight2: INDArray, val dBias2: INDArray
    )
}
