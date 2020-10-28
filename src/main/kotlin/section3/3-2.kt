package section3

import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Condition
import org.nd4j.linalg.indexing.conditions.Conditions
import org.nd4j.linalg.ops.transforms.Transforms
import util.*

//活性化関数
fun main(args: Array<String>) {
    section3_2_7()
}

fun section3_2_3() {
    println("section3.2.3")
    val xArray = Nd4j.arange(-50.0,50.0)/10
    val stepArray = stepND(xArray)
    val data = mapOf<String, Any>(
            "x" to xArray.data().asDouble(),
            "y" to stepArray.data().asDouble()
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
    }
    val p = ggplot(data) + geom + ggtitle("step")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    println("plotSpec=$plotSpec")
    println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "step")
}

fun section3_2_4() {
    println("section3.2.4")
    val xArray = Nd4j.arange(-50.0,50.0)/10
    val stepArray = sigmoidND(xArray)
    val data = mapOf<String, Any>(
            "x" to xArray.data().asDouble(),
            "y" to stepArray.data().asDouble()
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
    }
    val p = ggplot(data) + geom + ggtitle("sigmoid")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    println("plotSpec=$plotSpec")
    println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "sigmoid")
}

fun section3_2_7() {
    println("section3.2.7")
    val xArray = Nd4j.arange(-50.0,50.0)/10
    val stepArray = reluND(xArray)
    val data = mapOf<String, Any>(
            "x" to xArray.data().asDouble(),
            "y" to stepArray.data().asDouble()
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
    }
    val p = ggplot(data) + geom + ggtitle("relu")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    println("plotSpec=$plotSpec")
    println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "relu")
}

private fun step(x: Double): Int {
    return if(x > 0) 1 else 0
}

private fun stepND(x: INDArray): INDArray {
    val res = x.dup()
    BooleanIndexing.applyWhere(res, Conditions.greaterThan(0),{ 1 }, { 0 })
    return res
}

private fun sigmoidND(x: INDArray): INDArray {
    return 1 / (1 + Transforms.exp(-x))
}

private fun reluND(x: INDArray): INDArray {
    return Transforms.max(x, 0.0)
}
