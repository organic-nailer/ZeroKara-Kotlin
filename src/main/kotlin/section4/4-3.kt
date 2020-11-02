package section4

import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import java.util.function.Function
import kotlin.math.pow
import util.*

fun main() {
    section4_3_2()
}

private fun section4_3_2() {
    val xArray = Nd4j.arange(0.0,200.0) / 10
    val yArray = function1ND(xArray)

    val data = mapOf<String, Any>(
        "x" to xArray.data().asDouble(),
        "y" to yArray.data().asDouble()
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
    }
    val p = ggplot(data) + geom + ggtitle("function1")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    println("plotSpec=$plotSpec")
    println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "function1")
}
private fun numericalDiff(f: (Double) -> Double, x: Double): Double {
    val h = 10e-4
    return(f(x+h) - f(x-h)) / 2*h
}
//0.01x^2+0.1x
private fun function1(x: Double): Double {
    return 0.01 * x.pow(2) + 0.1 * x
}
private fun function1ND(x: INDArray): INDArray {
    return 0.01 * Transforms.pow(x, 2) + 0.1 * x
}

//(x_0)^2+(x_1)^2
private fun function2ND(x: INDArray): Double {
    return Transforms.pow(x, 2).sumNumber().toDouble()
}
