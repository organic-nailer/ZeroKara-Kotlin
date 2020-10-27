package section1

import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

fun main(args: Array<String>) {
    section1_6_1()
    section1_6_2()
    section1_6_3()
}

private fun section1_6_1() {
    val xArray = Nd4j.arange(0.0,60.0) / 10.0
    val sinArray = Transforms.sin(xArray)
    val data = mapOf<String, Any>(
            "x" to xArray.data().asDouble(),
            "y" to sinArray.data().asDouble()
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
    }
    val p = ggplot(data) + geom + ggtitle("sin")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    println("plotSpec=$plotSpec")
    println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "sin")
}

private fun section1_6_2() {
    val xArray = Nd4j.arange(0.0,60.0) / 10.0
    val sinArray = Transforms.sin(xArray)
    val cosArray = Transforms.cos(xArray)
    val data = mapOf<String, Any>(
            "x" to xArray.data().asDouble() + xArray.data().asDouble(),
            "y" to sinArray.data().asDouble() + cosArray.data().asDouble(),
            "s" to List(xArray.size(1)) { "sin" } + List(xArray.size(1)) { "cos" }
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
        color = "s"
    }
    val p = ggplot(data) + geom + ggtitle("sin&cos")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    println("plotSpec=$plotSpec")
    println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "sin&cos")
}

private fun section1_6_3() {
    FastPlot.showImg("mouse.png")
}
