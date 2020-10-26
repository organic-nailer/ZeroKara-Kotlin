import jetbrains.datalore.base.geometry.DoubleVector
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms.sin;
import jetbrains.letsPlot.geom.geom_line
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle

fun main(args: Array<String>) {
//    println("Hello, World!")
//    playKMath()
    letsPlot()
}

fun playKMath() {
    operator fun INDArray.plus(other: INDArray): INDArray = this.add(other)
    operator fun INDArray.minus(other: INDArray): INDArray = this.sub(other)
    operator fun INDArray.times(other: Number): INDArray = this.mul(other)
    operator fun INDArray.div(other: Number): INDArray = this.div(other)
    val x = Nd4j.arange(0.0,60.0) / 10.0
    val y = sin(x)
    println("x=$x")
    println("y=$y")
    println("x+y=${x + y}")
}

fun letsPlot() {
    operator fun INDArray.plus(other: INDArray): INDArray = this.add(other)
    operator fun INDArray.minus(other: INDArray): INDArray = this.sub(other)
    operator fun INDArray.times(other: Number): INDArray = this.mul(other)
    operator fun INDArray.div(other: Number): INDArray = this.div(other)
    val xArray = Nd4j.arange(0.0,6.0) / 10.0
    val yArray = sin(xArray)
    val data = mapOf<String, Any>(
            "x" to xArray.data().asDouble() + xArray.data().asDouble(),
            "y" to yArray.data().asDouble() + (yArray * 0.5).data().asDouble(),
            "s" to List(xArray.size(1)) { "A" } + List(xArray.size(1)) { "B" }
    )

    // Create plot specs using Lets-Plot Kotlin API
    val geom = geom_line() {
        x = "x"
        y = "y"
        color = "s"
    }
    val p = ggplot(data) + geom + ggtitle("sin")

    // Create Swing Panel showing the plot.
    val plotSpec = p.toSpec()
    val plotSize = DoubleVector(1000.0, 1000.0)

    println("plotSpec=$plotSpec")
    println("plotSize=$plotSize")

    FastPlot.showPlot(plotSpec, plotSize, "minimal")
}
