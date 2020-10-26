import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.datalore.plot.MonolithicAwt
import jetbrains.datalore.vis.svg.SvgSvgElement
import jetbrains.datalore.vis.swing.BatikMapperComponent
import jetbrains.datalore.vis.swing.BatikMessageCallback
import jetbrains.letsPlot.geom.geom_image
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.intern.toSpec
import jetbrains.letsPlot.label.ggtitle
import java.awt.Font
import java.io.File
import javax.imageio.ImageIO
import javax.swing.JFrame
import javax.swing.SwingUtilities
import javax.swing.WindowConstants

object FastPlot {
    private val SVG_COMPONENT_FACTORY_BATIK =
            { svg: SvgSvgElement -> BatikMapperComponent(svg, BATIK_MESSAGE_CALLBACK) }

    private val BATIK_MESSAGE_CALLBACK = object : BatikMessageCallback {
        override fun handleMessage(message: String) {
            println(message)
        }

        override fun handleException(e: Exception) {
            if (e is RuntimeException) {
                throw e
            }
            throw RuntimeException(e)
        }
    }

    private val AWT_EDT_EXECUTOR = { runnable: () -> Unit ->
        // Just invoke in the current thread.
        runnable.invoke()
    }

    fun showPlot(plotSpec: MutableMap<String, Any>, plotSize: DoubleVector?, title: String) {
        SwingUtilities.invokeLater {
            val component =
                    MonolithicAwt.buildPlotFromRawSpecs(plotSpec, plotSize, SVG_COMPONENT_FACTORY_BATIK, AWT_EDT_EXECUTOR) {
                        for (message in it) {
                            println("PLOT MESSAGE: $message")
                        }
                    }

            // Show plot in Swing frame.
            val frame = JFrame(title)
            frame.contentPane.add(component)
            frame.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
            frame.pack()
            frame.isVisible = true
        }
    }

    fun showImg(resourceName: String) {
        val filePath = ClassLoader.getSystemClassLoader().getResource(resourceName) ?: kotlin.run {
            println("$resourceName のなにかがおかしい")
            return
        }
        val file = File(filePath.toURI())
        if(!file.canRead()) {
            println("file ${filePath.toURI()} cannot be read!!")
            return
        }
        val image = ImageIO.read(file) ?: kotlin.run {
            println("${file.name} が読み込めません")
            return
        }
        //println("image${image.type}")

        println("path=${filePath.path}")

        val data = mapOf<String, Any>(
            "x" to doubleArrayOf(0.0),
            "xmax" to doubleArrayOf(image.width.toDouble()),
            "y" to doubleArrayOf(0.0),
            "ymax" to doubleArrayOf(image.height.toDouble())
        )

        val geom = geom_image(data = data ,href = "file://${filePath.path}") {
            xmin = "x"
            xmax = "xmax"
            ymin = "y"
            ymax = "ymax"
        }
        val p = ggplot() + geom + ggtitle(resourceName)

        println("${p.toSpec()}")

        FastPlot.showPlot(p.toSpec(), DoubleVector(500.0, 500.0), "image")
    }
}