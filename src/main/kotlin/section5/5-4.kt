package section5

import java.lang.Exception

fun main() {
    section5_4_2()
}

private fun section5_4_1() {
    val applePrice = 100
    val appleNum = 2
    val tax = 1.1

    val mulAppleLayer = MulLayer()
    val mulTaxLayer = MulLayer()

    val appleSumPrice = mulAppleLayer.forward(applePrice.toDouble(), appleNum.toDouble())
    val price = mulTaxLayer.forward(appleSumPrice, tax)
    println("price=$price")

    val dPrice = 1
    val (dAppleSumPrice, dTax) = mulTaxLayer.backward(dPrice.toDouble())
    val (dApplePrice, dAppleNum) = mulAppleLayer.backward(dAppleSumPrice)

    println("$dApplePrice, $dAppleNum, $dTax")
}

class MulLayer {
    var x: Double? = null
    var y: Double? = null

    fun forward(inX: Double, inY: Double): Double {
        x = inX
        y = inY
        return inX * inY
    }

    fun backward(dOut: Double): BackLayerData {
        return BackLayerData(
                dX = dOut * (y ?: throw Exception("データがありません")),
                dY = dOut * (x ?: throw Exception("データがありません"))
        )
    }

    data class BackLayerData(val dX: Double, val dY: Double)
}

private fun section5_4_2() {
    val applePrice = 100
    val appleNum = 2
    val orangePrice = 150
    val orangeNum = 3
    val tax = 1.1

    val mulAppleLayer = MulLayer()
    val mulOrangeLayer = MulLayer()
    val addAppleOrangeLayer = AddLayer()
    val mulTaxLayer = MulLayer()

    val appleSumPrice = mulAppleLayer.forward(applePrice.toDouble(), appleNum.toDouble())
    val orangeSumPrice = mulOrangeLayer.forward(orangePrice.toDouble(), orangeNum.toDouble())
    val allPrice = addAppleOrangeLayer.forward(appleSumPrice, orangeSumPrice)
    val price = mulTaxLayer.forward(allPrice, tax)
    println("price=$price")

    val dPrice = 1
    val (dAllPrice, dTax) = mulTaxLayer.backward(dPrice.toDouble())
    val (dAppleSumPrice, dOrangeSumPrice) = addAppleOrangeLayer.backward(dAllPrice)
    val (dOrangePrice, dOrangeNum) = mulOrangeLayer.backward(dOrangeSumPrice)
    val (dApplePrice, dAppleNum) = mulAppleLayer.backward(dAppleSumPrice)

    println("$dApplePrice, $dAppleNum, $dTax")
    println("$dOrangePrice, $dOrangeNum")
}

class AddLayer {
    fun forward(inX: Double, inY: Double): Double {
        return inX + inY
    }

    fun backward(dOut: Double): BackLayerData {
        return BackLayerData(
                dX = dOut,
                dY = dOut
        )
    }

    data class BackLayerData(val dX: Double, val dY: Double)
}
