package util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms


operator fun INDArray.plus(other: INDArray): INDArray = this.add(other)
operator fun INDArray.minus(other: INDArray): INDArray = this.sub(other)
operator fun INDArray.times(other: INDArray): INDArray = this.mul(other)
operator fun INDArray.div(other: INDArray): INDArray = this.div(other)

operator fun INDArray.plus(other: Number): INDArray = this.add(other)
operator fun INDArray.minus(other: Number): INDArray = this.sub(other)
operator fun INDArray.times(other: Number): INDArray = this.mul(other)
operator fun INDArray.div(other: Number): INDArray = this.div(other)

operator fun Number.plus(other: INDArray): INDArray = other.add(this)
operator fun Number.minus(other: INDArray): INDArray = other.sub(this).neg()
operator fun Number.times(other: INDArray): INDArray = other.mul(this)
operator fun Number.div(other: INDArray): INDArray = Transforms.pow(other, -1) * this

operator fun INDArray.unaryMinus(): INDArray = this.neg()
