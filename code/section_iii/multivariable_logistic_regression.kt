import javafx.scene.paint.Color
import org.apache.commons.math3.distribution.NormalDistribution
import java.net.URL
import kotlin.math.exp
import kotlin.math.ln

var b0 = .01 // constant
var b1 = .01 // red beta
var b2 = .01 // green beta
var b3 = .01 // blue beta


fun predictProbability(color: Color) = 1.0 / (1 + exp(-(b0 + b1 * color.red + b2 * color.green + b3 * color.blue)))

class LabeledColor(val color: Color, val darkFontIndicator: Double) {
    val red get() = color.red
    val green get () = color.green
    val blue get() = color.blue
}

val inputs = URL("https://tinyurl.com/y2qmhfsr")
        .readText().split(Regex("\\r?\\n"))
        .asSequence()
        .drop(1)
        .filter { it.isNotEmpty() }
        .map { it.split(",") }
        .map { (r,g,b,s) ->  LabeledColor(Color.rgb(r.toInt(), g.toInt(), b.toInt()), s.toDouble()) }
        .toList()

fun main() {


    var bestLikelihood = -10_000_000.0

    // use hill climbing for optimization
    val normalDistribution = NormalDistribution(0.0, 1.0)

    b0 = .01 // constant
    b1 = .01 // red beta
    b2 = .01 // green beta
    b3 = .01 // blue beta

    // 1 = DARK FONT, 0 = LIGHT FONT

    repeat(50_000) {

        val selectedBeta = ThreadLocalRandom.current().nextInt(0,4)
        val adjust = normalDistribution.sample()

        // make random adjustment to two of the colors
        when {
            selectedBeta == 0 -> b0 += adjust
            selectedBeta == 1 -> b1 += adjust
            selectedBeta == 2 -> b2 += adjust
            selectedBeta == 3 -> b3 += adjust
        }

        // calculate maximum likelihood
        val darkEstimates = inputs.asSequence()
                .filter { it.darkFontIndicator == 1.0 }
                .map { ln(predictProbability(it.color)) }
                .sum()

        val lightEstimates = inputs.asSequence()
                .filter { it.darkFontIndicator == 0.0 }
                .map { ln(1 - predictProbability(it.color)) }
                .sum()

        val likelihood = darkEstimates + lightEstimates

        if (bestLikelihood < likelihood) {
            bestLikelihood = likelihood
        } else {
            // revert if no improvement happens
            when {
                selectedBeta == 0 -> b0 -= adjust
                selectedBeta == 1 -> b1 -= adjust
                selectedBeta == 2 -> b2 -= adjust
                selectedBeta == 3 -> b3 -= adjust
            }
        }
    }

    println("1.0 / (1 + exp(-($b0 + $b1*R + $b2*G + $b3*B))")
    println("BEST LIKELIHOOD: $bestLikelihood")


    fun predictFontShade(color: Color)= if (predictProbability(color) > .5) "DARK" else "LIGHT"

    while(true) {
        println("Predict light or dark font. Input values R,G,B:")
        val inputColor = readLine()!!.split(",").map { it.trim().toInt() }.let { (r,g,b) -> Color.rgb(r,g,b) }
        println(predictFontShade(inputColor))
    }
}