import org.apache.commons.math3.distribution.NormalDistribution
import java.net.URL
import java.util.concurrent.ThreadLocalRandom
import kotlin.math.exp
import kotlin.math.ln


// See graph
// https://www.desmos.com/calculator/6cb10atg3l

// Helpful Resources:
// StatsQuest on YouTube: https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
// Brandon Foltz on YouTube: https://www.youtube.com/playlist?list=PLIeGtxpvyG-JmBQ9XoFD4rs-b3hkcX7Uu

data class Observation(val independent: Double, val probability: Double)

val trainingData = URL("https://tinyurl.com/y2cocoo7")
        .readText().split(Regex("\\r?\\n"))
        .asSequence()
        .drop(1)
        .filter { it.isNotEmpty() }
        .map { it.split(",") }
        .map { (independent,probability) -> Observation(independent.toDouble(), probability.toDouble()) }
        .toList()

fun main() {
    /*trainingData.forEach {
        println("${it.independent},${if (it.probability) 1 else 0}")
    }*/

    var bestLikelihood = -10_000_000.0

    // use hill climbing for optimization
    val normalDistribution = NormalDistribution(0.0, 1.0)

    var b0 = .01
    var b1 = .01

    fun predictProbability(independent: Double) = 1.0 / (1 + exp(-(b0 + b1*independent)))

    repeat(10000) {

        val selectedBeta = ThreadLocalRandom.current().nextInt(0,2)
        val adjust = normalDistribution.sample()

        // make random adjustment to two of the colors
        when {
            selectedBeta == 0 -> b0 += adjust
            selectedBeta == 1 -> b1 += adjust
        }

        // calculate maximum likelihood
        val trueEstimates = trainingData.asSequence()
                .filter { it.probability == 1.0 }
                .map { ln(predictProbability(it.independent)) }
                .sum()

        val falseEstimates = trainingData.asSequence()
                .filter { it.probability == 0.0 }
                .map { ln(1 - predictProbability(it.independent)) }
                .sum()

        val likelihood = trueEstimates + falseEstimates

        if (bestLikelihood < likelihood) {
            bestLikelihood = likelihood
        } else {
            // revert if no improvement happens
            when {
                selectedBeta == 0 -> b0 -= adjust
                selectedBeta == 1 -> b1 -= adjust
            }
        }
    }
    println("1.0 / (1 + exp(-($b0 + $b1*x))")
    println("BEST LIKELIHOOD: $bestLikelihood")
}
