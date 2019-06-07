import org.apache.commons.math3.distribution.NormalDistribution
import java.net.URL
import kotlin.math.pow

// Desmos graph: https://www.desmos.com/calculator/pb4ewmqdvy

val points = URL("https://tinyurl.com/y25lvxug")
        .readText().split(Regex("\\r?\\n"))
        .asSequence()
        .drop(1)
        .filter { it.isNotEmpty() }
        .map { it.split(",") }
        .map { (x,y) -> ObservedPoint(x.toDouble(),y.toDouble()) }
        .toList()

sealed class Point {
    abstract val x: Double
    abstract val y: Double
}

data class ObservedPoint(override val x: Double, override val y: Double): Point()

class Centroid(val index: Int): Point() {
    override var x = 0.0
    override var y = 0.0
}

fun distanceBetween(point1: Point, point2: Point) =
        ((point2.x - point1.x).pow(2) + (point2.y - point1.y).pow(2)).pow(.5)

fun main() {

    val k = 4
    val centroids = (0 until k).map { Centroid(it) }

    var bestLoss = Double.MAX_VALUE

    val normalDistribution = NormalDistribution(0.0, 1.0)

    repeat(100_000) {

        val randomCentroid = centroids.random()

        val xAdjust = normalDistribution.sample()
        val yAdjust = normalDistribution.sample()

        randomCentroid.x += xAdjust
        randomCentroid.y += yAdjust

        val newLoss = points.asSequence()
                .map { pt ->
                    centroids.asSequence().map { distanceBetween(it, pt) }.min()!!.pow(2)
                }.sum()

        if (newLoss < bestLoss) {
            bestLoss = newLoss
        } else {
            randomCentroid.x -= xAdjust
            randomCentroid.y -= yAdjust
        }
    }

    centroids.forEach { println("${it.x},${it.y}") }
}
