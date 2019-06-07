import kotlin.math.exp
import kotlin.math.ln

// Helpful reference
// https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
// https://github.com/joelgrus/data-science-from-scratch/blob/master/code-python3/naive_bayes.py


// `k` is a probability estimate a non-spam word could actually be in spam
// we never want a non-spam word to drag a message's spam probability down to 0
val k = .5

// globally all emails in my population, spam or not spam
val emailPopulation = listOf(
		CategorizedEmail("Hey there! I thought you might find this interesting. Click here.", isSpam = true),
		CategorizedEmail("Get viagra for a discount as much as 90%", isSpam = true),
		CategorizedEmail("Viagra prescription for less", isSpam = true),
		CategorizedEmail("Even better than Viagra, try this new prescription drug", isSpam = true),

		CategorizedEmail("Hey, I left my phone at home. Email me if you need anything. I'll be in a meeting for the afternoon.", isSpam = false),
		CategorizedEmail("Please see attachment for notes on today's meeting. Interesting findings on your market research.", isSpam = false),
		CategorizedEmail("An item on your Amazon wish list received a discount", isSpam = false),
		CategorizedEmail("Your prescription drug order is ready", isSpam = false),
		CategorizedEmail("Your Amazon account password has been reset", isSpam = false),
		CategorizedEmail("Your Amazon order", isSpam = false)
)


// words with their probability metrics
val wordsWithProbability = emailPopulation.asSequence()
        .flatMap { it.body.splitWords() }
		.distinct()
		.map { word ->
			WordProbability(word = word)
		}.toList()


fun main(args: Array<String>) {


    // Test two incoming messages
    val message1 = "discount viagra wholesale, hurry while this offer lasts"
	println("\r\nScore for an email containing message: \"$message1\"")
	spamScoreForMessage(message1).also {
		println(it)
	}

    val message2 = "interesting meeting on amazon cloud services discount program"
	println("\r\nScore for an email containing message: \"$message2\"")
	spamScoreForMessage(message2).also {
		println(it)
	}


    // Some reporting
    println("\r\nSpammiest Words")
    wordsWithProbability.asSequence()
            .filter { it.probabilityWordAppearsInSpam > 0.0 }
            .sortedByDescending { it.probabilityWordAppearsInSpam }
            .take(5)
            .forEach {
                println("${it.word} ${it.probabilityWordAppearsInSpam}")
            }

    println("\r\nHammiest Words")
    wordsWithProbability.asSequence()
            .filter { it.probabilityWordAppearsInSpam > 0.0 }
            .sortedByDescending { it.probabilityWordAppearsInHam }
            .take(5)
            .forEach {
                println("${it.word} ${it.probabilityWordAppearsInHam}")
            }
}

fun spamScoreForMessage(message: String): Double {
    val distinctMsgWords = message.splitWords().toHashSet()

    val probIfSpam = wordsWithProbability.asSequence().map {
        if (it.word in distinctMsgWords) {
            ln(it.probabilityWordAppearsInSpam)
        } else {
            ln(1.0 - it.probabilityWordAppearsInSpam)
        }
    }.sum().let(::exp)

    val probIfNotSpam = wordsWithProbability.asSequence().map {
        if (it.word in distinctMsgWords) {
            ln(it.probabilityWordAppearsInHam)
        } else {
            ln(1.0 - it.probabilityWordAppearsInHam)
        }
    }.sum().let(::exp)

    return probIfSpam / (probIfSpam + probIfNotSpam)
}

data class CategorizedEmail(val body: String, val isSpam: Boolean) {

	val words = body.split(Regex("\\s")).asSequence().map {
		it.replace(Regex("[^A-Za-z]"),"").toLowerCase()
	}.distinct().toSet()
}

class WordProbability(val word: String) {

	// Pr(W|S), probability word appears in spam message
	val probabilityWordAppearsInSpam =
            (k + emailPopulation.count { it.isSpam && word in it.words }.toDouble()) /
                    ((2 * k) + emailPopulation.count { it.isSpam }.toDouble())


	//Pr(W|H), probability word appears in ham message
	val probabilityWordAppearsInHam =
            (k + emailPopulation.count { !it.isSpam && word in it.words }.toDouble()) /
                    ((2 * k) + emailPopulation.count { !it.isSpam }.toDouble())

}


fun String.splitWords() =  split(Regex("\\s")).asSequence()
    .map { it.replace(Regex("[^A-Za-z]"),"").toLowerCase() }
    .filter { it.isNotEmpty() }
