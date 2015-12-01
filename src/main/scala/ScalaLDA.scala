import org.apache.spark.mllib.clustering.{LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.{SparseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

object ScalaLDA {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("LDA test")
      .setMaster("local")
      .set("spark.default.parallelism", "2")

    val sc = new SparkContext(conf)

    // Load and parse the data
    val data = sc.textFile("/Users/joris/Developer/sherlock/analyzing-corpora/data/VraagTextCorpus.mm")
    val header = data.take(2)(1).split(' ')
    val n_docs = header(0).toLong
    val n_words = header(1).toInt
    val n_counts = header(2).toLong
    val sparseRows : RDD[(Long, (Int, Double))] = data
      .mapPartitionsWithIndex((idx, part) => if (idx == 0) part.drop(2) else part)
      .map(s => {
        val values = s.trim.split(' ')
        (values(0).toLong, (values(1).toInt, values(2).toDouble))
      })

    // Index documents by document ID
    val corpus : RDD[(Long, Vector)] = sparseRows
      .groupByKey()
      .map(x => {
        val doc_id = x._1
        val iter = x._2
        val cts = Array[Double](iter.size)
        val wds = Array[Int](iter.size)
        var i = 0
        var ct : (Int, Double) = null
        for (ct <- iter) {
          wds(i) = ct._1
          cts(i) = ct._2
          i += 1
        }
        (doc_id, new SparseVector(n_words, wds, cts))
      })

    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA().setK(3).run(corpus)

    // Output topics. Each is a distribution over words (matching word count vectors)
    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")
    val topics = ldaModel.topicsMatrix
    for (topic <- Range(0, 3)) {
      print("Topic " + topic + ":")
      for (word <- Range(0, ldaModel.vocabSize)) { print(" " + topics(word, topic)); }
      println()
    }

    // Save and load model.
    ldaModel.save(sc, "myLDAModel")
    val sameModel = DistributedLDAModel.load(sc, "myLDAModel")
  }
}
