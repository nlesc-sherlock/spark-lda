import org.apache.spark.mllib.clustering.{EMLDAOptimizer, OnlineLDAOptimizer, LDA, DistributedLDAModel}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors, Matrix}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
import scala.collection.mutable

import scopt.OptionParser

object ScalaLDASeq {
  private case class Params(
     input: String = null,
     output: String = null,
     fraction: Double = 0.01,
     k: Int = 20,
     maxIterations: Int = 10,
     docConcentration: Double = -1,
     topicConcentration: Double = -1,
     vocabSize: Int = 10000,
     stopwordFile: String = "",
     algorithm: String = "em")

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("ScalaLDASeq") {
      head("ScalaLDASeq: an LDA app for plain text data.")
      opt[Double]("fraction")
        .text(s"Fraction of the sequence file. default: ${defaultParams.fraction}")
        .action((x, c) => c.copy(fraction = x))
      opt[Int]("k")
        .text(s"number of topics. default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations of learning. default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[Double]("docConcentration")
        .text(s"amount of topic smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.docConcentration}")
        .action((x, c) => c.copy(docConcentration = x))
      opt[Double]("topicConcentration")
        .text(s"amount of term (word) smoothing to use (> 1.0) (-1=auto)." +
          s"  default: ${defaultParams.topicConcentration}")
        .action((x, c) => c.copy(topicConcentration = x))
      opt[String]("algorithm")
        .text(s"inference algorithm to use. em and online are supported." +
          s" default: ${defaultParams.algorithm}")
        .action((x, c) => c.copy(algorithm = x))
      arg[String]("<input>")
        .text("input path to the sequence file of corpus.")
        .required()
        .action((x, c) => c.copy(input = x))
      arg[String]("<output>")
        .text("output csv file with a matrix with word frequencies, with rows as topics and columns as word id")
        .required()
        .action((x, c) => c.copy(output = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      parser.showUsageAsError
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val n_topics = 100

    // Look at configuration to determine parallelism and run node
    val conf = new SparkConf()
      .setAppName("ScalaLDASeq")
      .set("spark.default.parallelism", "8")
    val sc = new SparkContext(conf)

    // Cache for faster processing
    val (corpus, corpusLookup, vocabArray, actualNumTokens) =
      preprocess(sc, params.input, params.fraction, params.vocabSize, params.stopwordFile)
    
    val cachedCorpus = corpus.cache()

    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      // add (1.0 / actualCorpusSize) to MiniBatchFraction be more robust on tiny datasets.
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / cachedCorpus.count())
      case _ => throw new IllegalArgumentException(
        s"Only em, online are supported but got ${params.algorithm}.")
    }

    // Cluster the documents into three topics using LDA
    val ldaModel = new LDA()
      .setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .run(cachedCorpus)

    // Save as CSV for other groups
    saveAsCsv(sc, params.output, ldaModel.topicsMatrix)

    // Save as parquet for further processing
    //    ldaModel.save(sc, "myLDAModel")
  }

  /**
   * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
   * @return (corpus, vocabulary as array, total token count in corpus)
   */
  private def preprocess(
      sc: SparkContext,
      path: String,
      fraction: Double,
      vocabSize: Int,
      stopwordFile: String): (RDD[(Long, Vector)], RDD[(Long, String)], Array[String], Long) = {
    // Get dataset of document texts
    // One document per line in each text file. If the input consists of many small files,
    // this can result in a large number of small partitions, which can degrade performance.
    // In this case, consider using coalesce() to create fewer, larger partitions.
    val data = sc.sequenceFile[String, String](path).sample(false, fraction)

    val lookup = data.zipWithIndex().map{ case ((path, text), id) => id -> path}

    val textRDD = data.map{ case (path, text) => text}

    // Split text into words
    val tokenizer = new SimpleTokenizer(sc, stopwordFile)
    val tokenized: RDD[(Long, IndexedSeq[String])] = textRDD.zipWithIndex().map { case (text, id) =>
      id -> tokenizer.getWords(text)
    }

    tokenized.cache()
    // Counts words: RDD[(word, wordCount)]
    val wordCounts: RDD[(String, Long)] = tokenized
      .flatMap { case (_, tokens) => tokens.map(_ -> 1L) }
      .reduceByKey(_ + _)
    wordCounts.cache()
    val fullVocabSize = wordCounts.count()
    // Select vocab
    //  (vocab: Map[word -> id], total tokens after selecting vocab)
    val (vocab: Map[String, Int], selectedTokenCount: Long) = {
      val tmpSortedWC: Array[(String, Long)] = if (vocabSize == -1 || fullVocabSize <= vocabSize) {
        // Use all terms
        wordCounts.collect().sortBy(-_._2)
      } else {
        // Sort terms to select vocab
        wordCounts.sortBy(_._2, ascending = false).take(vocabSize)
      }
      (tmpSortedWC.map(_._1).zipWithIndex.toMap, tmpSortedWC.map(_._2).sum)
    }
    val documents = tokenized.map { case (id, tokens) =>
      // Filter tokens by vocabulary, and create word count vector representation of document.
      val wc = new mutable.HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices = wc.keys.toArray.sorted
      val values = indices.map(i => wc(i).toDouble)
      val sb = Vectors.sparse(vocab.size, indices, values)
      (id, sb)
    }
    val vocabArray = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabArray(i) = term }
    (documents, lookup, vocabArray, selectedTokenCount)
  }
  def saveAsCsv(sc: SparkContext, filename: String, matrix: Matrix) = {
    sc.makeRDD(Range(0, matrix.numCols), 1)
      .map(i => {
        var str = ""
        for (j : Int <- Range(0, matrix.numRows)) {
          if (j > 0) str += ","
          str += matrix(j, i)
        }
        str
      })
      .saveAsTextFile(filename)
  }

  def parseCustomCsv(data: RDD[String]) : RDD[(Long, Vector)] = {
    // Parse the header and disregard the first line (comment)
    val header = data.take(2)(1).split(' ')
    // The second line contains the number of values
    val n_words = header(0).toInt
    val n_docs = header(1).toLong

    // Parse each row except the header separately
    return data
      .mapPartitionsWithIndex((idx, part) => if (idx == 0) part.drop(2) else part)
      .filter(!_.isEmpty())
      .map(s => {
        val values = s.trim.split(';')
        val doc_id = values(0).toLong
        val words = values(1).split(',').map(_.toInt)
        val counts = values(2).split(',').map(_.toDouble)
        (doc_id, new SparseVector(n_words, words, counts))
      })
  }

  def parseMatrixMarket(data: RDD[String]) : RDD[(Long, Vector)] = {
    // Parse the header and disregard the first line (%%MatrixMarket etc)
    val header = data.take(2)(1).split(' ')
    // The second line contains the number of values
    val n_docs = header(0).toLong
    val n_words = header(1).toInt
    val n_counts = header(2).toLong

    // Parse each row except the header separately
    val sparseRows : RDD[(Long, (Int, Double))] = data
      .mapPartitionsWithIndex((idx, part) => if (idx == 0) part.drop(2) else part)
      .map(s => {
        val values = s.trim.split(' ')
        (values(0).toLong - 1, (values(1).toInt - 1, values(2).toDouble))
      })

    // Combine the rows by document ID, and store the words and counts in Arrays
    return sparseRows
      .groupByKey()
      .map(x => {
        val doc_id = x._1
        val iter = x._2
        val cts = Array[Double]().padTo(iter.size, 0.0)
        val wds = Array[Int]().padTo(iter.size, 0)
        var i = 0
        var ct : (Int, Double) = null
        for (ct <- iter) {
          wds(i) = ct._1
          cts(i) = ct._2
          i += 1
        }
        // TODO: Sparse vector assumes that wds is sorted, we don't check or enforce that yet
        (doc_id, new SparseVector(n_words, wds, cts))
      })
  }
}
