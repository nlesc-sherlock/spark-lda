import java.io.{DataOutput, DataInput}
import java.util.Properties
import javax.mail.Session
import org.apache.commons.mail.util.{MimeMessageUtils, MimeMessageParser}
import org.apache.hadoop.io.Writable

import scala.util.matching.Regex
import collection.JavaConversions._

import org.apache.lucene.analysis.core.StopAnalyzer

import scala.collection.mutable.ListBuffer
import edu.stanford.nlp.ling.CoreAnnotations.{PartOfSpeechAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.util.CoreMap
import edu.stanford.nlp.ling.CoreLabel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

import scopt.OptionParser

case class DictionaryItem(id: Int, word: String, occurrences: Long)
case class ParsedEmail(id: Long, path: String, message: MimeMessageParser)
case class EmailContents(id: Long, contents: String)
case class TokenizedDocument(id: Long, tokens: Array[String])
case class RawDocument(id: Long, path: String, document: String)
case class BagOfWords(id: Long, words: Array[Int], counts: Array[Int])
case class EmailMetadata(var id: Long, var path: String, var message_id: String, var user: String,
                         var date: java.util.Date, var from: String, var to: Array[String], var cc: Array[String],
                         var bcc: Array[String], var subject: String) extends Writable {
  override def write(out: DataOutput): Unit = {
    out.writeLong(id)
    out.writeUTF(path)
    out.writeUTF(message_id)
    out.writeUTF(user)
    out.writeLong(date.getTime)
    out.writeUTF(from)
    writeUTFArray(out, to)
    writeUTFArray(out, cc)
    writeUTFArray(out, bcc)
    out.writeUTF(subject)
  }

  private def readUTFArray(in: DataInput): Array[String] = Array[String]().padTo(in.readInt, null).map(v => in.readUTF)
  private def writeUTFArray(out: DataOutput, arr: Array[String]): Unit = {
    out.writeInt(arr.size)
    arr.map(s => out.writeUTF(s))
  }

  override def readFields(in: DataInput): Unit = {
    id = in.readLong
    path = in.readUTF
    message_id = in.readUTF
    user = in.readUTF
    date = new java.util.Date(in.readLong)
    from = in.readUTF
    to = readUTFArray(in)
    cc = readUTFArray(in)
    bcc = readUTFArray(in)
    subject = in.readUTF
  }
}

object EmailParser {
  private case class Params(input: String = null,
                            output: String = null,
                            dictionary: String = null,
                            metadata: String = null,
                            above: Double = 0.5,
                            below: Int = 5,
                            keep_n: Int = 1000000)

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("EmailParser") {
      head("EmailParser: parses input files and produces tokens")
      opt[Double]("above")
        .text(s"remove all words occurring in more than 'above' FRACTION of documents. default: ${defaultParams.above}")
        .action((x, c) => c.copy(above = x))
      opt[Int]("below")
        .text(s"remove all words occurring in less than 'below' NUMBER of documents. default: ${defaultParams.below}")
        .action((x, c) => c.copy(below = x))
      opt[Int]("keep_n")
        .text(s"keep the n most frequent words, after the previous step. default: ${defaultParams.keep_n}")
        .action((x, c) => c.copy(keep_n = x))
      opt[String]("metadata")
        .text("output metadata sequence file with (document id, EmailMetadata)")
        .action((x, c) => c.copy(metadata = x))
      arg[String]("<input>")
        .text("input directory with a number of plain text emails.")
        .required()
        .action((x, c) => c.copy(input = x))
      arg[String]("<dictionary>")
        .text("output dictionary file, tab separated. Each line is " +
          "word_id<TAB>word<TAB>occurrences")
        .required()
        .action((x, c) => c.copy(dictionary = x))
      arg[String]("<output>")
        .text("plain text output file of one parsed document per line" +
              "Each text file line holds 1 document. The first line is a comment, the second line is " +
              "[number of documents]<space>[number of words] and the other lines are ordered as " +
              "doc_id;word_id_1,...,word_id_n;word_count_1,...,word_count_n")
        .required()
        .action((x, c) => c.copy(output = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }
  def run(params: Params) = {
    // Look at configuration to determine parallelism and run node
    val conf = new SparkConf()
      .setAppName("EmailParser")
      .set("spark.default.parallelism", "8")
    val sc = new SparkContext(conf)

    // Sequence file of path -> contents
    // val data : RDD[(String, String)] = sc.sequenceFile(params.input)
    // Testing with single directory
    val data : RDD[(String, String)] = sc.wholeTextFiles(params.input)

    val documents = data
      .zipWithIndex()
      .map(v => RawDocument(v._2, v._1._1, v._1._2))

    val emails = parseEmail(documents).cache()
    val emailContents = emails.map(email => {
      if (email.message.hasPlainContent) {
        EmailContents(email.id, email.message.getPlainContent)
      } else {
        EmailContents(email.id, email.message.getHtmlContent)
      }
    })

    if (params.metadata != null) {
      val metadata = emails.map(email => {
        var date = email.message.getMimeMessage.getSentDate
        if (date == null) {
          date = email.message.getMimeMessage.getReceivedDate
        }
        val split_path = email.path.split("[/\\\\]")
        // This is purely a heuristic, a user may not be the first element of the path.
        val user = if (split_path(0) == ".") split_path(1) else split_path(0)
        val from = email.message.getFrom
        val to = email.message.getTo.toArray(Array[String]().padTo(email.message.getTo.size, null))
        val cc = email.message.getCc.toArray(Array[String]().padTo(email.message.getCc.size, null))
        val bcc = email.message.getBcc.toArray(Array[String]().padTo(email.message.getBcc.size, null))
        val subject = email.message.getSubject
        val message_id = email.message.getMimeMessage.getMessageID
        (email.id, EmailMetadata(email.id, email.path, message_id, user, date, from, to, cc, bcc, subject))
      }).saveAsSequenceFile(params.metadata)
    }

    val filtered = filter(emailContents)
    val tokens = tokenizeStanford(filtered)
    var dictionary = generateDictionary(tokens)
    dictionary = filterDictionary(dictionary, params.above, params.below, params.keep_n)
    val n_words = dictionary.count()

    dictionary
      .sortBy(item => item.id)
      .map(item => s"${item.id}\t${item.word}\t${item.occurrences}")
      .saveAsTextFile(params.dictionary)

    val bow = bagOfWords(tokens, dictionary)
    val n_docs = bow.count()
    bow
      .sortBy(d => d.id)
      .map(d => s"${d.id};${d.words.mkString(",")};${d.counts.mkString(",")}") // stringify
      .mapPartitionsWithIndex((idx, iter) => { // prepend header
        if (idx == 0) {
          Array[String]("# comment", s"${n_words} ${n_docs}").iterator ++ iter
        } else {
          iter
        }
      })
      .saveAsTextFile(params.output) // write

//    val bagOfWords = tokens.map(doc => {
//      val tokenMap = mutable.Map[String, Int]()
//      for (token <- doc.tokens) {
//        tokenMap(token) = tokenMap.getOrElse(token, 0) + 1
//      }
//      val words = ListBuffer[Int]()
//      val counts = ListBuffer[Int]()
//      for ((word, count) <- tokenMap) {
//        words += dictionary.
//      }
//    })
  }

  def bagOfWords(tokens : RDD[TokenizedDocument], dictionary : RDD[DictionaryItem]): RDD[BagOfWords] = {
    val wordLookup = dictionary.map(item => (item.word, item.id))
    val documentWords = tokens.flatMap(doc => doc.tokens.map((_, doc.id)))

    documentWords.join(wordLookup)
      .map(occurrence => ((occurrence._2._1, occurrence._2._2), 1)) // ((document, word_id), 1)
      .reduceByKey((x, y) => x + y) // ((document, word_id), count)
      .map(occurrence => (occurrence._1._1, (occurrence._1._2, occurrence._2))) // (document, (word_id, count))
      .groupByKey() // (document, Iterator[(word_id, count)])
      .map(bow => {
      val b = bow._2.toArray.sortBy(v => v._1)
      val words = ListBuffer[Int]()
      val counts = ListBuffer[Int]()
      for ((word_id, count) <- b) {
        words.append(word_id)
        counts.append(count)
      }
      BagOfWords(bow._1, words.toArray, counts.toArray)
    })
  }
  def parseEmail(data: RDD[RawDocument]) : RDD[ParsedEmail] = {
    data.map( raw => {
      val s : Session = Session.getDefaultInstance(new Properties())
      val m = MimeMessageUtils.createMimeMessage(s, raw.document)
      val p = new MimeMessageParser(m)
      p.parse()
      ParsedEmail(raw.id, raw.path, p)
    })
  }

  def filter(emails: RDD[EmailContents]) : RDD[EmailContents] = {
    val original_pattern = new Regex("-+\\s*[Oo]riginal\\s*[Mm]essage|[Ff]orwarded\\s+by\\s+")
    // replace
    // > forward/reply text
    // with a line ending
    val forward_pattern = new Regex("[\\r\\n]+>[^\\r\\n]*[\\r\\n]+")
    // replace <tags>, =D0 mime tags and urls with a space
    val html_pattern = new Regex("<[^<]+?>|=\\d\\d|http\\S+|www\\S+")
    // replace misc symbols with single space
    val symbol_empty_pattern = new Regex("\\s[-\"`=%><#_~+*/\\\\\\[\\]|{}]+|[-\"`=%><#_~+*/\\\\\\[\\]|{}]+\\s|[\"`=%><#_~+*/\\\\\\[\\]|{}]+")
    // replace by-sentence tokens ,
    val symbol_comma_pattern = new Regex("[();&]+")
    // Don't match 8:00 or 100,000,
    // otherwise replace something:else or something,else with something, else
    val symbol_embedded_comma_pattern = new Regex("([^0-9,:])[,:]+([^0-9])")
    // replace end of line tokens with .
    val symbol_end_pattern = new Regex("[?!]+")
    // Don't match:
    // p.m., p.s., e.g.
    // 0.1, 100.00
    // otherwise replace something.else with something. else
    val symbol_embedded_end_pattern = new Regex("(\\S[^0-9.])\\.+([^0-9 \t\n\r][^.])")

    emails
      .map(email => {
        var text = email.contents
        text = original_pattern.findFirstMatchIn(text) match {
          case Some(m) => text.substring(0, m.start)
          case None => text
        }
        text = forward_pattern.replaceAllIn(text, "\n")
        text = html_pattern.replaceAllIn(text, " ")
        text = symbol_empty_pattern.replaceAllIn(text, " ")
        text = symbol_comma_pattern.replaceAllIn(text, ", ")
        text = symbol_embedded_comma_pattern.replaceAllIn(text, "$1, $2")
        text = symbol_end_pattern.replaceAllIn(text, ". ")
        text = symbol_embedded_end_pattern.replaceAllIn(text, m => "$1. $2")
        EmailContents(email.id, text)
      })
  }

  def tokenizeStanford(emails: RDD[EmailContents]) : RDD[TokenizedDocument] = {
    val nlp_tags = Set(
      // None, "(", ")", ",", ".", // extras
      // "CC", // conjunction
      // "CD", // cardinal (numbers)
      // "DT", // determiner (de, het)
      "FW",  // foreign word
      // "IN", //conjunction
      "JJ",  // adjectives -- // "JJR", "JJS",
      // "MD", // Modal verb
        "NN", "NNP", "NNPS", "NNS",  // Nouns
      // "PRP", // Pronouns -- // "PRP$",
      "RB",  // adverb
      "RP",  // adverb
      // "SYM", // Symbol
      // "TO", // infinitival to
      // "UH", // interjection
      "VB", "VBD", "VBG", "VBN", "VBP", "VBZ" // Verb forms
    )
    val stopwords : Set[String] =
      Set(StopAnalyzer.ENGLISH_STOP_WORDS_SET).map(v => v.toString) ++
        Set("", ".", ",",
            "\u2019", "\u2018", "\u2013", "\u2022",
            "\u2014", "\uf02d", "\u20ac", "\u2026")

    val props : Properties = new Properties
    props.setProperty("annotators", "tokenize, ssplit, pos, lemma")

    // Don't use direct map, the stanford core nlp takes quite some memory and
    // time to initialize and it is not serializable. Take advantage of
    // partition parallelism.
    emails.mapPartitions(
      emailIterator => {
        val pipeline = new StanfordCoreNLP(props)

        emailIterator.map(email => {
          val document : Annotation = new Annotation(email.contents)
          pipeline.annotate(document)
          val sentences : java.util.List[CoreMap] = document.get(classOf[SentencesAnnotation])

          var words = new ListBuffer[String]
          for (sentence : CoreMap <- sentences) {
            for (token : CoreLabel <- sentence.get(classOf[TokensAnnotation])) {
              val word = token.lemma.toLowerCase
              if (!stopwords.contains(word) && nlp_tags.contains(token.get(classOf[PartOfSpeechAnnotation]))) {
                words += word
              }
            }
          }
          TokenizedDocument(email.id, words.toArray)
        })
      })
  }

  def filterDictionary(dic: RDD[DictionaryItem], above : Double = 0.5, below : Int = 5, keep_n : Int = 1000000): RDD[DictionaryItem] = {
    val max = (above * dic.count()).toLong
    var filtered = dic.filter(item => item.occurrences <= max && item.occurrences >= below)
    if (filtered.count() > keep_n) {
      filtered = filtered
        .sortBy(item => item.occurrences, false) // high first
        .zipWithIndex() // index
        .filter(v => v._2 < keep_n) // keep only indexes smaller than keep_n
        .map(v => v._1) // remove index
    }
    filtered.zipWithIndex().map( v => { // reindex
      DictionaryItem(v._2.toInt, v._1.word, v._1.occurrences)
    })
  }

  def generateDictionary(documents: RDD[TokenizedDocument]): RDD[DictionaryItem] = {
    val uniqueWords = documents
      .flatMap(_.tokens.distinct) // one word per document
      .map((_, 1L)) // one count per word per document
      .reduceByKey((c1, c2) => c1 + c2) // add up
      .cache() // for zip with index

    uniqueWords
      .zipWithIndex() // give each word a unique index
      .map(v => DictionaryItem(v._2.toInt, v._1._1, v._1._2)) // map it to the correct format
      .cache() // for later use
  }
}
