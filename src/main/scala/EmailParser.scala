import java.time.format.DateTimeFormatter
import java.util.regex.Pattern
import java.util.{Locale, Properties}
import javax.mail.{Address, Session}

import edu.stanford.nlp.ling.CoreAnnotations.{PartOfSpeechAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.util.CoreMap
import org.apache.commons.mail.util.{MimeMessageParser, MimeMessageUtils}
import org.apache.lucene.analysis.core.StopAnalyzer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import scala.util.control.Exception.allCatch
import scala.util.matching.Regex

case class DictionaryItem(id: Int, word: String, occurrences: Long)
case class ParsedEmail(id: Long, path: String, message: MimeMessageParser)
case class EmailContents(id: Long, contents: String)
case class TokenizedDocument(id: Long, tokens: Array[String])
case class RawDocument(id: Long, path: String, document: String)
case class BagOfWords(id: Long, words: Array[Int], counts: Array[Int])
case class EmailMetadata(var id: Long, var path: String, var message_id: String,
                         var date: String, var from: String, var to: Array[String], var cc: Array[String],
                         var bcc: Array[String], var subject: String, var references: Array[String])

object EmailParser {
  private case class Params(input: String = null,
                            corpus: String = null,
                            dictionary: String = null,
                            metadata: String = null,
                            above: Double = 0.5,
                            below: Int = 5,
                            keep_n: Int = 1000000)

  val emailDateFormatter = DateTimeFormatter.ofPattern("EEE, d MMM yyyy HH:mm:ss X (z)", Locale.US)

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
        .text("output metadata Avro file with EmailMetadata")
        .action((x, c) => c.copy(metadata = x))
      opt[String]("dictionary")
        .text("output dictionary file, tab separated. Each line is " +
          "word_id<TAB>word<TAB>occurrences")
        .action((x, c) => c.copy(dictionary = x))
      opt[String]("corpus")
        .text("plain text output file of one parsed document per line" +
          "Each text file line holds 1 document. The first line is a comment, the second line is " +
          "[number of documents]<space>[number of words] and the other lines are ordered as " +
          "doc_id;word_id_1,...,word_id_n;word_count_1,...,word_count_n")
        .action((x, c) => c.copy(corpus = x))
      arg[String]("<input>")
        .text("input sequence file containing tuples of (path, plain text email content).")
        .required()
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case None => sys.exit(1)
    }
  }
  def run(params: Params) = {
    // Look at configuration to determine parallelism and run node
    val conf = new SparkConf()
      .setAppName("EmailParser")
    val sc = new SparkContext(conf)

    // Sequence file of path -> contents
    val data : RDD[(String, String)] = sc.sequenceFile(params.input)
    // Testing with single directory
    //val data : RDD[(String, String)] = sc.wholeTextFiles(params.input)

    val documents = data
      .zipWithIndex()
      .map(v => RawDocument(v._2, v._1._1, v._1._2))

    val (metadataOption, tokensOption) = parseEmail(documents,
                                                    params.metadata != null,
                                                    params.dictionary != null || params.corpus != null)

    metadataOption.foreach(metadata => {
      val sqlContext = new SQLContext(sc)
      import sqlContext.implicits._
      metadata.toDF().write
        .format("com.databricks.spark.avro")
        .save(params.metadata)
    })

    tokensOption.foreach(tokens => {
      var dictionary = generateDictionary(tokens)
      dictionary = filterDictionary(dictionary, params.above, params.below, params.keep_n)
        .cache()

      if (params.dictionary != null) {
        dictionary
          .sortBy(item => item.id)
          .map(item => s"${item.id}\t${item.word}\t${item.occurrences}")
          .saveAsTextFile(params.dictionary)
      }

      if (params.corpus != null) {
        val bow = bagOfWords(tokens, dictionary).cache()
        val n_words = dictionary.count()
        val n_docs = bow.count()
        bow
          .sortBy(d => d.id)
          .map(d => s"${d.id};${d.words.mkString(",")};${d.counts.mkString(",")}") // stringify
          .mapPartitionsWithIndex((idx, iterator) => {
          // prepend header
          if (idx == 0) {
            Array[String]("# comment", s"$n_words $n_docs").iterator ++ iterator
          } else {
            iterator
          }
        }).saveAsTextFile(params.corpus) // write
      }
    })
  }

  def bagOfWords(tokens : RDD[TokenizedDocument], dictionary : RDD[DictionaryItem]): RDD[BagOfWords] = {
    val wordLookup = dictionary.map(item => (item.word, item.id))
    val documentWords = tokens.flatMap(doc => doc.tokens.map((_, doc.id)))

    documentWords.join(wordLookup)
      .map(occurrence => ((occurrence._2._1, occurrence._2._2), 1)) // ((document, word_id), 1)
      .reduceByKey(_ + _) // ((document, word_id), count)
      .map(occurrence => (occurrence._1._1, (occurrence._1._2, occurrence._2))) // (document, (word_id, count))
      .groupByKey // (document, Iterator[(word_id, count)])
      .map(bow => {
      val b = bow._2.toArray.sortBy(_._1)
      val words = ArrayBuffer[Int]()
      val counts = ArrayBuffer[Int]()
      for ((word_id, count) <- b) {
        words.append(word_id)
        counts.append(count)
      }
      BagOfWords(bow._1, words.toArray, counts.toArray)
    })
  }

  def parseEmail(data: RDD[RawDocument], parseMetadata : Boolean, parseTokens : Boolean) : (Option[RDD[EmailMetadata]], Option[RDD[TokenizedDocument]]) = {
    val emails = data.map( raw => {
      val s : Session = Session.getDefaultInstance(new Properties())
      val m = MimeMessageUtils.createMimeMessage(s, raw.document)
      val p = new MimeMessageParser(m)
      p.parse()
      ParsedEmail(raw.id, raw.path, p)
    }).cache()

    val metadata = Option(parseMetadata).collect {
      case true => emails.map(email => {
        val from = allCatch.opt(email.message.getFrom) getOrElse ""
        val to = allCatch.opt(getAddresses(email.message.getTo)) getOrElse Array[String]()
        val cc = allCatch.opt(getAddresses(email.message.getCc)) getOrElse Array[String]()
        val bcc = allCatch.opt(getAddresses(email.message.getBcc)) getOrElse Array[String]()
        val subject = allCatch.opt(email.message.getSubject) getOrElse ""

        var date = ""
        var references = Array[String]()
        var message_id = ""
        allCatch.opt(email.message.getMimeMessage).foreach(mimeMessage => {
          date = Option(mimeMessage.getHeader("Date", null))
            .flatMap(d => allCatch.opt(emailDateFormatter.parse(d)))
            .map(DateTimeFormatter.ISO_OFFSET_DATE_TIME.format) getOrElse ""
          references = Option(mimeMessage.getHeader("References")) getOrElse Array[String]()
          message_id = Option(mimeMessage.getMessageID) getOrElse ""
        })
        EmailMetadata(email.id, email.path, message_id, date, from, to, cc, bcc, subject, references)
      })
    }

    val tokens = Option(parseTokens).collect {
      case true => {
        val emailContents = emails.map(email => {
          val msg = email.message
          EmailContents(email.id,
            if (msg.hasPlainContent) msg.getPlainContent
            else if (msg.hasHtmlContent) msg.getHtmlContent
            else "")
        })
        val filtered = filter(fixMalformedMime(emailContents))
        tokenizeStanford(filtered).persist(StorageLevel.MEMORY_AND_DISK)
      }
    }
    emails.unpersist()
    (metadata, tokens)
  }

  def fixMalformedMime(emails: RDD[EmailContents]) : RDD[EmailContents] = {
    // Regular expressions for custom MIME parsing (due to malformed headers)
    val boundaryRegex = new Regex("Content-Type: (?i)(?:multipart)/.*;\\s*boundary=\"(.*)\"")
    val textPartRegex = new Regex("Content-Type: ((?i)(?:text)/)")
    // MIME headers end with double line ending. According to the spec, this should be
    // \r\r but since we already have a malformed header, we accept any line endings.
    val endOfHeader = new Regex("\\n{2}|\\r{2}|(\\n\\r){2}|(\\r\\n){2}")

    emails.map(email => {
      var parts = Array[String](email.contents)

      // remove all boundaries
      boundaryRegex.findAllMatchIn(email.contents).foreach(
        m => { parts = parts.flatMap(_.split(Pattern.quote("--" + m.group(1)) + "(--)?\\s*")) }
      )

      // if the headers are malformed, make sure we don't include any attachments
      // by doing the MIME multipart splitting manually.
      EmailContents(email.id,
        if (parts.length == 1) {
          parts(0)
        } else {
          parts.map(part =>
            // only match text MIME parts
            textPartRegex.findFirstIn(part) match {
              case Some(_) => endOfHeader.findFirstMatchIn(part) match {
                // everything after the header is content
                case Some(eh) => part.substring(eh.end)
                case None => "" // there is no end of header
              }
              case None => "" // content type is not text/*
            }
          ).mkString(" ") // concatenate all text
        })
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
    val symbol_empty_pattern = new Regex("\\s[-\\u0001-\\u001f\"`=%><#_~+*/\\\\\\[\\]|{}$]+|[-\\u0001-\\u001f\"`=%><#_~+*/\\\\\\[\\]|{}]+\\s|[\\u0001-\\u001f\"`=%><#_~+*/\\\\\\[\\]|{}$]+")
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
    val stopWords : Set[String] =
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

          var words = new ArrayBuffer[String]
          for (sentence : CoreMap <- sentences) {
            for (token : CoreLabel <- sentence.get(classOf[TokensAnnotation])) {
              val word = token.lemma.toLowerCase
              if (!stopWords.contains(word) && nlp_tags.contains(token.get(classOf[PartOfSpeechAnnotation]))) {
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
        .sortBy(item => item.occurrences, ascending = false) // high first
        .zipWithIndex() // index
        .filter(_._2 < keep_n) // keep only indexes smaller than keep_n
        .map(_._1) // remove index
    }
    filtered.zipWithIndex().map( v => { // reindex
      DictionaryItem(v._2.toInt, v._1.word, v._1.occurrences)
    })
  }

  def generateDictionary(documents: RDD[TokenizedDocument]): RDD[DictionaryItem] = {
    val uniqueWords = documents
      .flatMap(_.tokens.distinct) // one word per document
      .map((_, 1L)) // one count per word per document
      .reduceByKey(_ + _) // add up
      .cache() // for zip with index

    uniqueWords
      .zipWithIndex() // give each word a unique index
      .map(v => DictionaryItem(v._2.toInt, v._1._1, v._1._2)) // map it to the correct format
  }

  def getAddresses(list : java.util.List[Address]) : Array[String] = {
    list.toArray(Array[Address]().padTo(list.length, null))
      .map(_.toString)
  }
}
