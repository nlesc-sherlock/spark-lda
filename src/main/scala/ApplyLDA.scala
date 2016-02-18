import org.apache.spark.mllib.clustering.DistributedLDAModel
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import FileUtil._

object ApplyLDA {
  private case class Params(
     ldaFile: String = null,
     bowFile: String = null,
     outFile: String = null
     )

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("ApplyLDA") {
      head("ApplyLDA: ...")
      arg[String]("<ldaFile>")
        .text("...")
        .required()
        .action((x, c) => c.copy(ldaFile = x))
      arg[String]("<bowFile>")
        .text("...")
        .required()
        .action((x, c) => c.copy(bowFile = x))
      arg[String]("<outFile>")
        .text("...")
        .required()
        .action((x, c) => c.copy(outFile = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf()
      .setAppName("ApplyLDA")
      .set("spark.default.parallelism", "8")
    val sc = new SparkContext(conf)

    val modelName = params.ldaFile  //"data/enron_small_lda.csv.model"
    val dataFile = params.bowFile   //"data/enron_small_bow.csv/part-00000"
    val outFile = params.outFile    //"neoResponse.txt"
    val myModel = DistributedLDAModel.load(sc, modelName)
    val localModel = myModel.toLocal

    val data = sc.textFile(dataFile)
    val corpus = parseCustomCsv(data)

    val res = localModel.topicDistributions(corpus)
    res.saveAsTextFile(outFile)
  }
}
