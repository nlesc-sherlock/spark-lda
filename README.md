

## Installation

First, install Java, Scala 2.10 and Spark on the system. Make sure that environment variables `$JAVA_HOME`, `$SCALA_HOME` and `$SPARK_HOME` are set.

```shell
./gradlew shadowJar
```

## Usage

To run it at the root of spark-lda, first define a function to join jars together and save your spark configuration
```shell
myjar="$(pwd)/build/libs/ScalaLDA-0.1-SNAPSHOT-all.jar"
```

Then to preprocess email, run:
```shell
spark-submit --class EmailParser $myjar data/email/ --metadata data/metadata.seq --dictionary data/dic.csv --corpus data/bow.csv
```

To run the LDA algorithm on `--k topics`:
```shell
spark-submit --class ScalaLDA $myjar --k 10 data/bow.csv data/lda.csv
```

To find the topic proportions of each document, run:
```shell
spark-submit --class ApplyLDA $myjar data/lda.csv.model data/bow.csv data/document_topics.csv
```

### Note
All the locally-compiled jars need to be copied on the Spark machine. Currently the files are organized:
```shell
/home/shelly/spark-lda/
                      /ScalaLDA-0.1-SNAPSHOT.jar
                      /lib/*.jar
                      /data/enron_data.seq
                      /test.sh
```
The input data is expected in Sequence file format. The easiest way to compress it is using the
[forqlift] (www.exmachinatech.net/projects/forqlift) tool. See [here](https://github.com/nlesc-sherlock/analyzing-corpora#step-1---the-original-data) for more information).


The data needs to be registered with hadoop as well:
```shell
$hdfs dfs -put <localdata> <hdfsdata>
```


