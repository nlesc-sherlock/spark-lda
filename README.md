

## Installation

First, install Java, Scala 2.10 and Spark on the system. Make sure that environment variables `$JAVA_HOME`, `$SCALA_HOME` and `$SPARK_HOME` are set.

```shell
./gradlew build
```

## Usage

First define a function to join jars together and save your spark configuration
```shell
function join() {
    local IFS=$1
    shift
    echo "$*"
}
myspark=$SPARK_HOME/bin/spark-submit --master yarn-cluster --num-executors 15 --jars $(join ',' libs/*.jar)
myjar=<path>/spark-lda/build/libs/ScalaLDA-0.1-SNAPSHOT.jar
```

Then to preprocess email, run:
```shell
$myspark --class EmailParser $myjar data/email/ data/dic.csv data/bow.csv
```

To run the LDA algorithm on `--k topics`:
```shell
$myspark --class ScalaLDA $myjar --k 10 data/bow.csv data/lda.csv
```
