

## Installation

First, install spark on the system. Then

```shell
gradle assemble
```

now, the system runs with
spark

To run it at the root of spark-lda:
```
$SPARK_HOME/bin/spark-submit --class ScalaLDA --master yarn-cluster --num-executors 15 `pwd`/build/libs/ScalaLDA-0.1-SNAPSHOT.jar /user/sherlock/lda/training/data/corpus.joris /user/sherlock/lda/myLDAModel.csv

$SPARK_HOME/bin/spark-submit --class ScalaLDASeq --master yarn-cluster --num-executors 15 `pwd`/build/libs/ScalaLDA-0.1-SNAPSHOT.jar /user/sherlock/enron_mail/enron_mail_clean.seq /user/sherlock/lda/seqLDAModel.csv
```

