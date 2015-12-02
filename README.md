

## Installation

First, install spark on the system. Then

```shell
gradle assemble
```

now, the system runs with
spark

To run it:
```
$SPARK_HOME/bin/spark-submit --class ScalaLDA --master yarn-cluster --num-executors 15 <path>/spark-lda/build/libs/ScalaLDA-0.1-SNAPSHOT.jar /user/sherlock/lda/training/data/VraagTextCorpus.mm /user/sherlock/lda/myLDAModel.csv
```

