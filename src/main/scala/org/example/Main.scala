package org.example

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.fpm.AssociationRules

object Main {

  def main(args: Array[String]) {
    val sf = new SparkConf().setAppName("anything").setMaster("local")
    val sc = new SparkContext(sf)

    val data = sc.textFile("C:\\Users\\ToThang\\Desktop\\ML3\\models\\test.csv")
    val transactions: RDD[Array[String]] = data.map(s => s.trim.split(','))

    val fpg = new FPGrowth().setMinSupport(0.1)
    val model = fpg.run(transactions)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    val ar = new AssociationRules().setMinConfidence(0.1)
    val results = ar.run(model.freqItemsets)

    results.collect().foreach { rule =>
      println("[" + rule.antecedent.mkString(",") + " => " + rule.consequent.mkString(",") + "]," + rule.confidence)
    }
  }
}

