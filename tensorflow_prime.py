# Auther        : wangrc
# Date          : 2019-03-29
# Description   :
# Refers        :
# Returns       :
import argparse
from pyspark.sql import SparkSession


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default='mlg')
    args = parser.parse_args()
    return args


def df_to_hive(spark, df, table_name):
    tmp_table_name = "tmp_" + table_name
    df.registerTempTable(tmp_table_name)
    delete_sql = "drop table if exists " + table_name
    create_sql = "create table " + table_name + " as select * from " + tmp_table_name
    spark.sql(delete_sql)
    spark.sql(create_sql)


def main(args):
    ss = SparkSession.builder \
        .appName("********************") \
        .enableHiveSupport() \
        .getOrCreate()
    ss.sql(f'use {args.db}')


if __name__ == '__main__':
    args = parse_args()
    main(args)