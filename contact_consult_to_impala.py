from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.conf import SparkConf
import os
import shutil
import pandas as pd
import mariadb
import numpy as np
import chardet
import logging
import sys
from datetime import datetime, timedelta
import pymysql
from impala.dbapi import connect

from dotenv import load_dotenv

load_dotenv()


################## log설정 ##################

# 로그 디렉토리 생성
log_directory = os.getenv('LOG_DIRECTORY')
os.makedirs(log_directory, exist_ok=True)
# 날짜별 로그 파일명 설정
log_filename = os.path.join(log_directory, f"conatact_consult_to_impala{datetime.now().strftime('%Y%m%d')}.log")

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 모든 레벨의 로그를 기록

if not logger.handlers:
    # 파일 핸들러 설정 (파일에는 모든 로그 기록)
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 모든 레벨 기록

    # 콘솔 핸들러 설정 (콘솔에는 기본적으로 누적되지 않도록 sys.stdout을 덮어씀)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # 모든 레벨 기록

    # 포맷 설정
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 핸들러를 로거에 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# 전체 프로세스 시작 시간 기록
three_days_ago = (datetime.now() - timedelta(days=4)).strftime('%Y%m%d')
start_time = datetime.now()
now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

############### maria db : contact_consult 데이터 추출 #########################
# maria db(contact_consult) 접속
## 5/13 기준 : 3/27 ~ 4/27 까지 intent 정보가 있음
logging.info("데이터 추출 및 삽입 프로세스 시작(Maria DB → HUE)")

try:
    logging.info("Maria DB 연결 시도 (consult 테이블 조회)")
    db_config = {
        'host': os.getenv("MYSQL_HOST"), 
        'port': int(os.getenv('MYSQL_PORT')),  # 문자열을 정수로 변환
        'user': os.getenv('MYSQL_USER'),
        'password': os.getenv('MYSQL_PASSWORD'),
        'database': os.getenv('MYSQL_DATABASE')
    }
    connection = mariadb.connect(**db_config)
    logging.info("contact_consult 테이블 연결 성공")

    with connection.cursor() as cursor:
        query = f'''
        SELECT
            distinct contact_id, 
            detail_intent_0523 AS detail_intent,
            current_timestamp() AS hdfs_upd_dttm
        FROM contact_consult
        WHERE 1=1
        AND DATE_FORMAT(connected_at, "%Y%m%d") >= {three_days_ago}
        AND detail_intent_0523 IS NOT NULL
        #AND DATE_FORMAT(connected_at, "%Y%m%d") BETWEEN 20250327 AND 20250622        
'''
        logging.info(f"contact_consult 테이블 쿼리 실행: {query}")
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # 컬럼 이름 추출
        columns = [desc[0] for desc in cursor.description]
        #logging.info(f"조회된 rows: {rows}")
        #logging.info(f"컬럼 이름: {columns}")

        if rows:
            logging.info(f"contact_consult 테이블 조회된 데이터 행 수: {len(rows)}")
            df = pd.DataFrame(rows, columns=columns)
            logging.info(f"데이터프레임 생성 완료: {df.shape[0]}행, {df.shape[1]}컬럼")
            #logging.info(f"df 내용: \n{df}")
        else:
            logging.warning("contact_consult 테이블에서 데이터가 조회되지 않았습니다.")
            df = pd.DataFrame()

except mariadb.Error as e:
    logging.error(f"contact_consult 테이블 조회 중 MariaDB 오류 발생: {e}")
    df = pd.DataFrame()
except Exception as e:
    logging.error(f"contact_consult 테이블 조회 중 예기치 않은 오류 발생: {e}")
    df = pd.DataFrame()

finally:
    if 'connection' in locals() and connection.open:
        connection.close()
        logging.info("MariaDB 데이터베이스 연결 종료 (contact_consult 조회)")


################## Impala update ##################
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # SparkConf 설정
    conf = SparkConf()
    conf.set("spark.jars", "/home/konadmin/cust_cntr/kudu-spark3_2.12-1.15.0.7.1.7.1000-141.jar")
    
    # SparkSession 생성
    spark = SparkSession.builder.appName("Upsert_to_CONTACT_CONSULT").config(conf=conf).getOrCreate()
    logging.info("SparkSession 생성 완료")

    # DataFrame 확인
    if df.empty:
        logging.warning("DataFrame이 비어 있습니다. UPSERT 종료.")
        spark.stop()
        exit()
    logging.info(f"DataFrame 데이터 행 수: {df.shape[0]}, 컬럼: {df.columns.tolist()}")

    # 스키마 정의
    schema = StructType([
        StructField("contact_id", StringType(), False),  # 기본 키는 null 불가로 설정
        StructField("detail_intent", StringType(), True),
        StructField("hdfs_upd_dttm", TimestampType(), True)
    ])

    # Pandas DataFrame을 Spark DataFrame으로 변환
    df_spark = spark.createDataFrame(df, schema)
    logging.info("Spark DataFrame 생성 완료")
    df_spark.show(truncate=False)  # 데이터 확인

    # hdfs_upd_dttm을 current_timestamp로 설정
    from pyspark.sql.functions import current_timestamp
    df_spark = df_spark.withColumn("hdfs_upd_dttm", current_timestamp())
    
    # Kudu 테이블에 UPSERT
    df_spark.write.format("org.apache.kudu.spark.kudu") \
        .option("kudu.master", '10.150.150.71:7051, 10.150.150.72:7051, 10.150.150.73:7051') \
        .option("kudu.table", "impala::cti.stt_pre") \
        .option("kudu.operation", "upsert") \
        .mode("append") \
        .save()
    logging.info("Kudu 테이블에 UPSERT 완료")

except Exception as e:
    logging.error(f"Spark UPSERT 중 오류 발생: {e}")

finally:
    if 'spark' in locals():
        spark.stop()
        logging.info("SparkSession 종료")  