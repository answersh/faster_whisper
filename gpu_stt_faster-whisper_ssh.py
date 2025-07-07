import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import mysql.connector
import pandas as pd
from datetime import datetime
import warnings
import json
import logging
import time
from datetime import datetime, timedelta
from faster_whisper import WhisperModel
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning, module='whisper')

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

class Logger:
    def __init__(self):
        self.logger = None

    def setup(self):
        log_dir = os.getenv('GPU_STT_INTENT_LOG_PATH')
        if log_dir is None:
            raise ValueError('GPU_STT_INTENT_LOG_PATH 환경변수가 설정되지 않음')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        gpu = os.getenv('STT_GPU')
        log_filename = f"gpu{gpu}_stt_log_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        self.logger = logging.getLogger('ProcessLogger')
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        return self.logger

    def info(self, message):
        if self.logger:
            self.logger.info(message)
            print(message)

    def error(self, message):
        if self.logger:
            self.logger.error(message)
            print(message)

class DataProcessor:
    def __init__(self, db_config, logger):
        self.db_config = db_config
        self.logger = logger
        self.conn = None
        self.cursor = None

    def connect_db(self):
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            self.logger.error(f"DB연결 실패: {e}")
            return False

    def close_db(self):
        if self.cursor:
            self.cursor.close()
        if self.conn and self.conn.is_connected():
            self.conn.close()

    def fetch_data(self, query):
        try:
            self.cursor.execute(query)
            columns = [column[0] for column in self.cursor.description]
            data = self.cursor.fetchall()
            return pd.DataFrame(data, columns=columns)
        except Exception as e:
            self.logger.error(f"데이터 조회 실패: {e}")
            return pd.DataFrame()

    def execute_query(self, query, params=None):
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"쿼리 실행실패: {e}")
            self.conn.rollback()
            return False

    def update_task_status(self, sno, status):
        try:
            query = "UPDATE contact_consult SET task_status = %s WHERE sno = %s"
            self.cursor.execute(query, (status, sno))
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"DB 업데이트 실패 (sno: {sno}): {e}")
            self.conn.rollback()
            return False

    def update_result(self, result):
        try:
            query = '''
            UPDATE contact_consult
            SET stt = %s,
                task_status = 'COMPLETED'
            WHERE sno = %s
            '''
            params = (
                result['stt'],
                result['sno']
            )
            self.cursor.execute(query, params)
            self.conn.commit()
            self.logger.info(f"sno: {result['sno']} 데이터 업데이트 완료")
            return True
        except Exception as e:
            self.logger.error(f"데이터 업데이트 실패: {e}")
            self.conn.rollback()
            return False

class STTProcessor:
    def __init__(self, model_path, logger):
        self.logger = logger
        self.model = None
        self.model_path = model_path

    def load_model(self):
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            self.model = WhisperModel(self.model_path, device='cuda')
            return True
        except Exception as e:
            self.logger.error(f"faster-whisper모델 로드 실패: {e}")
            return False

    def transcribe(self, audio_path):
        try:
            base_path = os.getenv("RECORD_FILE_PATH")
            full_audio_path = os.path.join(base_path, audio_path)
            initial_prompt = '이 대화는 경기지역 고객센터 상담사, 고객센터, 인천이음 고객센터, 상담사, 인천이음, 인천이음카드, 인천이음 고객센터 등의 단어가 사용됩니다.'
            segments, _ = self.model.transcribe(full_audio_path, language='ko', task='transcribe', initial_prompt=initial_prompt)
            
            transcript_text = ""
            for segment in segments:
                transcript_text += segment.text + " "
            return transcript_text.strip()
        except Exception as e:
            self.logger.error(f"STT 실패 (파일: {full_audio_path}): {e}")
            return ""

def mask_content(stt, name, oper_name):
    stt = str(stt) if pd.notna(stt) else ''
    name = str(name) if pd.notna(name) else ''
    oper_name = str(oper_name) if pd.notna(oper_name) else ''

    if name and name in stt:
        masked_name = name[0] + "*" * (len(name)-1)
        stt = stt.replace(name, masked_name)

    if oper_name and oper_name in stt:
        masked_oper_name = oper_name[0] + "*" * (len(oper_name)-1)
        stt = stt.replace(name, masked_oper_name)

    stt = ''.join(['1' if char.isdigit() else char for char in stt])
    return stt

def process_record(row, stt_processor, data_processor, logger):
    start_time = time.time()
    sno = row["sno"]
    result = {
        'sno': sno,
        'recording': row['recording'],
        'stt': None,
        'task_status':'FAILED'
    }

    try:
        data_processor.update_task_status(sno, 'IN_PROGRESS')
        transcript_text = stt_processor.transcribe(row['recording'])
        if not transcript_text:
            raise Exception("STT 결과가 없습니다.")

        masked_contents = mask_content(
            transcript_text,
            row['name'],
            row['oper_name']
        )

        stt_end_time = time.time()
        stt_processing_time = round(stt_end_time - start_time, 2)

        result['stt'] = masked_contents
        result['task_status'] = 'SUCCESS'

        total_processing_time = round(time.time() - start_time, 2)
        logger.info(f"sno: {sno}|STT_LENGTH: {len(masked_contents)}|STT_PROCESSING_TIME: {stt_processing_time}|TOTAL_PROCESSING_TIME: {total_processing_time}")
        return result

    except Exception as e:
        logger.error(f"sno: {sno} - 처리오류: {e}")
        return result

def main():
    load_dotenv()
    logger_obj = Logger()
    logger = logger_obj.setup()
    logger.info("##### Start #####")

    try:
        db_config = {
            'host': os.getenv('MYSQL_HOST'),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'database': os.getenv('MYSQL_DATABASE'),
            'port': int(os.getenv('MYSQL_PORT'))
        }
        whisper_model_path = os.getenv('FASTER_WHISPER_MODEL_PATH')

        data_processor = DataProcessor(db_config, logger)
        if not data_processor.connect_db():
            raise Exception("DB연결 실패")

        stt_processor = STTProcessor(whisper_model_path, logger)
        if not stt_processor.load_model():
            raise Exception('STT모델 로드실패')

        gpu = os.getenv("STT_GPU")

        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        query = f'''
        	SELECT
        		sno,
        		contact_id,
        		name,
                oper_name,
        		asp_id,
        		recording,
        		task_status,
        		gpu_no
        	FROM ksb.contact_consult
            WHERE gpu_no = {gpu} AND task_status = 'READY'  AND method IN ('INBOUND', 'CALLBACK') AND recording LIKE "%.wav" AND date_format(connected_at, "%Y%m%d") = {yesterday}
            '''
        ready_data = data_processor.fetch_data(query)
        ready_data['recording'] = ready_data['recording'].str.replace('\\','/',regex=False)

        if ready_data.empty:
            logger.info("TASK_STATUS가 READY상태인 데이터가 없습니다.")
            return

        logger.info(f"{len(ready_data)}행의 데이터를 진행합니다.")

        for _, row in ready_data.iterrows():
            sno = row['sno']
            try:
                update_query = '''
                    UPDATE contact_consult
                    SET task_status = 'IN_PROGRESS'
                    WHERE sno = %s
                '''
                if not data_processor.execute_query(update_query, (sno,)):
                    logger.error(f"sno: {sno} 상태 업데이트 실패, 다음으로 진행")
                    continue
                logger.info(f"sno: {sno} 데이터 처리 시작")

                result = process_record(
                    row,
                    stt_processor,
                    data_processor,
                    logger
                )

                if result['task_status'] == 'SUCCESS':
                    if data_processor.update_result(result):
                        logger.info(f"sno: {sno} 처리완료")
                    else:
                        logger.error(f"sno: {sno} 결과 업데이트 실패")
                else:
                    logger.error(f"sno: {sno} 처리 실패")
                    data_processor.update_task_status(sno, 'IN_PROGRESS')

            except Exception as e:
                logger.error(f"sno: {sno} 처리 중 오류 발생: {str(e)}")
                data_processor.update_task_status(sno, 'IN_PROGRESS')

    except Exception as e:
        logger.error(f"처리 중 오류발생: {e}")

    finally:
        if 'data_processor' in locals():
            data_processor.close_db()
        logger.info("STT 처리 종료")

if __name__ == "__main__":
    main()
