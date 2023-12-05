import sys
import os

sys.path.append(os.path.realpath(__file__)[0:-18]+"..")
#print(os.path.realpath(__file__)[0:-18]+"..")
from machine.bithumb_machine import BithumbMachine
from db.mongodb.mongodb_handler import MongoDBHandler
from AI.lstm_machine import LstmMachine
from machine.chart_machine import ChartMachine
from machine.chatGPT_machine import ChatMachine
import datetime

def extract_close_prices(data):
    close_prices = [float(entry['close_price']) for entry in data]
    return close_prices

bithumbMachine = BithumbMachine()
lstmMachine = LstmMachine()
mongodbMachine = MongoDBHandler(db_name="AI", collection_name="actual_data")

#하루치 데이터 저장
data = bithumbMachine.get_last_data()
recv = mongodbMachine.insert_item(data=data, db_name="AI", collection_name="actual_data")

#model 예측 과정
past_data = mongodbMachine.find_items_for_db(db_name="AI", collection_name="actual_data")
data = extract_close_prices(past_data)
data.reverse()
data = lstmMachine.data_processing(data)
result = lstmMachine.get_predict_value(data)

#예측된 값 저장
one_day_data = {}
one_day_data["timestamp"] = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
one_day_data["predicted_price"] = result + 0.0
mongodbMachine.insert_item(data=one_day_data, db_name="AI", collection_name="predicted_data")

chart_machine = ChartMachine()
chat_machine = ChatMachine()

actual_data_str, predicted_data_str = chart_machine.get_analysis_chart()
res = chat_machine.get_analysis_result(actual_data_str, predicted_data_str)

analysis_data = {"gpt_response":res, "timestamp":datetime.date.today().strftime("%Y-%m-%d")}
print(analysis_data)

mongodbMachine.insert_item(data = analysis_data, db_name="AI", collection_name="analysis_data")