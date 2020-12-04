import pandas as pd
import numpy as np
import face_data_collect as f
import face_recognition as fr
from csv import writer

# df= pd.read_csv('student_details.csv')
# print(df.iloc[0]['roll']
l  = ['322','sdfasf']
with open('student_details.csv',mode='a+',newline='') as f:
	obj = writer(f)
	obj.writerow(l)
# fr.mark_attendance()