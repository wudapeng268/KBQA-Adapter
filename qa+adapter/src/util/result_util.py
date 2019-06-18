from pandas import DataFrame
import pandas as pd
import os

from src.util import FileUtil
def get_server_name():
    import socket
    return socket.gethostname()



def write_result(data,columns,path):
    for k in data:
        if type(data[k]) == float:
            data[k] = round(data[k] * 100, 2)
    df = DataFrame(data=data,index=[0],columns=columns)
    df['server'] = get_server_name()
    df.to_csv(path)
    print("write result in {}".format(path))

def read_result(path):
    if not os.path.exists(path):
        err = "result {} not found".format(path)
        # print()
        return None,err
    return pd.read_csv(path),True

def merge_result():
    names = FileUtil.readFile("current_train_model_name.txt")
    # names = FileUtil.readFile("tl_model.txt")
    errors = []
    candidate_df = []
    for n in names:
        result_path = "{}/result.csv".format(n)
        temp ,state = read_result(result_path)
        if state==True:
            candidate_df.append(temp)
        else:
            errors.append(state)
    concat_result = pd.concat(candidate_df,0)
    output_path = "result.csv"
    # output_path = os.path.join("result.csv")
    concat_result.to_csv(output_path)
    return output_path,errors



#


import os
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart


mail_host="114.212.189.224:25"
mail_user="wup"
mail_pass="nlp_wup"
mail_postfix="nlp.nju.edu.cn"

def send_mail(to_user,sub,content,send_file=None):
    print("Send email to {}".format(to_user))
    me="Result"+"<"+mail_user+"@"+mail_postfix+">"
    # msg = MIMEText(content,_subtype='html',_charset='gb2312')
    msg = MIMEMultipart()
    part = MIMEText(content)
    msg.attach(part)

    msg['Subject'] = sub
    msg['From'] = me
    msg['To'] = to_user + "@" + mail_postfix
    if send_file!=None:
        part = MIMEApplication(open(send_file, 'rb').read())
        part.add_header('Content-Disposition', 'attachment', filename=send_file)
        msg.attach(part)
    try:
        s = smtplib.SMTP()
        s.connect(mail_host)
        s.login(mail_user,mail_pass)
        s.sendmail(me, [msg['To']], msg.as_string())
        s.close()
        return True
    except Exception as err:
        print(str(err))
        return False
