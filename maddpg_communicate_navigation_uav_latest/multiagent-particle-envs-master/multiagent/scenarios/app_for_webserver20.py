# 导入 Flask 类
from flask import Flask,request, jsonify, render_template
import numpy as np
import json
import os
# 创建了这个类的实例。第一个参数是应用模块或者包的名称。
app = Flask(__name__)

web_input_path = '/home/zk/maddpg/com_navi_latest_for_ubuntu/maddpg_communicate_navigation_uav_latest/multiagent-particle-envs-master/multiagent/scenarios/json_for_web_input.json'
web_output_path = '/home/zk/maddpg/com_navi_latest_for_ubuntu/maddpg_communicate_navigation_uav_latest/multiagent-particle-envs-master/multiagent/scenarios/json_for_web_output.json'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# 使用 route() 装饰器来告诉 Flask 触发函数的 URL
@app.route("/")
def home():
    return render_template('page.html')  #这里只能是相对路径
# def hello():
#     print(demo.getlist())
#     return demo.getlist()

def getDeployment():
    result_line = {}
    if os.path.getsize(web_output_path):
        with open(web_output_path,"r") as f:    #设置文件对象
            lines = f.readlines()
            if len(lines) == 2:
                result_line = lines[1]  # 取第二行 才是结果
                # print("line0",lines[0])
                # print("line1",lines[1])
    return result_line

@app.route('/predict', methods=['POST'])
def predict():

    dict_input = {}

    for x in request.form.values():
        dict_input = x
    #写入输入
    with open(web_input_path,'w') as f:    #设置文件对象
        data = json.dumps(dict_input,cls=NpEncoder)
        f.write(eval(data))
        # os.system("bash /home/zk/maddpg/com_navi_latest_for_ubuntu/maddpg_communicate_navigation_uav_latest/maddpg-master/experiments/runwebserver.sh")
    
    deploy_result = {}
    
    while True:
        temp = getDeployment()
        if temp:
            deploy_result = temp
            break
    #  清空输出
    open(web_output_path,'w').close()
    return deploy_result  #直接返还结果即可
    # return render_template('page.html',prediction_display_area='无人机部署位置为：{}'.format(getDeployment()))

if __name__ == "__main__":
    #  清空输出
    open(web_output_path,'w').close()
    # 使用 run() 函数来运行本地服务器和我们的应用
    app.run()
