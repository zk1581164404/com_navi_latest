# 导入 Flask 类
from flask import Flask,request, jsonify, render_template
import com_navi_ga_maddpg_no_noise
import demo
import numpy as np
# 创建了这个类的实例。第一个参数是应用模块或者包的名称。
app = Flask(__name__)

# 使用 route() 装饰器来告诉 Flask 触发函数的 URL
@app.route("/")
def home():
    return render_template('page.html')  #这里只能是相对路径
# def hello():
#     print(demo.getlist())
#     return demo.getlist()

@app.route('/predict', methods=['POST'])
def predict():
    features_list = []
    for x in request.form.values():
        features_list.append(x)
    # features_list = [float(x) for x in request.form.values()]
    with open('/root/com_navi_latest/maddpg_communicate_navigation_uav_latest/multiagent-particle-envs-master/multiagent/scenarios/input_pos.txt','w') as f:    #设置文件对象
        for x in features_list:
            f.write(x)                 #将字符串写入文件中
        f.write('\n')
    print(demo.getlist())
    return render_template('page.html',prediction_display_area='无人机部署位置为：{}'.format(demo.getlist()))
    # np.savetxt('/root/com_navi_latest/maddpg_communicate_navigation_uav_latest/multiagent-particle-envs-master/multiagent/scenarios/input_pos.txt',features_list)

if __name__ == "__main__":
    # 使用 run() 函数来运行本地服务器和我们的应用
    app.run()
