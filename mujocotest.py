import mujoco
import mujoco.viewer
import numpy as np
import os
import time

# 1. 模型路径设置
model_path = "/home/alexhuge/Documents/GitHub/Alexbotmini_urdf/alexbotmini/mjcf/alexbotmini.xml"  # 必须使用MuJoCo格式的模型文件
# model_path = '/home/alexhuge/Documents/GitHub/Alexbotmini_urdf/ref/H1/h1_2_12dof.xml' 

assert os.path.exists(model_path), "Model file not found!"

# 2. 加载MuJoCo模型
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 3. 初始化参考轨迹参数（保留原有逻辑）
# initialAngle = np.array([-0.174,0,0,0.314,0.14,0,0.174,0,0,-0.314,-0.14,0])
initialAngle = np.zeros(12)
scale_1 = 0.17
scale_2 = 2 * scale_1

# 4. 创建PD控制器参数
kp = np.array([180, 120, 120, 180, 45, 45, 180, 120, 120, 180, 45, 45], dtype=np.double)
kd = np.array([10, 8, 8, 10, 2.5, 2.5, 10, 8, 8, 10, 2.5, 2.5,], dtype=np.double)
tau_limit = 200. * np.ones(12, dtype=np.double)
# for simple PD control, we can set kp and kd to 100,10
# kp = 50
# kd = 10

# 5. 运行仿真循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 设置初始状态
    data.qpos[3:7] = [1, 0, 0, 0]  # 四元数初始化
    data.qpos[2] = 0.76            # 初始高度
    
    # 主循环
    for i in range(100000):
        # 生成参考轨迹（保持原有逻辑）
        phase = i * 0.005
        sin_pos = np.sin(2 * np.pi * phase)
        
        ref_dof_pos = initialAngle.copy()
        # 左腿控制逻辑
        if sin_pos > 0:
            ref_dof_pos[0] = sin_pos * scale_1 + initialAngle[0]
            ref_dof_pos[3] = sin_pos * scale_2 + initialAngle[3]
            ref_dof_pos[4] = -sin_pos * scale_1 + initialAngle[4]
        # 右腿控制逻辑
        else:
            ref_dof_pos[6] = sin_pos * scale_1 + initialAngle[6]
            ref_dof_pos[9] = sin_pos * scale_2 + initialAngle[9]
            ref_dof_pos[10] = -sin_pos * scale_1 + initialAngle[10]
        
        # 6. 应用PD控制
        for j in range(12):  # 假设前12个是驱动关节
            error = ref_dof_pos[j] - data.qpos[7+j]
            derror = -data.qvel[6+j]  # 速度误差
            
            # 计算控制力矩并限制输出
            torque = kp[j] * error + kd[j] * derror
            data.ctrl[j] = np.clip(torque, -tau_limit[j], tau_limit[j])


        
        # 7. 步进仿真
        mujoco.mj_step(model, data)
        
        # 8. 同步可视化
        viewer.sync()
        
        # 控制仿真速度
        time_until_next_step = model.opt.timestep - (data.time % model.opt.timestep)
        # time.sleep(time_until_next_step)
        time.sleep(0.001)