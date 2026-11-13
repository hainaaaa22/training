import rclpy
import math
import numpy as np
import time
from geometry_msgs.msg import Twist

from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from rm_msgs.msg import WheelLegCmd

class WheelLegController(Node):

    def __init__(self):
        super().__init__('wheel_leg_controller')

        # 1. 初始化机器人状态
        self.orientation = np.zeros(4)
        self.angular_velocity = np.zeros(3)

        self.joint_position = np.zeros(6)
        self.joint_velocity = np.zeros(6)
        self.imu_received = False
        self.joint_state_received = False
        self.state_ready_logged = False

        self.wheel_radius = 0.06
        self.dt = 0.002

        self.kp = 0.05
        self.ki = 0.1

        self.left_integral = 0.0
        self.right_integral = 0.0

        self.target_linear_v = 0.0
        self.last_cmd_time = None

        self.debug_counter = 0

        # 2. 创建Publisher
        self.cmd_pub = self.create_publisher(WheelLegCmd, 'actuator_cmd', 10)

        # 3. 创建IMU Subscriber
        self.imu_sub = self.create_subscription(Imu, 'imu', self.imu_callback, 10)

        # 4. 创建JointState Subscriber
        self.joit_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)

        # 5. 创建 Timer
        self.sim_time = 0.0
        self.timer = self.create_timer(self.dt, self.control_loop)

        # 6. 创建cmd_vel Subscriber
        self.cmd_vel_sub = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)

    # 7. IMU 回调函数
    def imu_callback(self, msg):
        self.orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.imu_received = True

    # 8. JointState 回调函数
    def joint_callback(self, msg):
        if len(msg.position) != 6 or len(msg.velocity) != 6:
            self.get_logger().warning('Unexpected joint state length')
            return

        self.joint_position = np.array(msg.position)
        self.joint_velocity = np.array(msg.velocity)
        self.joint_state_received = True

    # 9. 控制循环
    def control_loop(self):
        # 9.1 等待传感器数据准备好
        if not self.imu_received or not self.joint_state_received:
            return

        # 9.2 获取目标速度
        # 超过1秒没有收到目标指令，目标自动回到0
        if (self.last_cmd_time is None or time.monotonic() - self.last_cmd_time > 1.0):
            target_v = 0.0
        else:
            target_v = self.target_linear_v

        # 9.3 线速度转换为目标角速度
        target_omega = target_v / self.wheel_radius

        # 9.4 读取左右轮实际角速度
        right_actual = float(self.joint_velocity[4])
        left_actual = float(self.joint_velocity[5])

        # 9.5 计算轮速误差
        right_error = -target_omega - right_actual
        left_error = target_omega - left_actual

        # 9.6 更新积分项
        self.right_integral += right_error * self.dt
        self.left_integral += left_error * self.dt

        # 简单积分限幅，避免积分无限增大
        self.right_intergral = max(-10.0, min(10.0, self.right_integral))
        self.left_integral = max(-10.0, min(10.0, self.left_integral))

        # 9.7 PI计算
        right_torque = (self.kp * right_error + self.ki * self.right_integral)
        left_torque = (self.kp * left_error + self.ki * self.left_integral)

        # 9.8 电机力矩限幅
        right_torque = max(-2.0, min(2.0, right_torque))
        left_torque = max(-2.0, min(2.0, left_torque))

        # 9.9 创建并发布执行器指令
        cmd = WheelLegCmd()

        cmd.right_phi1 = 0.0
        cmd.right_phi4 = 0.0
        cmd.left_phi1 = 0.0
        cmd.left_phi4 = 0.0

        cmd.right_wheel = right_torque
        cmd.left_wheel = left_torque

        self.cmd_pub.publish(cmd)

        # 9.10 定期打印状态，便于调试
        self.debug_counter += 1

        if self.debug_counter % 250 == 0:
            if abs(target_omega) > 1e-6:

                #左右轮目标角速度
                right_target = -target_omega
                left_target = target_omega

                #计算相对误差
                right_error_rate = (abs(right_target - right_actual) / abs(right_target) * 100)
                left_error_rate = (abs(left_target - left_actual) / abs(left_target) * 100)

                self.get_logger().info(
                    f"Target v = {target_v:.2f} m/s | "
                    f"Right = {right_actual:.2f} rad/s "
                    f"(error {right_error_rate:.1f}%) | "
                    f"Left = {left_actual:.2f} rad/s "
                    f"(error {left_error_rate:.1f}%)"
                )

    # 10. cmd_vel 回调函数
    def cmd_vel_callback(self, msg):
        self.target_linear_v = msg.linear.x
        self.last_cmd_time = time.monotonic()

def main(args = None):
        rclpy.init(args = args)

        node = WheelLegController()

        try:
            rclpy.spin(node)

        except KeyboardInterrupt:
            pass

        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()