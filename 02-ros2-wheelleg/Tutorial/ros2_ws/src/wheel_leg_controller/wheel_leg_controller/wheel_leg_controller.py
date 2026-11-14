import rclpy
import math
import numpy as np
import time
from geometry_msgs.msg import Twist

from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from rm_msgs.msg import WheelLegCmd
from .vmc_cal import five_links_param

class LegLengthPD:

    def __init__(self):
        self.kp = 700.0
        self.kd = 5.0
        self.prev_error = 0.0

    def compute(self, current, target):
        error = target - current

        # 保持原工程的差分计算方式
        derivative = error - self.prev_error
        self.prev_error = error
        force = self.kp * error + self.kd * derivative

        return float(np.clip(force, -150.0, 150.0))

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

        self.left_leg = five_links_param(self.dt)
        self.right_leg = five_links_param(self.dt)

        self.kp = 0.05
        self.ki = 0.1

        self.left_integral = 0.0
        self.right_integral = 0.0

        self.target_linear_v = 0.0
        self.last_cmd_time = None

        self.debug_counter = 0

        self.target_leg_length = 0.10

        self.left_length_pd = LegLengthPD()
        self.right_length_pd = LegLengthPD()

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

        if not self.imu_received or not self.joint_state_received:
            return

        pitch, gyro_pitch = self.get_pitch_state()

        left_length, right_length = self.update_leg_kinematics(
            pitch, gyro_pitch
        )

        cmd = WheelLegCmd()

        # 判断是否适合执行腿部控制
        safe = (
            abs(pitch) < math.radians(40.0)
            and abs(math.cos(self.left_leg.theta)) > 0.15
            and abs(math.cos(self.right_leg.theta)) > 0.15
        )

        if safe:
            # 计算两侧的虚拟支撑力
            left_force = self.calculate_support_force(
                self.left_leg,
                self.left_length_pd
            )

            right_force = self.calculate_support_force(
                self.right_leg,
                self.right_length_pd
            )

            # VMC: 将虚拟力映射为关节电机力矩
            left_t1, left_t2 = self.left_leg.VMC_torque_cal(
                left_force, 0.0
            )

            right_t1, right_t2 = self.right_leg.VMC_torque_cal(
                right_force, 0.0
            )

            # 按原工程的左右腿符号约定发送力矩
            cmd.right_phi1 = float(right_t1)
            cmd.right_phi4 = float(right_t2)

            cmd.left_phi1 = float(-left_t1)
            cmd.left_phi4 = float(-left_t2)

        else:
            self.left_length_pd.prev_error = 0.0
            self.right_length_pd.prev_error = 0.0

            cmd.right_phi1 = 0.0
            cmd.right_phi4 = 0.0
            cmd.left_phi1 = 0.0
            cmd.left_phi4 = 0.0

            #暂时不进行轮速PI控制
            cmd.right_wheel = 0.0
            cmd.left_wheel = 0.0

        self.cmd_pub.publish(cmd)

        self.debug_counter += 1

        # 测试期间每50次循环输出一次
        if self.debug_counter % 50 == 0:
            self.get_logger().info(
                f'Pitch={math.degrees(pitch):.2f} deg | '
                f'Left L0={left_length:.4f} m | '
                f'Right L0={right_length:.4f} m | '
                f'VMC active={safe}'
            )
                

    # 10. cmd_vel 回调函数
    def cmd_vel_callback(self, msg):
        self.target_linear_v = msg.linear.x
        self.last_cmd_time = time.monotonic()

    # 11. 更新腿部运动学
    def update_leg_kinematics(self, pitch, gyro_pitch):

        # 左腿运动学
        self.left_leg.forward_kinematics_cal(
            1,
            float(self.joint_position[2]),
            float(self.joint_position[3]),
            pitch,
            gyro_pitch
        )

        # 右腿运动学
        self.right_leg.forward_kinematics_cal(
            0,
            float(self.joint_position[0]),
            float(self.joint_position[1]),
            pitch,
            gyro_pitch
        )

        # 返回左右腿长度
        left_length = self.left_leg.L_0
        right_length = self.right_leg.L_0

        return left_length, right_length

    # 12. 获取俯仰角状态
    def get_pitch_state(self):
        x,y,z,w = self.orientation

        # 按照robot_info_update.py计算Pitch
        t2 = 2.0 * (w * y - z * x)

        # 限制到asin的定义域
        t2 = max(-1.0, min(1.0, t2))

        pitch_raw = math.asin(t2)

        # 保持原版控制器的正负号约定
        pitch = -pitch_raw

        gyro_pitch = -float(self.angular_velocity[1])

        return pitch, gyro_pitch

    # 13. VMC支撑力计算
    def calculate_support_force(self, leg, length_pd):

        f_pid = length_pd.compute(
            leg.L_0,
            self.target_leg_length
        )

        #原版重力补偿
        gravity_force = 4.0 / math.cos(leg.theta)

        f0 = gravity_force + f_pid
        return float(np.clip(f0, -100.0, 100.0))

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