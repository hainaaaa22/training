import rclpy
import math
import numpy as np

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

        # 2. 创建Publisher
        self.cmd_pub = self.create_publisher(WheelLegCmd, 'actuator_cmd', 10)

        # 3. 创建IMU Subscriber
        self.imu_sub = self.create_subscription(Imu, 'imu', self.imu_callback, 10)

        # 4. 创建JointState Subscriber
        self.joit_sub = self.create_subscription(JointState, 'joint_states', self.joint_callback, 10)

        # 5. 创建 Timer
        self.dt = 0.02
        self.sim_time = 0.0
        self.timer = self.create_timer(self.dt, self.control_loop)

    # 6. IMU 回调函数
    def imu_callback(self, msg):
        self.orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.angular_velocity = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.imu_received = True

    # 7. JointState 回调函数
    def joint_callback(self, msg):
        if len(msg.position) != 6 or len(msg.velocity) != 6:
            self.get_logger().warning('Unexpected joint state length')
            return

        self.joint_position = np.array(msg.position)
        self.joint_velocity = np.array(msg.velocity)
        self.joint_state_received = True

    # 8. 控制循环
    def control_loop(self):
        # 8.1 检查传感器数据是否准备好
        if not self.imu_received or not self.joint_state_received:
            return

        # 8.2 第一次运行时输出传感器状态
        if not self.state_ready_logged:
            self.get_logger().info(f'Angular velocity: {self.angular_velocity}')
            self.get_logger().info(f'Joint positions: {self.joint_position}')
            self.get_logger().info(f'Joint velocities: {self.joint_velocity}')
            self.state_ready_logged = True

        # 8.3 更新时间
        self.sim_time += self.dt

        # 8.4 计算一个小幅度正弦力矩
        wheel_torque = 0.2 * math.sin(2.0 * math.pi * 0.5 * self.sim_time)

        # 8.5 创建执行器指令消息
        cmd = WheelLegCmd()

        # 8.6 腿部四个执行器暂时设置为零力矩
        cmd.right_phi1 = 0.0
        cmd.right_phi4 = 0.0
        cmd.left_phi1 = 0.0
        cmd.left_phi4 = 0.0

        # 8.7 左右轮执行器施加相反方向力矩
        cmd.right_wheel = -wheel_torque
        cmd.left_wheel = wheel_torque

        # 8.8 发布消息
        self.cmd_pub.publish(cmd)

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