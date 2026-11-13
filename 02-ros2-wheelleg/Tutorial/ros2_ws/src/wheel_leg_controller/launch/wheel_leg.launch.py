
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # 1. 启动 MuJoCo 仿真节点
    sim_node = Node(
        package='wheel_leg_sim',
        executable='wheel_leg_sim',
        name='wheel_leg_sim',
        output='screen'
    )

    # 2. 启动轮腿机器人控制器节点
    controller_node = Node(
        package='wheel_leg_controller',
        executable='wheel_leg_controller',
        name='wheel_leg_controller',
        output='screen'
    )

    # 3. 将两个节点加入启动列表
    return LaunchDescription([
        sim_node,
        controller_node
    ])
