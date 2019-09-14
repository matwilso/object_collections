#!/usr/bin/env python2

import numpy as np
import sys
import copy
import rospy
import tf2_ros
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from baxter_paddle_moveit_config.srv import SimpleMove, SimpleMoveRequest, SimpleMoveResponse
from geometry_msgs.msg import Vector3, PoseStamped, Pose, Point, Quaternion
import tf2_geometry_msgs
import tf.transformations



def all_close(goal, actual, tolerance):
  """
  Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
  @param: goal       A list of floats, a Pose or a PoseStamped
  @param: actual     A list of floats, a Pose or a PoseStamped
  @param: tolerance  A float
  @returns: bool
  """
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

  return True


class CartesianPoseMoveitPlanner(object):
  def __init__(self):
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_paddle_node')

    self.tf_buffer = tf2_ros.Buffer()
    self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    self.robot = moveit_commander.RobotCommander()
    self.scene = moveit_commander.PlanningSceneInterface()
    self.group_names = self.robot.get_group_names()
    self.left_group = moveit_commander.MoveGroupCommander('left_arm')
    self.left_group.set_planner_id('RRTConnectkConfigDefault')
    self.left_group.set_planning_time(10)
    self.left_group.set_num_planning_attempts(3)

    self.right_group = moveit_commander.MoveGroupCommander('right_arm')
    self.right_group.set_planner_id('RRTConnectkConfigDefault')
    self.right_group.set_planning_time(10)
    self.right_group.set_num_planning_attempts(3)

    self.groups = {'left': self.left_group, 'right': self.right_group}

    self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)

    self.left_planning_frame = self.left_group.get_planning_frame()
    self.left_eef_link = self.left_group.get_end_effector_link()

    self.right_planning_frame = self.right_group.get_planning_frame()
    self.right_eef_link = self.right_group.get_end_effector_link()

    #self.NEUTRAL = np.array([
    neutral = {
               'left_e0':    -0.0015339807878854137,
               'left_e1':    0.744364177321397,
               'left_s0':    0.0019174759848567672,
               'left_s1':    -0.54686415088115,
               'left_w0':    -0.0015339807878854137,
               'left_w1':    1.2540292940963258,
               'left_w2':    0.0019174759848567672
               }

    self.HOME = np.zeros(7)
    self.box_name = 'table'

    self.simple_move_service = rospy.Service('simple_move', SimpleMove, self.handle_simple_move)

    rospy.sleep(rospy.Duration(3.0))

  def joint_goal(self, pose_goal=None, arm='left', dry_run=False):
    if arm == 'left':
      left_vals = [0.5606699779721187, -0.46824763550202253, -1.1524030668989171, 1.2808739578843205, -2.147189607842608, -1.3491361029452213, -0.559902987578176]
      self.left_group.go(left_vals, wait=True)
      self.left_group.stop()
    elif arm == 'right':
      right_vals = [0.048703890015361885, -0.8605632220037172, -0.28186896977394477, 1.3640924156271041, -2.7262673552693517, -1.0354370318226542, -0.45904375077471005]
      self.right_group.go(right_vals, wait=True)
      self.right_group.stop()


  def move_arm(self, pose_goal=None, arm='left', dry_run=False):
    group = self.groups[arm]

    if pose_goal is None:
      trans = self.tf_buffer.lookup_transform('base', 'table_center', rospy.Time.now()-rospy.Duration(0.1)).transform.translation
      pose_goal = geometry_msgs.msg.Pose()
      pose_goal.orientation.x = 1.0
      pose_goal.orientation.w = 0.0
      pose_goal.position.x = trans.x
      pose_goal.position.y = trans.y
      pose_goal.position.z = trans.z + 0.4
      if arm == 'left':
        pose_goal.position.y += 0.7
      else:
        pose_goal.position.y -= 0.7
    group.set_pose_target(pose_goal)

    if dry_run:
      plan = group.plan()
      self.display_trajectory(plan)
      return True
    else:
      plan = group.go(wait=True)
      group.stop()
      group.clear_pose_targets()
      current_pose = group.get_current_pose().pose
      return all_close(pose_goal, current_pose, 0.01)

  def add_table(self, timeout=10):
    HEIGHT = 0.74
    box_pose = geometry_msgs.msg.PoseStamped()
    box_pose.header.frame_id = 'world'
    box_pose.header.stamp = rospy.Time.now()
    trans = self.tf_buffer.lookup_transform('base', 'table_center', rospy.Time.now()-rospy.Duration(0.1)).transform.translation
    box_pose.pose.position.x = trans.x
    box_pose.pose.position.y = trans.y
    box_pose.pose.position.z = trans.z - (HEIGHT/2) - 0.03
    size = (0.61, 1.22, HEIGHT)
    
    self.scene.add_box(self.box_name, box_pose, size=size)
    return self.wait_for_state_update(box_is_known=True, timeout=timeout)

  def handle_simple_move(self, req):
    def convert_to_good_pose(pose):
      transform = self.tf_buffer.lookup_transform('world', 'table_center', rospy.Time(0), rospy.Duration(1.0))
      pose = tf2_geometry_msgs.do_transform_pose(pose, transform).pose
      pose.position.z = -0.00
      pose.position.x += 0.015
      pose.position.x = max(0.40, pose.position.x)
      pose.position.x = min(0.79, pose.position.x)
      pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(np.pi,0,req.yaw))
      return pose
    
    waypoints = []
    start_pose = convert_to_good_pose(req.start_pose)
    end_pose = convert_to_good_pose(req.end_pose)

    pre_pose = copy.deepcopy(start_pose)
    pre_pose.position.z = 0.3

    pre1_pose = copy.deepcopy(start_pose)
    pre1_pose.position.z = 0.2

    pre2_pose = copy.deepcopy(start_pose)
    pre2_pose.position.z = 0.1

    #print(pre_pose.position.x)
    # hack to make it not get jammed into is body as much
    x = pre_pose.position.x
    if pre_pose.position.x < 0.45:
      pre_pose.position.x = 0.45
      pre_pose.position.z = 0.2
      pre1_pose.position.x = x
      pre1_pose.position.z = 0.1
      pre2_pose.position.z = 0.05


    post_pose = copy.deepcopy(end_pose)
    post_pose.position.z = 0.2

    if (req.start_pose.pose.position.y + req.end_pose.pose.position.y) / 2.0 >= 0.0:
      arm = 'left'
      other = 'right'
    else:
      arm = 'right'
      other = 'left'

    self.move_arm(pre_pose, arm=arm, dry_run=False)
    waypoints = [pre1_pose, pre2_pose, start_pose, end_pose, post_pose]
    plan, fraction = self.plan_push_path(waypoints, arm=arm, dry_run=False)
    success = fraction == 1.0

    self.joint_goal(arm=arm, dry_run=False)
    self.joint_goal(arm=other, dry_run=False)

    return SimpleMoveResponse(success)

  def plan_push_path(self, waypoints, arm='left', dry_run=False, jump=10.0):
    group = self.groups[arm]
    (plan, fraction) = group.compute_cartesian_path(
                                       waypoints,   # waypoints to follow
                                       0.01,        # eef_step
                                       jump)         # jump_threshold
    self.display_trajectory(plan)
    if not dry_run:
      self.execute_plan(plan, arm)
    return plan, fraction

  def display_trajectory(self, plan):
    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
    display_trajectory.trajectory_start = self.robot.get_current_state()
    display_trajectory.trajectory.append(plan)
    self.display_trajectory_publisher.publish(display_trajectory)

  def execute_plan(self, plan, arm='left'):
    group = self.groups[arm]
    group.execute(plan, wait=True)
    group.stop()

  def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
    start = rospy.get_time()
    seconds = rospy.get_time()
    while (seconds - start < timeout) and not rospy.is_shutdown():
      is_known = self.box_name in self.scene.get_known_object_names()

      if box_is_known == is_known:
        return True

      # Sleep so that we give other threads time on the processor
      rospy.sleep(0.1)
      seconds = rospy.get_time()
    print('fail')
    return False


  def remove_table(self, timeout=4):
    self.scene.remove_world_object(self.box_name)
    return self.wait_for_state_update(box_is_known=False, timeout=timeout)


  def take_pushes(self):
    while True:

      sx, sy = map(float, raw_input('start x y: ').split())
      spose = Pose(Point(sx,sy,0.0), Quaternion(*tf.transformations.quaternion_from_euler(np.pi,0,0)))
      prpose = Pose(Point(sx,sy,0.2), Quaternion(*tf.transformations.quaternion_from_euler(np.pi,0,0)))

      ex, ey = map(float, raw_input('end x y: ').split())
      epose = Pose(Point(ex,ey,0.0), Quaternion(*tf.transformations.quaternion_from_euler(np.pi,0,0)))
      popose = Pose(Point(ex,ey,0.2), Quaternion(*tf.transformations.quaternion_from_euler(np.pi,0,0)))

      #jump = map(float, raw_input('jump: ').split())[0]

      self.move_arm(prpose, arm='left', dry_run=False)
      prpose.position.z = 0.1
      plan, fraction = self.plan_push_path([prpose, spose, epose, popose], arm='left', dry_run=False)#, jump=jump)
      print(fraction)

def main():
    node = CartesianPoseMoveitPlanner()
    node.remove_table()
    rospy.sleep(rospy.Duration(0.5))
    node.add_table()
    #node.move_arm(arm='left', dry_run=False)
    #node.move_arm(arm='right', dry_run=False)
    node.joint_goal(arm='left', dry_run=False)
    node.joint_goal(arm='right', dry_run=False)
    print('ready for simple_move')
    #node.take_cmds()
    #node.take_pushes()
    rospy.spin()

if __name__ == '__main__':
  main()
