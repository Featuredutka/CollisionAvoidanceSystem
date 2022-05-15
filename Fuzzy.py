from re import L
import numpy as np
import skfuzzy as fuzz
# import matplotlib.pyplot as plt

person_dist = np.arange(50, 100, 1) #40000 to 100000 
person_direction = np.arange(0, 1, 1) # Binary?

robot_speed = np.arange(0, 1, 0.01) # Consider lower upper limit???
robot_relpos = np.arange(0, 3, 1) #Relative position compared to the center of the road

sign_det = np.arange(0,1,1) # Binary?

drd = np.arange(0, 1, 0.01)
# rds = np.arange(0, 1, 0.01) Needed? 

#Fuzzy membership functions for created vars

#INPUTS
# Person distance
pds_far = fuzz.trimf(person_dist, [80, 90, 10000])
pds_near = fuzz.trimf(person_dist, [70, 75, 80])
pds_close = fuzz.trimf(person_dist, [-10000, 60, 70])
# Person direction
pdr_left = fuzz.trimf(person_dist, [0, 0.5, 1])
pdr_right = fuzz.trimf(person_dist, [0.5, 1, 1])
# Robot speed
rs_low = fuzz.trimf(robot_speed, [-10000, 0.2, 0.3])
rs_moderate = fuzz.trimf(robot_speed, [0.3, 0.45, 0.6])
rs_fast = fuzz.trimf(robot_speed, [0.6, 0.7, 10000])
# Robot relative position
rp_left = fuzz.trimf(robot_relpos, [-10000, 0.3, 0.45])
rp_center = fuzz.trimf(robot_relpos, [0.4, 0.5, 0.6])
rp_right = fuzz.trimf(robot_relpos, [0.55, 0.7, 10000])
# Sign detected
# sd_no
# sd_yes

#OUTPUTS
# Desired robot speed
drs_low = fuzz.trimf(robot_speed, [-10000, 0.2, 0.3])
drs_moderate = fuzz.trimf(robot_speed, [0.3, 0.45, 0.6])
drs_fast = fuzz.trimf(robot_speed, [0.6, 0.7, 10000])
# Desired robot direction
drd_hl = fuzz.trimf(drd, [-10000, 0.2, 0.3])
drd_ml = fuzz.trimf(drd, [0.3, 0.45, 0.45])
drd_c = fuzz.trimf(drd, [0.45, 0.5, 0.55])
drd_mr = fuzz.trimf(drd, [0.55, 0.55, 0.7])
drd_hr = fuzz.trimf(drd, [0.7, 0.8, 10000])

def regulator(p_dist:int, p_drct:str, speed:float, position:float, sign:bool):

    # Defining ranges related to the output

    p_dist_far = fuzz.interp_membership(person_dist, pds_far, p_dist)
    p_dist_near = fuzz.interp_membership(person_dist, pds_near, p_dist)
    p_dist_close = fuzz.interp_membership(person_dist, pds_close, p_dist)

    speed_slow  = fuzz.interp_membership(robot_speed, rs_low, speed)
    speed_moderate = fuzz.interp_membership(robot_speed, rs_moderate, speed)
    speed_fast = fuzz.interp_membership(robot_speed, rs_fast, speed)

    position_left_border = fuzz.interp_membership(robot_relpos, rp_left, position)
    position_center  = fuzz.interp_membership(robot_relpos, rp_center, position)
    position_right_border = fuzz.interp_membership(robot_relpos, rp_right, position)

    # Applying rules

