import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib as mp
import matplotlib.pyplot as plt

mp.use("QtCairo")
# INPUT VARIABLES
person_dist = ctrl.Antecedent(np.arange(50, 101, 1),'person_dist') #40000 to 100000 
robot_speed = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'robot_speed') # Consider lower upper limit???
robot_relpos = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'robot_relpos') #Relative position compared to the center of the road

person_dir = ctrl.Antecedent(np.arange(0, 3, 0.01),'person_dir')
sign_det = ctrl.Antecedent(np.arange(0, 3, 0.01),'sign_det')

# OUTPUT VARIABLES
drd = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'desired_Direction')
drs = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'desired_Speed')

# Terms
person_dist.automf(names=['far', 'near', 'close']) # No need for adding 'none' - just pass the parameter when no person is there to condsider it far
robot_speed.automf(names=['low', 'moderate', 'fast'])
robot_relpos.automf(names=['left', 'center', 'right'])
drd.automf(names=['hl', 'ml', 'c', 'mr','hr'])
drs.automf(names=['low', 'moderate', 'fast'])

person_dir.automf(names=['left', 'right', 'none'])
sign_det.automf(names=['yes', 'no'])

#Fuzzy membership functions for created vars

#INPUTS
# Person distance
# person_dist['far'] = fuzz.trimf(person_dist.universe, [80, 90, 10000])
# person_dist['near'] = fuzz.trimf(person_dist.universe, [70, 75, 80])
# person_dist['close'] = fuzz.trimf(person_dist.universe, [-10000, 60, 70])
person_dist['far'] = fuzz.trimf(person_dist.universe, [90, 92, 10000])
person_dist['near'] = fuzz.trimf(person_dist.universe, [77, 79, 90.1])
person_dist['close'] = fuzz.trimf(person_dist.universe, [-10000, 71, 77.1])
# Robot speed NOT NEEDED?
# robot_speed['low'] = fuzz.trimf(robot_speed.universe, [-10000, 0.2, 0.3])
# robot_speed['moderate'] = fuzz.trimf(robot_speed.universe, [0.3, 0.45, 0.6])
# robot_speed['fast'] = fuzz.trimf(robot_speed.universe, [0.6, 0.7, 10000])
# Robot relative position
robot_relpos['left'] = fuzz.trimf(robot_relpos.universe, [-10000, 0.2, 0.48])
robot_relpos['center'] = fuzz.trimf(robot_relpos.universe, [0.4, 0.5, 0.6])
robot_relpos['right'] = fuzz.trimf(robot_relpos.universe, [0.53, 0.8, 10000])
# Person direcrion
person_dir['left'] = fuzz.trimf(person_dir.universe, [-10000, 0, 0])
person_dir['right'] = fuzz.trimf(person_dir.universe, [-10000, 2, 2])
person_dir['none'] = fuzz.trimf(person_dir.universe, [2, 2, 10000])
# Sign detection
sign_det['no'] = fuzz.trimf(sign_det.universe, [-10000, 2, 2])
sign_det['yes'] = fuzz.trimf(sign_det.universe, [2, 2, 10000])

#OUTPUTS
# Desired robot speed
drs['stop'] = fuzz.trimf(robot_speed.universe, [-10000, 0.0, 0.0])
drs['low'] = fuzz.trimf(robot_speed.universe, [0.01, 0.1, 0.2])
drs['moderate'] = fuzz.trimf(robot_speed.universe, [0.2, 0.4, 0.6])
drs['fast'] = fuzz.trimf(robot_speed.universe, [0.5, 0.6, 10000])
# Desired robot direction
drd['hl'] = fuzz.trimf(drd.universe, [-10000, 0.2, 0.3])
drd['ml'] = fuzz.trimf(drd.universe, [0.2, 0.35, 0.45])
drd['c'] = fuzz.trimf(drd.universe, [0.43, 0.5, 0.57])
drd['mr'] = fuzz.trimf(drd.universe, [0.55, 0.65, 0.8])
drd['hr'] = fuzz.trimf(drd.universe, [0.7, 0.8, 10000])

"""
  HARD RULES
  1) If the person is close then stop ###(add another state for output speed)
  2) If the robot is close to the left or right border then go to the opposite direction
  3) If no person is detected (person dist is far) then speed is fast 
  MEDIUM RULES
  4) If the person is near or the sign is detected then speed is moderate ###(sign detected is bool nd this is have to be taken care of)
  5) If the person is near then go to the opposite direction of the person's movement
  6) 

"""
#1
rule1 = ctrl.Rule(person_dist['close'], drs['stop'])
#2
rule2 = ctrl.Rule(robot_relpos['left'], drd['hr'])
rule3 = ctrl.Rule(robot_relpos['right'], drd['hl'])
rule4  = ctrl.Rule(robot_relpos['center'], drd['c'])
#3
rule5 = ctrl.Rule(person_dist['far'], drs['fast'])
#4 IN PROGRESS (BITWISE LOGIC)
rule6 = ctrl.Rule(person_dist['near'] | sign_det['yes'], drs['moderate'])
#5 IN PROGRESS (BITWISE LOGIC)
rule7 = ctrl.Rule(person_dist['near'] & person_dir['left'], drd['mr'])
rule8 = ctrl.Rule(person_dist['near'] & person_dir['right'], drd['ml'])

rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]

tipping_ctrl = ctrl.ControlSystem(rules)
tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

def fuz_reg(p_dist: int, p_dir: float, r_relpos: float, sign_det: int) -> list:
    tipping.input['person_dist'] = p_dist
    # tipping.input['robot_speed'] = 0.5 # not sure if I need to pass speed as a reg parameter
    tipping.input['robot_relpos'] = r_relpos

    tipping.input['person_dir'] = p_dir
    tipping.input['sign_det'] = sign_det

    tipping.compute()
    print(drs)
    # drs.view(sim=tipping)

    return [tipping.output['desired_Direction'], tipping.output['desired_Speed']]
