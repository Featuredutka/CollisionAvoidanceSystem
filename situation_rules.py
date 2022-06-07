def define_situation(left_coord:int, p_direction:int) ->int:
    ### ZONE CHECK
    zone = None
    if left_coord == 0:
        return None
    elif left_coord < 200:
        zone = 1
    elif left_coord < 400:
        zone = 2
    else:
        zone = 3

    # SITUATION CHECK BASED ON ZONE AND DIRECTION

    situation_map = { # situation dictionary {Key: Pair (zone , person_dir) Value: situation number} 
        (1, 0): 1,
        (1, 1): 2,
        (1, 2): 3,
        (2, 0): 4,
        (2, 1): 5,
        (2, 2): 6,
        (3, 0): 7,
        (3, 1): 8,
        (3, 2): 9,
    }

    situation = situation_map.get((zone, p_direction)) # None if not found
    
    return situation