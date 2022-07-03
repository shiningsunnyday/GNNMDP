import numpy as np


# sage
# time_list =[0.3460053539276123, 0.3460053539276123, 0.3460053539276123,
#             0.3460053539276123, 0.3460053539276123, 0.3460053539276123,
#             0.3460053539276123, 0.3460053539276123, 0.3460053539276123,
#             0.3460053539276123]
# time_list =[0.3406886792182922, 0.3406886792182922, 0.34114267826080324, 0.34114267826080324, 0.34114267826080324, 0.34114267826080324, 0.34114267826080324,
#             0.34114267826080324, 0.34114267826080324, 0.34114267826080324]
# time_list =[0.38523526191711427, 0.38523526191711427, 0.38523526191711427, 0.3657206153869629, 0.3657206153869629, 0.3657206153869629, 0.3657206153869629,
#             0.3657206153869629, 0.3657206153869629, 0.3657206153869629]
# time_list =[0.3399805188179016, 0.3402359962463379, 0.3402359962463379, 0.3402359962463379, 0.3402359962463379, 0.3402359962463379, 0.3402359962463379,
#             0.3402359962463379, 0.3402359962463379, 0.3402359962463379]
# time_list =[0.3407685685157776, 0.35454684019088745, 0.35519506692886355, 0.35519506692886355, 0.35519506692886355, 0.35519506692886355, 0.35519506692886355,
#             0.35004884958267213, 0.35004884958267213, 0.35004884958267213]

# time_list =[0.40982250452041624, 0.40386635780334473, 0.41630891799926756, 0.41630891799926756, 0.41630891799926756, 0.41630891799926756, 0.39719942331314084,
#             0.39719942331314084, 0.39719942331314084, 0.39719942331314084]
# time_list =[0.41474161148071287, 0.40633116483688353, 0.41107398986816407, 0.41107398986816407, 0.394621376991272, 0.394621376991272, 0.394621376991272,
#             0.394621376991272, 0.394621376991272, 0.394621376991272]
# time_list =[0.45813244581222534, 0.4954677939414978, 0.4954677939414978, 0.4954677939414978, 0.4954677939414978, 0.4954677939414978, 0.4954677939414978,
#             0.4954677939414978, 0.4954677939414978, 0.4954677939414978]
# time_list =[0.7050557708740235, 0.6805443358421326, 0.6805443358421326, 0.6921493697166443, 0.6921493697166443, 0.6921493697166443, 0.6921493697166443,
#             0.6921493697166443, 0.6921493697166443, 0.6921493697166443]
# time_list =[0.679630651473999, 0.6701735544204712, 0.6701735544204712, 0.6701735544204712, 0.6963061237335205, 0.6321158862113953, 0.6321158862113953,
#             0.6321158862113953, 0.6321158862113953, 0.6321158862113953]



# gcn
# time_list =[0.4977924990653992, 0.7771159553527832, 0.5572978854179382, 0.5572978854179382, 0.5572978854179382, 0.5572978854179382, 0.5572978854179382,
#             0.6649249696731567, 0.6649249696731567, 0.6649249696731567]
# time_list =[0.4480914330482483, 0.5167065644264222, 0.4630458974838257, 0.4630458974838257, 0.4630458974838257, 0.4630458974838257,
#             0.4630458974838257, 0.4711706590652466, 0.4711706590652466, 0.4711706590652466]
# time_list =[0.4549542737007141, 0.5942735362052918, 0.5942735362052918, 0.5942735362052918, 0.7830757713317871, 0.7830757713317871,
#             0.7830757713317871, 0.7830757713317871, 0.7830757713317871, 0.7830757713317871]
# time_list =[0.776667742729187, 0.776667742729187, 0.7920292377471924, 0.7920292377471924, 0.7920292377471924, 0.7920292377471924,
#             0.7920292377471924, 0.7920292377471924, 0.7920292377471924, 0.7920292377471924]
# time_list =[0.7938028907775879, 0.7773873949050903, 0.7773873949050903, 0.6974265813827515, 0.6974265813827515, 0.6974265813827515,
#             0.6974265813827515, 0.6974265813827515, 0.6974265813827515, 0.6974265813827515]
# time_list =[0.39967319250106814, 0.39967319250106814, 0.39967319250106814, 0.39967319250106814, 0.39967319250106814, 0.39967319250106814,
#             0.39967319250106814, 0.39967319250106814, 0.40084897994995117, 0.40084897994995117]
# time_list =[0.3951440715789795, 0.3951440715789795, 0.3951440715789795, 0.3951440715789795, 0.40053491115570067, 0.40053491115570067,
#             0.40053491115570067, 0.40053491115570067, 0.40053491115570067, 0.40053491115570067]
# time_list =[0.4085634708404541, 0.4085634708404541, 0.4085634708404541, 0.4085634708404541, 0.4085634708404541, 0.4085634708404541,
#             0.4085634708404541, 0.4085634708404541, 0.4085634708404541, 0.4085634708404541]
time_list =[0.403710298538208, 0.403710298538208, 0.4048572278022766, 0.3998813986778259, 0.3998813986778259, 0.40503292083740233,
            0.40503292083740233, 0.40503292083740233, 0.40503292083740233, 0.40503292083740233]


ave_time = np.mean(time_list)
print(ave_time)




