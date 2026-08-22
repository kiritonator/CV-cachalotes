import json
import math as m


def coordinates_calc(xp, yp):
    R = 6378137
    alpha = m.atan(37/(2 * 142)) #полуугол обзора камеры по вертикали картинки
    beta = m.atan(64/(2 * 142)) #по горизонтали

    with open('home/raspberrypiuser/latest_state.json', 'r', encoding='utf-8') as file:
        mavlink_data = json.load(file)

    N = mavlink_data['lat']
    E = mavlink_data['lon']
    H = mavlink_data['relative_alt_m']
    gamma = -mavlink_data['roll_rad']
    teta = mavlink_data['pitch_rad']
    yaw = mavlink_data['yaw_rad']

    L = H * m.tan(teta + (- yp + 0.5)/0.5 * alpha)
    D = H * m.tan(gamma + (xp - 0.5)/0.5 * beta)
    Ln = L * m.cos(yaw) - D * m.sin(yaw)
    Le = L * m.sin(yaw) + D * m.cos(yaw)

    res_n = N + Ln / (2 * m.pi * R) * 360
    res_e = E + Le / (2 * m.pi * R * m.cos(m.radians(N))) * 360
    return res_n, res_e
    '''
    with open('resultscoord.txt', 'w', encoding='utf-8') as file:
        file.write(res_n, res_e)
    '''