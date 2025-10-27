import math
import numpy as np

def deg2rad(d):
    """Ubah derajat ke radian"""
    return d * math.pi / 180.0

def rad2deg(r):
    """Ubah radian ke derajat"""
    return r * 180.0 / math.pi

def H_z(theta_deg, L=0):
    """Matriks transformasi homogen 2D (4x4)"""
    th = deg2rad(theta_deg)
    H = np.array([
        [math.cos(th), -math.sin(th), 0, L * math.cos(th)],
        [math.sin(th),  math.cos(th), 0, L * math.sin(th)],
        [0,             0,            1, 0],
        [0,             0,            0, 1]
    ])
    return H

def fk_2d_homogeneous(theta1_deg, theta2_deg, L1, L2):
    """Menghitung posisi ujung lengan (end effector)"""
    H0 = np.eye(4)
    H1 = H_z(theta1_deg, L1)
    H2 = H_z(theta2_deg, L2)

    H01 = H0.dot(H1)
    H02 = H01.dot(H2)

    joint0 = np.array([0.0, 0.0, 0.0])
    joint1 = H01[:3, 3]
    joint2 = H02[:3, 3]

    return {
        'joints': [joint0, joint1, joint2],
        'endpoint': joint2
    }

def ik_2d(x, y, L1, L2):
    d_sq = x**2 + y**2
    d = math.sqrt(d_sq)

    if d > (L1 + L2) or d < abs(L1 - L2):
        return {'reachable': False, 'solutions': []}

    cos_q2 = (d_sq - L1**2 - L2**2) / (2 * L1 * L2)
    cos_q2 = max(-1.0, min(1.0, cos_q2))
    q2_a = math.acos(cos_q2)
    q2_b = -q2_a

    k1 = L1 + L2 * cos_q2
    k2a = L2 * math.sin(q2_a)
    k2b = L2 * math.sin(q2_b)

    q1_a = math.atan2(y, x) - math.atan2(k2a, k1)
    q1_b = math.atan2(y, x) - math.atan2(k2b, k1)

    sol1 = (rad2deg(q1_a), rad2deg(q2_a))
    sol2 = (rad2deg(q1_b), rad2deg(q2_b))

    return {
        'reachable': True,
        'solutions': [sol1, sol2]
    }

import matplotlib.pyplot as plt

def plot_fk_2d(joints, title='Visualisasi Forward Kinematics (2D)', savepath=None):
    joints = np.array(joints)
    plt.figure(figsize=(6,6))
    plt.plot(joints[:,0], joints[:,1], '-o', linewidth=3, markersize=8)
    plt.scatter([0],[0], color='red', label='Base (0,0)')
    plt.axis('equal')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()

    if savepath:
        plt.savefig(savepath, dpi=200)
        print(f"Gambar disimpan ke: {savepath}")
    else:
        plt.show()


if __name__ == '__main__':
    L1 = 26.0   # femur
    L2 = 69.0   # tibia
    theta1 = 40.0
    theta2 = 30.0

    print("FORWARD KINEMATICS (FK)")
    res = fk_2d_homogeneous(theta1, theta2, L1, L2)
    endpoint = res['endpoint']
    print(f"Input sudut: θ1 = {theta1}°, θ2 = {theta2}°")
    print(f"Panjang link: L1 = {L1}, L2 = {L2}")
    print(f"Posisi ujung (x, y, z): {endpoint[0]:.4f}, {endpoint[1]:.4f}, {endpoint[2]:.4f}")

    print("\nINVERSE KINEMATICS (IK) ")
    ik = ik_2d(endpoint[0], endpoint[1], L1, L2)
    print(f"Reachable: {ik['reachable']}")
    if ik['reachable']:
        print("Solusi sudut (derajat):")
        print(f"  Solusi 1: θ1 = {ik['solutions'][0][0]:.4f}°, θ2 = {ik['solutions'][0][1]:.4f}°")
        print(f"  Solusi 2: θ1 = {ik['solutions'][1][0]:.4f}°, θ2 = {ik['solutions'][1][1]:.4f}°")

    joints = res['joints']
    plot_fk_2d(joints, title='Visualisasi FK', savepath=None)