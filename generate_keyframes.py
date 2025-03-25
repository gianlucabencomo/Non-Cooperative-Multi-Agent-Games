import numpy as np
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
from amc_parser import parse_asf, parse_amc

original = ["root", "lhipjoint", "lfemur", "ltibia", "lfoot", "ltoes", "rhipjoint", "rfemur", "rtibia", "rfoot", "rtoes", "lowerback", "upperback", "thorax", "lowerneck", "upperneck", "head", "lclavicle", "lhumerus", "lradius", "lwrist", "lhand", "lfingers", "lthumb", "rclavicle", "rhumerus", "rradius", "rwrist", "rhand", "rfingers", "rthumb"]
mirrored = ["root", "rhipjoint", "rfemur", "rtibia", "rfoot", "rtoes", "lhipjoint", "lfemur", "ltibia", "lfoot", "ltoes", "lowerback", "upperback", "thorax", "lowerneck", "upperneck", "head", "rclavicle", "rhumerus", "rradius", "rwrist", "rhand", "rfingers", "rthumb", "lclavicle", "lhumerus", "lradius", "lwrist", "lhand", "lfingers", "lthumb"]

def motion_to_keyframes(joints, motions, root_joint_name="root", flip: bool = False, simple: bool = False):
    keyframes = []
    SCALE = (1.0 / 0.45) * 2.54 / 100.0  # Convert ASF lengths to meters

    # Correction quaternion (90-degree rotation about X-axis for MuJoCo alignment)
    correction_quat = euler.euler2quat(np.pi / 2, 0, 0, axes='sxyz')

    for motion in motions:
        root_motion = motion[root_joint_name]  # [TX, TY, TZ, RX, RY, RZ]
        root_pos = np.array(root_motion[:3]) * SCALE
        root_euler = np.deg2rad(root_motion[3:])
        root_quat = euler.euler2quat(*root_euler, axes='sxyz')
        root_quat = quaternions.qmult(correction_quat, root_quat)  # Apply correction
        root_quat /= np.linalg.norm(root_quat)  # Normalize quaternion

        if flip:
            root_pos[0] = -root_pos[0]  # Mirror across YZ-plane

        qpos = list(root_pos) + list(root_quat)


        joint_names = mirrored if flip else original
        for joint_name in joint_names:
            if joint_name != root_joint_name and joint_name in motion:
                joint_angles = np.deg2rad(motion[joint_name])  # Convert angles to radians
                if flip and "l" in joint_name.lower():
                    # Mirror left joint to right
                    if joint_name in ["lclavicle"]:
                        joint_angles[0] = -joint_angles[0]  # Negate yaw (Ry)
                        joint_angles[1] = -joint_angles[1]  # Negate roll (Rz)
                    if joint_name in ["lhumerus", "lfemur"]:
                        joint_angles[1] = -joint_angles[1]  # Negate yaw (Ry)
                        joint_angles[2] = -joint_angles[2]  # Negate roll (Rz)
                    if joint_name in ["lwrist"]:
                       joint_angles[0] = -joint_angles[0]  # Negate yaw (Ry)
                    if joint_name in ["lhand", "lthumb", "lfoot"]:
                        
                        joint_angles[1] = -joint_angles[1]  # Negate roll (Rz)
                elif flip and "r" in joint_name.lower():
                    # Mirror right joint to left
                    if joint_name in ["rclavicle"]:
                        joint_angles[0] = -joint_angles[0]  # Negate yaw (Ry)
                        joint_angles[1] = -joint_angles[1]  # Negate roll (Rz)
                    if joint_name in ["rhumerus", "rfemur"]:
                        joint_angles[1] = -joint_angles[1]  # Negate yaw (Ry)
                        joint_angles[2] = -joint_angles[2]  # Negate roll (Rz)
                    if joint_name in ["rwrist"]:
                        
                        joint_angles[0] = -joint_angles[0]  # Negate yaw (Ry)
                    if joint_name in ["rhand", "rthumb", "rfoot"]:
                        joint_angles[1] = -joint_angles[1]  # Negate roll (Rz)
                        

                if joint_name in ["ltoes", "rtoes", "rwrist", "rhand", "rfingers", "rthumb", "lwrist", "lhand", "lfingers", "lthumb", "lowerneck", "upperneck"] and simple:
                    continue
                else:
                    qpos.extend(joint_angles)

        keyframes.append(qpos)

    return keyframes

if __name__ == "__main__":
    asf_path = './mocap/14.asf'  # Path to ASF file
    amc_path = './mocap/14_12.amc'  # Path to AMC file
    save_file = './keyframes/14_12.txt'

    joints = parse_asf(asf_path)
    motions = parse_amc(amc_path)

    keyframes = motion_to_keyframes(joints, motions, flip=False, simple=True)

    # Check keyframe consistency
    keyframe_length = len(keyframes[0]) if keyframes else 0
    print(f"Keyframe length: {keyframe_length}")
    print(f"Number of keyframes: {len(keyframes)}")

    # Save keyframes to a file for later use in MuJoCo
    with open(save_file, "w") as f:
        for keyframe in keyframes:
            f.write(" ".join(map(str, keyframe)) + "\n")

    print(f"Keyframes saved to {save_file}")