import typer
import mujoco
import numpy as np

from tqdm.auto import tqdm
from utils import load_keyframes

def find_best_vertical_offset(model, data, key, tol=1e-8, max_iter=200, height_bounds=(-4, 4)):
    low, high = height_bounds

    def vertical_force(offset):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[:] = key
        data.qpos[2] += offset  # Apply vertical offset
        mujoco.mj_forward(model, data)  # Compute forward dynamics
        data.qacc = 0
        mujoco.mj_inverse(model, data)  # Compute inverse dynamics
        return data.qfrc_inverse[2]  # Return vertical force at index 2
    

    force_low = vertical_force(low)
    force_high = vertical_force(high)

    if np.sign(force_low) == np.sign(force_high):
        raise ValueError("Vertical force does not change sign within bounds. Adjust the search interval.")

    for i in range(max_iter):
        mid = (low + high) / 2
        force_mid = vertical_force(mid)

        if abs(force_mid) < tol:
            return mid  # Converged to the optimal offset

        # Narrow down the interval
        if np.sign(force_low) != np.sign(force_mid):
            high = mid
        else:
            low = mid
            force_low = force_mid

        # Check convergence
        if abs(high - low) < tol:
            return (low + high) / 2
        
    print("Bisection method did not converge within the maximum number of iterations.")
    return (low + high) / 2  # Return the best approximation

def correct_height(model, data, keyframes, zero_origin: bool = True):
    for i in tqdm(range(len(keyframes))):
        offset = find_best_vertical_offset(model, data, keyframes[i])
        keyframes[i][2] += offset
        if zero_origin:
            keyframes[i][0] = 0
            keyframes[i][1] = 0
    return keyframes

def load_model(dp: str, width: int = 1920, height: int = 1080):
    with open(dp, "r") as f:
        xml = f.read()
    model = mujoco.MjModel.from_xml_string(xml)  # static definition of physics model
    data = mujoco.MjData(model)  # real-time, evolving simulation state
    renderer = mujoco.Renderer(model, width=width, height=height)
    return model, data, renderer


def main(
    seed: int = 0,
    model_dp: str = "./models/simple.xml",
    keyframes_dp: str = "./keyframes/simple.txt",
):
    print(keyframes_dp)
    np.random.seed(seed)
    model, data, _ = load_model(model_dp)
    keyframes = load_keyframes(keyframes_dp)

    keyframes = correct_height(model, data, keyframes, zero_origin=True)

    with open(keyframes_dp, "w") as f:
        pass  # This will create or clear the file

    with open(keyframes_dp, "w") as f:
        for keyframe in keyframes:
            f.write(" ".join(map(str, keyframe)) + "\n")

if __name__ == "__main__":
    typer.run(main)
