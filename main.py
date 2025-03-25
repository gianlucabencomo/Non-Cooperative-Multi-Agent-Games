import typer
import numpy as np
import mujoco
import cv2

from utils import create_video_writer, load_keyframes, show_video

def imitate(model, data, renderer, keyframes, frame_skip: int = 2, save: bool = False, show: bool = False):
    # Set up a free, following camera.
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 5
    mujoco.mj_resetDataKeyframe(model, data, 0)

    if save:
        video_writer = create_video_writer()

    frames = []
    total_frames = len(keyframes)
    
    for frame_index in range(0, total_frames, frame_skip):
        # Update the simulation state with the current keyframe.
        data.qpos[:] = keyframes[frame_index]

        # Move simulation 1 step forward
        mujoco.mj_step(model, data)
        camera.lookat = data.body("root").subtree_com

        # Update renderer
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        frames.append(pixels)

        if save:
            video_writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))

    if save:
        video_writer.release()

    if show:
        show_video(frames)

def main(seed: int = 0, model: str = "./models/simple.xml", keyframe_file: str = "./keyframes/14_01.txt", show: bool = False, save: bool = False):
    np.random.seed(seed)

    # Load XML files + mujoco setup
    with open(model, "r") as f:
        xml = f.read()
    model = mujoco.MjModel.from_xml_string(xml)  # Static definition of the physics model.
    data = mujoco.MjData(model)  # Real-time simulation state.
    renderer = mujoco.Renderer(model, width=1920, height=1080)  # Visual output.
    keyframes = load_keyframes(keyframe_file)

    imitate(model, data, renderer, keyframes, 2, save, show)

if __name__ == '__main__':
    typer.run(main)