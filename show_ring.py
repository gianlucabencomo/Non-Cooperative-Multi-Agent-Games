import mujoco
from mujoco import viewer

# Load the model from the XML file
model = mujoco.MjModel.from_xml_path("./models/match.xml")
data = mujoco.MjData(model)

# Launch the built-in viewer (freeze-frame simulation)
viewer.launch(model, data)
