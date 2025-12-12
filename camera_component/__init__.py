import os
import streamlit.components.v1 as components

_component_dir = os.path.dirname(os.path.abspath(__file__))
_frontend_path = os.path.join(_component_dir, "frontend")

camera = components.declare_component(
    "camera",
    path=_frontend_path
)
