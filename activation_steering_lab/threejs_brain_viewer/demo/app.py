
import gradio as gr
from gradio_threejs_brain_viewer import threejs_brain_viewer


example = threejs_brain_viewer().example_value()

demo = gr.Interface(
    lambda x:x,
    threejs_brain_viewer(),  # interactive version of your component
    threejs_brain_viewer(),  # static version of your component
    # examples=[[example]],  # uncomment this line to view the "example version" of your component
)


if __name__ == "__main__":
    demo.launch()
