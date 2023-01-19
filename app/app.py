from fastai.vision.all import load_learner
import gradio as gr


vehicles_labels = (
'Aerial tramway', 
'Agricultural vehicles', 
'Airplanes', 
'Bicycles', 
'Boats', 
'Buses', 
'Cars', 
'Construction vehicles',
 'Emergency vehicles', 
 'Military vehicles ', 
 'Mining vehicles', 
 'Motorcycles', 
 'Public transportation', 
 'RVs', 
 'Specialty vehicles ', 
 'Trains', 
 'Trucks ', 
 'Watercrafts'
)

model = load_learner('models/vehicles-recognizer-v3.1.pkl')

def recognize_image(image):
  pred, idx, probs = model.predict(image)
  return dict(zip(vehicles_labels, map(float, probs)))


image = gr.inputs.Image(shape=(256,256))
label = gr.outputs.Label(num_top_classes=5)
examples = [
    'test_images/unknown_00.jpg',
    'test_images/unknown_01.jpg',
    'test_images/unknown_03.jpg'
    ]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False)