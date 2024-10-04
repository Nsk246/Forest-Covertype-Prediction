import gradio as gr
import numpy as np
import pickle

# Load the pre-trained model from the pickle file
with open('/content/tuned_random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Mapping of numerical values to forest cover types
cover_type_mapping = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}


def predict_forest_cover(Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, 
                         Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, 
                         Hillshade_9am, Hillshade_Noon, Hillshade_3pm, 
                         Horizontal_Distance_To_Fire_Points, selected_wilderness_areas, 
                         selected_soil_types):
    
    # Set the dummy Id value
    Dummy_Id = 1  # Fixed value for dummy Id
    
    # Wilderness Area: Create a binary array from the selected values
    Wilderness_Area = [1 if str(i) in selected_wilderness_areas else 0 for i in range(1, 5)]
    
    # Soil Type: Create a binary array from the selected values
    Soil_Types = [1 if str(i) in selected_soil_types else 0 for i in range(1, 41)]
    
    # Prepare the input feature array
    features = np.array([[Dummy_Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, 
                          Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, 
                          Hillshade_9am, Hillshade_Noon, Hillshade_3pm, 
                          Horizontal_Distance_To_Fire_Points] + Wilderness_Area + Soil_Types])
    
    # Predict the forest cover type
    prediction = model.predict(features)[0]
    
    # Return the forest cover type
    return cover_type_mapping[prediction]

# Define custom CSS for styling
css = """
body {
    background-color: #f7f7f7; /* Light background for the interface */
}

.input-container {
    background-color: #dff0d8; /* Light green background for input area */
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
}

.output-container {
    background-color: #f2dede; /* Light red background for output area */
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
}

h1, p {
    color: #31708f; /* Title and description text color */
}

input {
    font-size: 1rem;
}
"""

# Define the Gradio interface
interface = gr.Interface(
    fn=predict_forest_cover,
    inputs=[
        gr.Textbox(label="Elevation", lines=1, placeholder="Enter elevation"),
        gr.Textbox(label="Aspect", lines=1, placeholder="Enter aspect"),
        gr.Textbox(label="Slope", lines=1, placeholder="Enter slope"),
        gr.Textbox(label="Horizontal Distance To Hydrology", lines=1, placeholder="Enter horizontal distance to hydrology"),
        gr.Textbox(label="Vertical Distance To Hydrology", lines=1, placeholder="Enter vertical distance to hydrology"),
        gr.Textbox(label="Horizontal Distance To Roadways", lines=1, placeholder="Enter horizontal distance to roadways"),
        gr.Textbox(label="Hillshade 9am", lines=1, placeholder="Enter hillshade at 9am"),
        gr.Textbox(label="Hillshade Noon", lines=1, placeholder="Enter hillshade at noon"),
        gr.Textbox(label="Hillshade 3pm", lines=1, placeholder="Enter hillshade at 3pm"),
        gr.Textbox(label="Horizontal Distance To Fire Points", lines=1, placeholder="Enter horizontal distance to fire points"),
        gr.CheckboxGroup(label="Wilderness Area", choices=['1', '2', '3', '4']),
        gr.CheckboxGroup(label="Soil Type", choices=[str(i) for i in range(1, 41)])
    ],
    outputs=gr.Textbox(label="Predicted Forest Cover Type"),
    title="Forest Cover Type Prediction",
    description="Input features to predict the forest cover type using the pre-trained Random Forest model.",
    css=css,  # Apply custom CSS
)

# Launch the interface
interface.launch(share=True, debug=True)
