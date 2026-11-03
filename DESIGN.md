# WeatherVision interface

WeatherVision uses a light, minimal working dashboard with clear upload, prediction, comparison, and explanation flows.

The visual reference is a university exhibition caption: generous white space, a restrained green ink palette, clear sans-serif text, and one strong image. The image workspace and result panel form the main two-column composition, stacking on narrow screens. Standard Streamlit tabs separate Classify, Compare models, and About. Forest green indicates an action or selection; warm amber indicates a pending state. No performance claims appear before evaluation.

The signature detail is a compact, aligned four-row score distribution beside the prediction. Typography follows the Streamlit sans-serif system at readable sizes. Color and type are subordinate to the upload, model choice, and result. Focus states, responsive containers, image captions, readable tables, and plain recovery messages are required. Grad-CAM shows original and overlay with an explicit explanation of the model's center crop.

The classification workspace now groups the image and prediction within one white surface. Input-source segments select Examples, Upload, or Camera. Category pills open dataset examples immediately, and the preview preserves the image aspect ratio at a bounded height. The result uses aligned score bars with explicit percentages, concise model/device/timing metadata, and JSON download. Grad-CAM is requested after prediction and preserves its state. A separate action compares the three trained model families on the current photograph. Experiments presents held-out results; Project holds attribution, team details, and report downloads. The mobile layout stacks these work areas and retains visible labels and touch targets.
