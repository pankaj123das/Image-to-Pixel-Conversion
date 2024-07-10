# Image-to-Pixel-Conversion

## Direct Pixel Extraction Approach
### Description
The direct pixel extraction approach involves iterating over the input image and extracting the RGB values of each pixel. This approach does not require any pre-trained models or complex algorithms and can be implemented using basic image processing libraries like OpenCV and NumPy. The output is a data structure containing the pixel information, such as a list or a NumPy array.

### Step 1: Set Up Your Environment
Install and Import Necessary Libraries:
- OpenCV,
- NumPy,
- pandas,
- matplotlib

### Step 2: Use OpenCV to load the image
Set the image path.
### Step 3: Execute the function for pixel extraction
Execute direct_pixel_extraction() function.
### Step 4. Get output
A NumPy array containing the RGB values of each pixel.

### Play with image pixel 

image Array represent (H,W,C) formate. ---->(height, width, channels)
images have 3 channels   RGB (read,green,blue)
pixel value range in between 0 to 255
max pixel value is 255 and min value is 0
for normalise in bettween 0 to 1 we dived pixcel value from 255.
#### Matplotlib Vs CV2 Numpy array
cv2 reads in channels as BGR, Matplotlib reads in channels as RGB

- Convert 3d array to 1d flatten array
- Distribution of pixel values
- Display RBG Channels for our image
- Converting from BGR to RGB in cv2 image
- Convert gray color image
- Resizing and Scaling of image
- Custom dimension resizing
- Sharping the image
- Bluring the image

#### CV2 kernels for image manipulation likes this 
- Identity = [[0,0,0],[0,1,0],[0,0,0]],
- Sharpen = [[0,-1,0],[-1,5,-1],[0,-1,0]],
- Box blur = 1/9[[1,1,1],[1,1,1],[1,1,1]],
- Gaussion blur = 1/16[[1,2,1],[2,4,2],[1,2,1]]

## Segmentation Model Approach

### Model 1 DeepLabV3 Segmentation Model

This approach uses a pre-trained DeepLabV3 model for image segmentation. The DeepLabV3 model is loaded from TensorFlow Hub and applied to the input image to generate a segmented output.

### Step 1: Set Up Your Environment
Install and Import Necessary Libraries:
- TensorFlow,
- Keras,
- NumPy,
- pandas,
- matplotlib

### Step 2: Define Model Architecture
- Load the pre-trained DeepLabV3 model
- Preprocess the input image
- Segmentation

### Step 3: Define the path to the input image and Excute Functions
 - Set the path to the input image
 - Load the DeepLabV3 model
 - Load and preprocess the image
 - Perform segmentation

### Step 4: Get output
- A segmented image where each pixel is assigned a class label
- Image visualization

### Step 5: Resize the segmented image to match the original image size
### Advantages and Disadvantages
- **Advantages**: 
  - State-of-the-art performance for semantic image segmentation.
  - Ability to capture complex spatial hierarchies in images.
  - Pre-trained models are readily available, which can be fine-tuned for specific tasks.
- **Disadvantages**: 
  - Requires more computational resources and may be slower compared to simpler models.
  - Pre-trained models might need fine-tuning to achieve optimal performance on specific datasets.

### Notes:
- **Model Weights**: The DenseNet201 model is used here as an example. DeepLabV3 can be found in various TensorFlow implementations or pre-trained models available online. You might need to adjust the script to load a specific DeepLabV3 model depending on its availability.
- **Preprocessing**: Ensure that the image is preprocessed according to the model's requirements. For DenseNet201, the preprocessing function tf.keras.applications.densenet.preprocess_input is used.
- **Output**: The script displays both the original and segmented images using matplotlib.

### MODEL 2. U-Net Segmentation Model
Here, we'll use a pre-trained U-Net model. However, note that TensorFlow/Keras does not provide a pre-trained U-Net directly. Instead, we'll use a simple U-Net architecture and load weights if available. For simplicity, we'll assume a model trained on a specific dataset. You may need to train a U-Net model on your own dataset for accurate results.

### Step 1: Set Up Your Environment
Install and Import Necessary Libraries:
- TensorFlow,
- Keras,
- NumPy,
- pandas,
- matplotlib

### Step 2: Define Model Architecture
- Define U-Net model architecture
- Perform segmentation
- Dummy data for training (replace with your dataset)

### Step 3: Train the Model for pre weights.
- Define weight the paths
- Load and compile the U-Net model
- Train the model (use your own dataset here)
- Save the trained weights

### Step 4: Define the path to the input image and Excute Functions
- Set path to the input image
- Load the image
- Perform segmentation

### Step 5. Get Output.
- A binary mask image representing the segmented regions
- image Visualization

### Step 6: Resize the segmented image to match the original image size
### Key Steps:
- **Training U-Net**: Train the U-Net model using a dummy dataset or your own dataset.
- **Saving Weights**: Save the trained model weights.
- **Loading Weights**: Load the saved weights to use the trained model for segmentation.

### Advantages and Disadvantages
- **Advantages**: Provides higher-level information by grouping pixels, useful for object detection and segmentation tasks.
- **Disadvantages**: Requires more computational resources and may be slower compared to direct pixel extraction.


### Notes:
- **Pre-trained Weights**: The model_weights_path should point to the location where you have the pre-trained weights of the U-Net model. If you don't have pre-trained weights, you'll need to train the U-Net model on a suitable dataset.
- **Image Path**: Make sure the image_path points to your input image.
- **Output**: The script saves the segmented image in the output directory and displays it using matplotlib.

## Advantages and Disadvantages in Direct Model and Segmented Model

### Direct Pixel Extraction
- **Advantages**: Simple and fast. Directly provides pixel values.
- **Disadvantages**: Does not provide higher-level information about the image content.

### Segmentation Model
- **Advantages**: Provides higher-level information by grouping pixels.
- **Disadvantages**: Requires more computational resources and may be slower.

## Conclusion
The image to pixel conversion can be approached in different ways depending on the requirements and computational resources available. The segmentation model with DeepLabV3,U-Net offers advanced capabilities for semantic segmentation, while the direct pixel extraction approach provides a straightforward method for obtaining raw pixel data.
