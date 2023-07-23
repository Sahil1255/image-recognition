# importing the required library  
from imageai.Detection import ObjectDetection  
  
# instantiating the class  
recognizer = ObjectDetection()  
  
# defining the paths  
path_model = "C:\\Users\sahil\\OneDrive\\Desktop\\test\\venv"  
path_input = "C:\\Users\\sahil\\OneDrive\\Desktop\\test\\input\\car.jpg"  
path_output = "C:\\Users\\sahil\\OneDrive\\Desktop\\test\\output\\newimage.jpg"  
  
# using the setModelTypeAsTinyYOLOv3() function  
recognizer.setModelTypeAsTinyYOLOv3()  
# setting the path of the Model  
recognizer.setModelPath(path_model)  
# loading the model  
recognizer.loadModel()  
# calling the detectObjectsFromImage() function  
recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
    )  
  
# iterating through the items found in the image  
for eachItem in recognition:  
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])