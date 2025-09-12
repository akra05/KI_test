import random as rd
import numpy as np 
from PIL import Image
import os  
import math

class_matrix = [[rd.uniform(-0.5, 0.5) for _ in range(784)] for _ in range(10)]

def convert_to_array(matrix):
    arr = []

    for i in matrix:
        for j in i:
            arr.append(j)
    
    return arr

def get_list(length):
    #create a list of random numbers between 0 and 9 with the given length
    #this list is used to load random images from the dataset
    i = 0 
    arr = []
    while i < length:
        arr.append(rd.randint(0,9))
        i += 1 
    return arr

def load_model(list,progress):
    #path to the folder where the dataset is stored 
    folder = r"D:\Projekte\Zahlenerkennung\mnist_images"

    #loop which loads an image from the given folder for each number in the list 
    #calculates the linear model for each image and the corresponding label
    for i in list:
        loading_path = folder + "\\" + str(i)
        files = os.listdir(loading_path)
        filename = files[progress[i]]
        progress[i] += 1

        img = Image.open(os.path.join(loading_path, filename)).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        
        
        python_matrix = img_array.tolist()
        python_array = convert_to_array(python_matrix)

        #print(python_array)
        print("---------------------------------------------------")

        calculate_linear_model(python_array, i)

def calculate_linear_model(array, label):
    
    
    #init array where absolute probabilities are stored
    result_arr = [0]*10

    #calculate absolute probabilities with class_matrix and input array
    for i in range(10):
        sum = 0 
        for j in range(784):
            sum += class_matrix[i][j] * array[j]
        result_arr[i] = sum
    
    
    return_val = 0 #index of the highest value
    highest_value = 0 #highest value in the array

    #find the index of the highest value
    for i in result_arr:
        if i > highest_value:
            highest_value = i
            return_val = result_arr.index(i)
    
    #normalize the result array with softmax function
    result_arr = calc_softmax(result_arr)
    print(result_arr)
    
    #check if probabilities add up to 1
    sum = 0
    for i in result_arr:
        sum += i
    print(sum)
    print(return_val)
    print(label)

    label_vector = [0]*10
    label_vector[label] = 1

    loss = 0
    for i in range(10):
        loss += (result_arr[i] - label_vector[i])**2

    print("Loss: " + str(loss))

# function to calculate softmax
def calc_softmax(arr):
    e = math.e #get the euler's number
    sum = 0 
    return_arr = [0]*10
    #calculate divisor
    for i in arr:
        sum += e**i
    #calculate result array with values between 0 and 1
    for i in arr:
        return_arr[arr.index(i)] = (e**i)/sum

    return return_arr

def main():
    length = 1
    progress = [0]*10
    list = get_list(length)
    load_model(list,progress)
    


if __name__ == "__main__":
    main()