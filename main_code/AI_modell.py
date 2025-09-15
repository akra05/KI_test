
import random as rd
import numpy as np 
from PIL import Image
import os  
import math

# Pfad zum aktuellen Skript
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PARENT_DIR = os.path.dirname(BASE_DIR)
    # Pfad zu den MNIST-Bildern
MNIST_PATH = os.path.join(PARENT_DIR, "mnist_images")

    # Pfad zum Modell
MODEL_PATH = os.path.join(BASE_DIR, "class_matrix.npy")

layers = []
learn_rate = 0.01

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

    for idx, i in enumerate(list):
        loading_path = folder + "\\" + str(i)
        files = os.listdir(loading_path)
        filename = files[progress[i]]
        progress[i] += 1

        img = Image.open(os.path.join(loading_path, filename)).convert("L")
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        
        
        python_matrix = img_array.tolist()
        python_array = convert_to_array(python_matrix)

        
        #print(python_array)
        #print("---------------------------------------------------")

        calculate_linear_model(python_array, i)

        if idx % 100 == 0:
            print(f"Lade/Trainiere Bild {idx}/{len(list)}")

def forward_pass(array):
    global layers
    
    activations = [np.array(array)]  # Eingabe als np.array
    current = activations[0]

    for i, W in enumerate(layers):
        z = np.dot(W, current)  # W @ current
        if i < len(layers) - 1:
            current = np.maximum(0, z)  # ReLU
        else:
            current = z  # letzte Schicht ohne ReLU
        activations.append(current)

    output = calc_softmax(current)
    return output, activations


def relu(arr):
    for idx, i in enumerate(arr):
        if i < 0:
            arr[idx] = 0
    return arr

def relu_derivative(x):
    return (x > 0).astype(float)

def backprop(result_arr, activations, label_vector):
    # Delta am Ausgang
    delta = np.array(result_arr) - np.array(label_vector)
    
    # Letztes Layer updaten
    last_activation = activations[-2]  # Output vom vorletzten Layer
    grad_W = delta[:, None] * last_activation[None, :]
    layers[-1] -= learn_rate * grad_W

    # Delta für vorherige Layer propagieren
    for l in range(len(layers)-2, -1, -1):
        # delta vom nächsten Layer
        W_next = np.array(layers[l+1])
        delta = W_next.T @ delta  # shape: (neurons_in_layer_l,)
        delta *= relu_derivative(np.array(activations[l+1]))  # Apply ReLU derivative

        # Gradienten für aktuelle Layer
        grad_W = delta[:, None] * np.array(activations[l])[None, :]
        layers[l] -= learn_rate * grad_W
        

def calculate_linear_model(array, label):
    
    result_arr, activations = forward_pass(array)
    
    #print(return_val)
    #print(label)

    label_vector = [0]*10
    label_vector[label] = 1

    backprop(result_arr, activations, label_vector)
    '''
    for i in range(10):
        loss_for_class = result_arr[i] - label_vector[i]

        for j in range(784):
            layers[0][i][j] -= learn_rate * (loss_for_class * array[j])
    '''
    
    #print("Loss: " + str(loss))

# function to calculate softmax
def calc_softmax(arr):
    exp_vals = [math.exp(x) for x in arr]
    total = sum(exp_vals)
    return_arr = [0]*len(arr)
    for idx, val in enumerate(exp_vals):
        return_arr[idx] = val / total
    return return_arr

def create_hidden_layers(neurons, layers_number):
    global layers
    layers = []  # neu starten, falls schon was drin ist

    if layers_number == 0:
        # Input (784) -> Output (10)
        weight_matrix = np.random.uniform(-0.1, 0.1, size=(10, 784))
        layers.append(weight_matrix)
        
        print(weight_matrix.shape)  # (10, 784)
        return

    # Input (784) -> erste Hidden-Layer (neurons)
    first_matrix = np.random.uniform(-0.1, 0.1, size=(neurons, 784))
    layers.append(first_matrix)

    # Hidden -> Hidden
    for _ in range(layers_number - 1):
        hidden_matrix = np.random.uniform(-0.1, 0.1, size=(neurons, neurons))
        layers.append(hidden_matrix)

    # Letzte Hidden -> Output (10)
    last_matrix = np.random.uniform(-0.1, 0.1, size=(10, neurons))
    layers.append(last_matrix)

    

def main():
    
    print(BASE_DIR)
    print(MNIST_PATH)
    print(MODEL_PATH)
    
    length = 30000
    progress = [0]*10
    list = get_list(length)
    create_hidden_layers(16, 0)
    
    load_model(list,progress)
    
    np.save(MODEL_PATH, layers, allow_pickle=True)
    


if __name__ == "__main__":
    main()
