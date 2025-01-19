import os
import cv2
import time
import torch
import torch.nn as nn
import torchvision
import pickle
import collections

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



DERNIER_MODELE = -1     # Pas d'entraînement fait
#DERNIER_MODELE = 50     # Numéro du dernier modèle sauvegardé
MAX_EPOCH = 25

device = torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda')

#################################################################################
# Net
def fn(x):
    return torch.abs(2*x)

traced_fn = torch.jit.trace(fn, torch.rand(()))

class TracedModule(torch.nn.Module):
    def forward(self, x):
        x = x.type(torch.float32)
        return torch.floor(torch.sqrt(x) / 5.)
    
class ScriptModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, x):
        r = -x
        if int(torch.fmod(x, 2.0)) == 0.0:
            r = x / 2.0
        return r

def script_fn(x):
    z = torch.ones([1], dtype=torch.int64)
    for i in range(int(x)):
        z = z * (i + 1)
    return z

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Modules must be attributes on the Module because if you want to trace
        # or script this Module, we must be able to inherit the submodules'
        # params.
        self.traced_module = torch.jit.trace(TracedModule(), torch.rand(()))
        self.script_module = ScriptModule()

        print('traced_fn graph', traced_fn.graph)
        print('script_fn graph', script_fn.graph)
        print('TracedModule graph', self.traced_module.__getattr__('forward').graph)
        print('ScriptModule graph', self.script_module.__getattr__('forward').graph)

    def forward(self, x):
        # Call a traced function
        x = traced_fn(x)

        # Call a script function
        x = script_fn(x)

        # Call a traced submodule
        x = self.traced_module(x)

        # Call a scripted submodule
        x = self.script_module(x)

        return x

#################################################################################

#################################################################################
# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

#################################################################################


#################################################################################
# LeNet5
class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = nn.functional.softmax(logits, dim=1)
        return logits, probs

#################################################################################


#################################################################################
# AlexNet

class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#################################################################################

#################################################################################
# Augmentation de données 

# Transformations des données
def transformation_de_base(img):
    # transformation = transforms.Compose([transforms.ToTensor(),transforms.Resize((224, 224))])
    img = transforms.ToTensor()(img)
    img = transforms.Resize((224, 224))(img)
    return img

def transformation_flip_horizontal(img):
    # Transformation horizontale
    # transformation = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Resize((224, 224))])
    img = transforms.ToTensor()(img)
    img = transforms.RandomHorizontalFlip()(img)
    
    img = transforms.Resize((224, 224))(img)
    return img

#################################################################################



def recuperer_image(chemin:str, transform) :
    img = cv2.imread(chemin)                    # taille variable
    return transform(img)


def recuperer_data_train(tranform_list) :
    print("recuperer_data_train")
    t0 = time.time()
    current_path = os.getcwd()
    data_read = pd.read_csv(current_path + "/train.csv")

    data_train = pd.DataFrame(columns=["label", "image"])
    # data_train["label"] = (data_train["label"]-1).astype("category")
    # data_train["image"] = data_train["id"].apply(lambda nom : recuperer_image("train/"+str(nom)+".jpg"))

    for transform in tranform_list:
        augmented_data = pd.DataFrame(columns=["label", "image"])
        augmented_data["label"] = (data_read["label"]-1).astype("category")
        augmented_data["image"] = data_read["id"].apply(lambda nom : recuperer_image("train/"+str(nom)+".jpg", transform=transform))
        data_train = pd.concat([data_train, augmented_data], ignore_index=True)
        print(data_train.info())

    print(data_train.info())
    print("fin recuperer_data_train en", time.time()-t0, "s")
    return data_train

def recuperer_images_test() :
    return [{"image" : recuperer_image("test/"+fichier)} for fichier in os.listdir("test")]


def afficher_repartition(data:pd.DataFrame) :
    print("affichage répartition par classe")
    x = data["label"].to_list()
    frequence = collections.Counter(x)
    print(dict(frequence))
    plt.hist(x, range=(-0.5, 3.5), bins=4, color="yellow", edgecolor="black")
    plt.xlabel("Label")
    plt.ylabel("Nombre")
    plt.title("Nombre d'image par label")
    plt.show()


def entrainer_modele(model,  optimizer, data_train:pd.DataFrame, loss=torch.nn.CrossEntropyLoss(), epochs=MAX_EPOCH+1) :
    print("Entraînement du modèle", time.ctime())
    t0 = time.time()

    # X = np.array(data_train["image"].to_list())
    # y = np.array(data_train["label"].to_list())
    nb_data = len(data_train["label"].to_list())

    data_dict = data_train[["image", "label"]].to_dict("records")
    X_load = torch.utils.data.DataLoader(data_dict, batch_size=64, num_workers=1)

    result_train_loss= []
    result_train_predictions = []

    for epoch in range(DERNIER_MODELE+1, epochs):
        t0 = time.time()
        print(str(epoch+1) +  "/" + str(epochs))
        model.train()
        train_losses = []
        predictions_train = []
        for _, data in enumerate(tqdm(X_load)) :
            X_data, y_data = data["image"].to(device), data["label"].to(device)
            optimizer.zero_grad()
            sortie = model(X_data)
            loss_train = loss(sortie, y_data)
            loss_train.backward()
            optimizer.step()
            train_losses.append(loss_train)
            predictions_train.append((torch.argmax(sortie, dim=1) == y_data).sum())

            
        result_train_loss.append(torch.stack(train_losses).mean().item())
        result_train_predictions.append(100 * torch.stack(predictions_train).sum().item() / nb_data)
        print(time.ctime(), "fini en", time.time() - t0, "s, accuracy :", result_train_predictions[-1], "%, loss :", result_train_loss[-1])
        with open(f"model{epoch}_{model.__name__}.pkl", "wb") as f :
            pickle.dump(model, f)
        print()


    print("Fin entraînement du modèle en", time.time()-t0, "s")


    return result_train_loss, result_train_predictions


def test_modele(model, data_test:list) :
    print("Application du modèle sur le jeu de test")
    t0 = time.time()
    predictions = []

    X_load = torch.utils.data.DataLoader(data_test, batch_size=1, num_workers=1)
    for _, data in enumerate(tqdm(X_load)) :
        X_data = data["image"].to(device)
        sortie = model(X_data)
        predictions.append(torch.argmax(sortie).item())
    print("Fin prédictions en ", time.time()-t0, "s")
    return predictions

    for i in range(data_test.shape[0]):
        sortie = model(data_test[i])
        predictions.append(torch.argmax(sortie).item())
    print("Fin prédictions en ", time.time()-t0, "s")

if __name__ == "__main__" :
    transform_list= [transformation_de_base, transformation_flip_horizontal]
    data_train = recuperer_data_train(tranform_list=transform_list)
    #afficher_repartition(data_train)
    if DERNIER_MODELE >= 0 :
        with open(f"model{DERNIER_MODELE}.pkl", "rb") as f :
            model = pickle.load(f)
    else :
        #model = AlexNet()
        model = ResNet()

    #model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    result_train_loss, result_test_loss, result_train_predictions, result_test_predictions = entrainer_modele(model, optimizer, data_train)
    with open("model_save.pkl", "wb") as f :
        pickle.dump(model, f)

    """
    # faire la prédiction
    data_test = recuperer_images_test()
    res = test_modele(model, data_test)
    with open("res_save.pkl", "wb") as f :
        pickle.dump(res, f)

    resultats = pd.DataFrame()
    resultats["id"] = np.arange(4000, 5082)
    resultats["label"] = res
    resultats["label"] = resultats["label"] + 1
    resultats.to_csv("resultats.csv", sep=",", index=False)
    """


    # fig, axes = plt.subplots(1, 2)
    # axes[0].plot(result_train_loss, label="train_loss")
    # axes[0].plot(result_test_loss, label="test_loss")
    # axes[0].legend()

    # axes[1].plot(result_train_predictions, label="train_accuracy")
    # axes[1].plot(result_test_predictions, label="test_accuracy")
    # axes[1].legend()

    # plt.show()
