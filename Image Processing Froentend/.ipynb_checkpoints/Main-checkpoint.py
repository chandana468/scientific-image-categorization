from tkinter import *
from tkinter import filedialog, messagebox
import os
import numpy as np
import joblib
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageTk

# ================= GLOBAL VARIABLES =================
filename = ""
X = None
Y = None
x_train = x_test = y_train = y_test = None
categories = []
model_folder = "model"

metrics_overall = []
class_metrics_storage = {}

# ================= TEXT CLEAR FUNCTION =================
def clearText():
    text.delete('1.0', END)

# ================= DATASET UPLOAD =================

def uploadDataset():
    clearText()
    global filename, categories

    filename = filedialog.askdirectory(initialdir="Dataset")
    if not filename:
        return

    text.insert(END, f"Folder Loaded:\n{filename}\n\n")

    categories = [
        d for d in os.listdir(filename)
        if os.path.isdir(os.path.join(filename, d))
    ]

    text.insert(END, "Subfolders found:\n")
    for label in categories:
        text.insert(END, f"- {label}\n")

# ===================================================
# ============ CREATE DATASET =======================
# ===================================================

def Image_Preprocessing():
    global X, Y

    if not filename:
        messagebox.showwarning("Warning", "Upload dataset first!")
        return

    IMG_SIZE = (64, 64)
    X = []
    Y = []

    text.insert(END, "\nCreating X.npy and Y.npy (RAW PIXELS)...\n")

    for label in categories:
        label_path = os.path.join(filename, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            print(img_path)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0
            img = img.flatten()

            X.append(img)
            Y.append(categories.index(label))

    X = np.array(X)
    Y = np.array(Y)

    os.makedirs(model_folder, exist_ok=True)
    np.save(os.path.join(model_folder, "X.txt.npy"), X)
    np.save(os.path.join(model_folder, "Y.txt.npy"), Y)

    text.insert(END, "Dataset created successfully!\n")
    text.insert(END, f"X shape: {X.shape}\n")
    text.insert(END, f"Y shape: {Y.shape}\n")

# ================= LOAD SAVED ARRAYS =================

def processDataset():
    clearText()
    global X, Y

    X_file = os.path.join(model_folder, "X.txt.npy")
    Y_file = os.path.join(model_folder, "Y.txt.npy")

    if not os.path.exists(X_file) or not os.path.exists(Y_file):
        text.insert(END, "\nX/Y not found. Creating dataset...\n")
        Image_Preprocessing()
        return

    text.insert(END, "\nLoading saved arrays...\n")
    X = np.load(X_file)
    Y = np.load(Y_file)

    text.insert(END, "Arrays loaded successfully!\n")
    text.insert(END, f"X shape: {X.shape}\n")
    text.insert(END, f"Y shape: {Y.shape}\n")

# ================= TRAIN TEST SPLIT =================

def trainTestSplit():
    clearText()
    global X, Y, x_train, x_test, y_train, y_test

    if X is None or Y is None:
        messagebox.showwarning("Warning","Please run Data Preprocessing first!")
        return

    text.insert(END, "\nPerforming Train-Test Split...\n")
    text.update()

    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.20, random_state=77
    )

    text.insert(END, "Train-Test Split Completed!\n")
    text.insert(END, f"x_train shape: {x_train.shape}\n")
    text.insert(END, f"x_test shape: {x_test.shape}\n")
    text.insert(END, f"y_train shape: {y_train.shape}\n")
    text.insert(END, f"y_test shape: {y_test.shape}\n")

# ================= METRICS =================

def calculateMetrics(model_name, predict, testY, labels):
    acc = accuracy_score(testY, predict) * 100
    prec = precision_score(testY, predict, average='macro') * 100
    rec = recall_score(testY, predict, average='macro') * 100
    f1 = f1_score(testY, predict, average='macro') * 100

    text.insert(END, f"\n{model_name} RESULTS\n")
    text.insert(END, "-" * 45 + "\n")
    text.insert(END, f"Accuracy  : {acc:.2f} %\n")
    text.insert(END, f"Precision : {prec:.2f} %\n")
    text.insert(END, f"Recall    : {rec:.2f} %\n")
    text.insert(END, f"F1-Score  : {f1:.2f} %\n\n")

    metrics_overall.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    report = classification_report(
        testY, predict, target_names=labels, output_dict=True
    )

    text.insert(END, "Class-wise Metrics:\n")

    for cls in labels:
        cls_prec = report[cls]['precision'] * 100
        cls_rec = report[cls]['recall'] * 100
        cls_f1 = report[cls]['f1-score'] * 100

        text.insert(
            END,
            f"{cls:25s}  "
            f"P: {cls_prec:6.2f}%  "
            f"R: {cls_rec:6.2f}%  "
            f"F1: {cls_f1:6.2f}%\n"
        )

        if cls not in class_metrics_storage:
            class_metrics_storage[cls] = []

        class_metrics_storage[cls].append({
            "Model": model_name,
            "Precision": cls_prec,
            "Recall": cls_rec,
            "F1-Score": cls_f1
        })

    cm = confusion_matrix(testY, predict)

    plt.figure(figsize=(5, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels
    )
    plt.title(f"{model_name} – Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# ================= MODELS =================

def trainMultinomialNB():
    clearText()
    if x_train is None:
        messagebox.showwarning("Warning", "Perform Train-Test Split first!")
        return

    model_path = os.path.join(model_folder, "Multi_NBC_Model.pkl")
    os.makedirs(model_folder, exist_ok=True)

    text.insert(END, "\nMultinomial Naive Bayes...\n")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(END, "Model loaded.\n")
    else:
        model = MultinomialNB()
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, "Model trained & saved.\n")

    y_pred = model.predict(x_test)
    calculateMetrics("Multinomial Naive Bayes", y_pred, y_test, categories)

def trainGaussianNB():
    clearText()
    if x_train is None:
        messagebox.showwarning("Warning", "Perform Train-Test Split first!")
        return

    model_path = os.path.join(model_folder, "naive_bayes.pkl")
    os.makedirs(model_folder, exist_ok=True)

    text.insert(END, "\nGaussian Naive Bayes...\n")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(END, "Model loaded.\n")
    else:
        model = GaussianNB()
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, "Model trained & saved.\n")

    y_pred = model.predict(x_test)
    calculateMetrics("Gaussian Naive Bayes", y_pred, y_test, categories)

def trainRandomForest():
    clearText()
    if x_train is None:
        messagebox.showwarning("Warning", "Perform Train-Test Split first!")
        return

    model_path = os.path.join(model_folder, "random_forest.pkl")
    os.makedirs(model_folder, exist_ok=True)

    text.insert(END, "\nRandom Forest Classifier...\n")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        text.insert(END, "Model loaded.\n")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=77, n_jobs=-1)
        model.fit(x_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, "Model trained & saved.\n")

    y_pred = model.predict(x_test)
    calculateMetrics("Random Forest Classifier", y_pred, y_test, categories)

# ================= MODEL COMPARISON =================

def modelComparison():
    clearText()
    if not metrics_overall or not class_metrics_storage:
        messagebox.showwarning("Warning", "Run models first!")
        return

    overall_df = pd.DataFrame(metrics_overall)
    overall_df.rename(columns={
        "Model": "Algorithm",
        "Accuracy": "Accuracy (%)",
        "Precision": "Precision (%)",
        "Recall": "Recall (%)",
        "F1-Score": "F1 Score (%)"
    }, inplace=True)

    x = np.arange(len(overall_df['Algorithm']))
    width = 0.2

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - 1.5*width, overall_df['Accuracy (%)'], width, label='Accuracy')
    bars2 = plt.bar(x - 0.5*width, overall_df['Precision (%)'], width, label='Precision')
    bars3 = plt.bar(x + 0.5*width, overall_df['Recall (%)'], width, label='Recall')
    bars4 = plt.bar(x + 1.5*width, overall_df['F1 Score (%)'], width, label='F1 Score')

    plt.xticks(x, overall_df['Algorithm'], rotation=45)
    plt.xlabel("Algorithm")
    plt.ylabel("Score (%)")
    plt.title("Overall Performance Comparison of Algorithms")
    plt.legend()

    plt.tight_layout()
    plt.show()

    text.insert(END, "Comparison Graph created successfully!\n")
# ================= PREDICTION =================

def predictImage():
    clearText()
    model_path = os.path.join(model_folder, "random_forest.pkl")
    if not os.path.exists(model_path):
        messagebox.showwarning("Warning", "Train model first!")
        return

    model = joblib.load(model_path)

    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten().reshape(1, -1)

    pred = model.predict(img)[0]
    label = categories[pred]

    text.insert(END, f"Prediction Result: {label}")

    img_show = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    plt.imshow(img_show)
    plt.title(f"Predicted: {label}")
    plt.axis('off')
    plt.show()

# ================= UI =================

main = Tk()
main.title("Automated Scientific Image Categorization for Domain Specific Knowledge Organization")

# window settings
screen_width = top.winfo_screenwidth()
screen_height = top.winfo_screenwidth()

main.geometry("1400x900")


# Get screen size
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()

# Set window to full screen size
main.geometry(f"{screen_width}x{screen_height}")

# Load and resize background image to screen size
bg_image = Image.open("image.jpeg")
bg_image = bg_image.resize((screen_width, screen_height), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Set background
bg_label = Label(main, image=bg_photo)
bg_label.image = bg_photo
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.lower()

font = ('times', 15, 'bold')
font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

Label(main, text="Automated Scientific Image Categorization for Domain Specific Knowledge Organization",
      bg="gold2", fg="black",
      font=font, height=3, width=120).place(x=0, y=5)

Button(main, text="Dataset", command=uploadDataset, font=ff,
       bg="#1ABC9C", fg="black", activebackground="#16A085").place(x=20, y=150)

Button(main, text="Image Preprocessing", command=processDataset, font=ff,
       bg="#3498DB", fg="white", activebackground="#2E86C1").place(x=20, y=200)

Button(main, text="Train-Test Split", command=trainTestSplit, font=ff,
       bg="#F1C40F", fg="black", activebackground="#D4AC0D").place(x=20, y=250)

Button(main, text="Multinomial NB", command=trainMultinomialNB, font=ff,
       bg="#E67E22", fg="white", activebackground="#CA6F1E").place(x=20, y=300)

Button(main, text="Gaussian NB", command=trainGaussianNB, font=ff,
       bg="#E74C3C", fg="white", activebackground="#C0392B").place(x=20, y=350)

Button(main, text="Train Random Forest", command=trainRandomForest, font=ff,
       bg="#8E44AD", fg="white", activebackground="#7D3C98").place(x=20, y=400)

Button(main, text="Model Comparison", command=modelComparison, font=ff,
       bg="#2ECC71", fg="black", activebackground="#27AE60").place(x=20, y=450)

Button(main, text="Predict Image", command=predictImage, font=ff,
       bg="#34495E", fg="white", activebackground="#2C3E50").place(x=20, y=500)


text = Text(main, height=20, width=70, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330, y=100)

main.mainloop()
