# Language Detection using Recurrent Neural Network (RNN)

[Language Detection App](https://kashiekay-language-detection-rnn.streamlit.app/)

## 📌 Project Overview

This project is an **end-to-end Language Detection application** built using a **Recurrent Neural Network (SimpleRNN)** and deployed with **Streamlit**. The model predicts the **language of a given input text** among **17 different languages** such as English, Hindi, French, Malayalam, Spanish, Tamil, Arabic, and more.

The project demonstrates how deep learning models can effectively understand and classify **natural language sequences** through a complete NLP pipeline.

**Project Highlights:**

* Multi-class text classification
* Handling class imbalance
* Complete NLP pipeline (training → evaluation → deployment)
* Simple and interpretable RNN-based model

## 🚀 Features

* End-to-end NLP deep learning workflow
* Accurate language detection using RNN
* Clean and interactive Streamlit interface
* Real-time text-based predictions

## 🧠 Model Architecture

The model is intentionally kept simple to focus on core RNN concepts.

**Architecture Components:**

* Embedding Layer
* SimpleRNN Layer
* Dropout Layer
* Dense Softmax Output Layer

**Flow:**
Embedding → SimpleRNN → Dropout → Dense (Softmax)

**Why SimpleRNN?**

* Easy to explain and interview-friendly
* Captures sequential patterns in text data
* Lightweight and fast for deployment

## 🧠 Project Structure

The project is divided into **three main components**:

### 1️⃣ Model Training

* Loading and preprocessing the Kaggle Language Detection dataset
* Text cleaning, tokenization, and sequence padding
* Converting text into numerical representations
* Building and training an RNN-based deep learning model
* Applying class weights to handle dataset imbalance

### 2️⃣ Model Prediction

* Using the trained RNN model to predict the **language of input text**
* Supports predictions across all languages present in the dataset
* Outputs the most probable language label with confidence

### 3️⃣ Streamlit Deployment

* Interactive web application built using Streamlit
* Users can enter custom text in any supported language
* Displays the predicted language in real time
* Makes the model accessible to non-technical users

## 📊 Dataset Information

* **Source:** Kaggle – Language Detection Dataset
  [https://www.kaggle.com/datasets/basilb2s/language-detection](https://www.kaggle.com/datasets/basilb2s/language-detection)
* **Total Languages:** 17
* **Total Samples:** 10,267
* **Columns:**

  * Text (input sentence)
  * Language (target label)

⚠️ The dataset is imbalanced, so **class weighting** is applied during training.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning / Deep Learning:**

  * TensorFlow / Keras
  * Recurrent Neural Networks (SimpleRNN)
* **Natural Language Processing (NLP):**

  * Tokenization
  * Sequence Padding
* **Data Processing:**

  * NumPy
  * Pandas
* **Model Evaluation:**

  * Scikit-learn
* **Web App & Deployment:**

  * Streamlit
  * 
## ⚙️ Installation Steps

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Adi3042/Language-Detection-using-RNN.git
cd Language-Detection-using-RNN
```

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

```bash
streamlit run app.py
```

## 🧪 Example

**Input:**

```text
यह एक अच्छा दिन है
```

**Output:**

```text
Predicted Language: Hindi
Confidence: 0.98
```

## 👤 Author

**Kaushik Das**
Machine Learning / Data Science Enthusiast

## ⭐ Acknowledgement

Thanks to Kaggle and the open-source community for providing datasets and libraries that made this project possible.

## 🚀 Connect With Me

* 📧 Email: [kudokaito.pd@gmail.com](mailto:kudokaito.pd@gmail.com)
* 🔗 LinkedIn: [https://www.linkedin.com/in/kaushik-das-919928317](https://www.linkedin.com/in/kaushik-das-919928317)
* 🐙 GitHub: [https://github.com/amiKaushik](https://github.com/amiKaushik)

⭐ If you found this project helpful, feel free to **star the repository** and share it with others learning Machine Learning and NLP.

This project is for **educational purposes only**.

