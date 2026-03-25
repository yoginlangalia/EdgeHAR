 _____    _            _   _    _    ____  
| ____|__| | __ _  ___| | | |  / \  |  _ \ 
|  _| / _` |/ _` |/ _ \ |_| | / _ \ | |_) |
| |__| (_| | (_| |  __/  _  |/ ___ \|  _ < 
|_____\__,_|\__, |\___|_| |_/_/   \_\_| \_\
            |___/                           
EdgeHAR — Human Activity RecognitionA deep learning project to recognize human activities using a CNN-LSTM model, an ESP32 microcontroller, and a Streamlit web dashboard.This is a machine learning project that classifies 6 different human activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) based on accelerometer and gyroscope data. It includes a trained deep learning model, a live dashboard to see the results, and code to send sensor data from an ESP32.📑 Table of ContentsHow It WorksProject FeaturesFolder StructureQuick StartStep-by-Step GuideModel DetailsProject ResultsESP32 SetupDocker SetupTech Stack🏗 How It WorksHere is a simple look at how data flows through the project:Plaintext┌──────────────┐    ┌────────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ Sensor Data  │───▶│ Clean up Data  │───▶│   CNN-LSTM   │───▶│ Predict    │───▶│  Dashboard   │
│   (IMU)      │    │  (Normalize)   │    │    Model     │    │ (Activity) │    │  (Streamlit) │
└──────────────┘    └────────────────┘    └──────────────┘    └────────────┘    └──────────────┘
       │                                                                              │
       │                    ┌───────────────────┐                                     │
       └────────────────────│  ESP32 Hardware   │─────────────────────────────────────┘
                            │ (Sends JSON Data) │
                            └───────────────────┘
✨ Project Features🧠 Deep Learning Model: Combines 3 Convolutional blocks with a 2-layer LSTM to understand time-based movement data.📊 Live Dashboard: A web app made with Streamlit to watch the model make predictions in real-time.📱 Hardware Connection: Arduino code included to simulate sending 6-axis sensor data from an ESP32 to the computer.🚀 Exporting: Saves the final model in ONNX and TorchScript formats so it can be used easily in other applications.🐳 Docker: Includes a Docker setup so you can run the whole project without installing dependencies directly on your computer.📁 Folder StructurePlaintextEdgeHAR/
├── data/
│   └── download_data.py          # Script to download the dataset
├── notebooks/
│   └── exploration.ipynb         # Jupyter notebook for checking out the data
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Code to load data for PyTorch
│   ├── model.py                  # The CNN-LSTM code
│   ├── train.py                  # Script to train the model
│   ├── evaluate.py               # Checks how well the model performs
│   └── export.py                 # Saves the model as ONNX/TorchScript
├── esp32/
│   └── sensor_simulator.ino      # Code to upload to the ESP32
├── dashboard/
│   └── app.py                    # The Streamlit web app
├── models/                       # Where saved models go
├── outputs/                      # Where graphs and scores go
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
🚀 Quick StartIf you just want to get it running fast:Bash# 1. Download this folder
git clone https://github.com/<your-username>/EdgeHAR.git
cd EdgeHAR

# 2. Install the required Python libraries
pip install -r requirements.txt

# 3. Run these three commands
python data/download_data.py      # Get the dataset
python src/train.py               # Train the model
streamlit run dashboard/app.py    # Start the website
📖 Step-by-Step Guide1. Download the DatasetBashpython data/download_data.py
This downloads the standard UCI HAR Dataset (about 60MB) and puts it in the data/ folder.2. Train the ModelBashpython src/train.py --epochs 50 --lr 0.001 --batch_size 64 --patience 10
--epochs: How many times it goes through the data (default: 50).--lr: Learning rate (default: 0.001).--batch_size: Number of samples processed at once (default: 64).--patience: Stops training early if the model isn't improving for 10 epochs.It saves the best model to models/best_model.pth.3. Test the ModelBashpython src/evaluate.py
This runs the test data through the model to see how accurate it is. It creates a confusion matrix and shows F1 scores.4. Export the ModelBashpython src/export.py
Converts the PyTorch model into ONNX (models/har_model.onnx) and TorchScript (models/har_model_scripted.pt) so it can run faster in the dashboard.5. Start the Web DashboardBashstreamlit run dashboard/app.py
Go to http://localhost:8501 in your browser. You can use the "Live Simulation" mode to watch the model guess activities in real-time.🧠 Model DetailsHere is the exact structure of our neural network:LayerShape OutputParametersInput(B, 6, 128)—Conv1d(6→64) + BN + ReLU + MaxPool(B, 64, 64)~1.3KConv1d(64→128) + BN + ReLU + MaxPool(B, 128, 32)~24.8KConv1d(128→256) + BN + ReLU(B, 256, 32)~99.1KLSTM(256→128, 2 layers)(B, 32, 128)~395.3KLinear(128→64) + ReLU + Dropout(B, 64)~8.3KLinear(64→6)(B, 6)~0.4KTotal~529K📊 Project Results(Note: Because of randomness in training, your exact numbers might be slightly different. We used seed=42 to keep it as consistent as possible.)MetricOur ResultTest Accuracy~92%+Macro F1 Score~0.91+Speed (Inference Time)<1ms per sample on CPUFinal File Size (ONNX)~2 MB📱 ESP32 SetupIf you have an ESP32 board, you can use it to send fake sensor data to the dashboard to see how hardware integration works.Open esp32/sensor_simulator.ino in the Arduino IDE.Select your board (Tools → Board → ESP32 Dev Module).Upload the code to the board.Keep the board plugged in via USB.In the Streamlit dashboard, select the serial port your ESP32 is connected to.The ESP32 will send JSON data 125 times a second that looks like this:{"ax":0.12,"ay":0.98,"az":0.05,"gx":0.01,"gy":-0.02,"gz":0.00,"ts":12345}🐳 Docker SetupIf you have Docker installed, you can skip installing Python libraries and just run these commands:Bash# Start the web dashboard
docker-compose up --build dashboard

# Run the training script inside a container
docker-compose --profile training run --rm trainer

# Shut everything down when you're done
docker-compose down
🛠 Tech StackMachine Learning: PyTorch (v2.0+)Web App: Streamlit, PlotlyData Handling: NumPy, Pandas, scikit-learnGraphs: Matplotlib, SeabornHardware: ESP32 (programmed with Arduino IDE)Communication: PySerial (for reading ESP32 data via USB)📄 LicenseThis project uses the standard MIT License. Feel free to use and modify it! See the source text for the full legal license.
</p>
