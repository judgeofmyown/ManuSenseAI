import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import random
import pandas as pd
import joblib

class PredictiveMaintenanceDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Predictive Maintenance Dashboard")
        self.root.geometry("900x600")

        self.streaming = False
        self.time_step = 0
        self.time_data = []
        self.vibration_data = []
        self.health_data = []
        self.data = pd.read_csv(r"models\data.csv")
        self.feature_cols = [col for col in self.data.columns if col not in ['number', 'time', 'RUL']]
        self.sensor_6_data = []
        self.sensor_6_iterator = 0
        self.setup_ui()
        self.model = joblib.load("./models/model_file_name.joblib")

    def setup_ui(self):
        # grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(1, weight=1)


        # top left block: Sensor data
        sensor_frame = ttk.LabelFrame(self.root, text="Sensor Data", padding=10)
        sensor_frame.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.vibration_label = ttk.Label(sensor_frame, text="Vibration: --")
        self.vibration_label.pack(anchor='w', pady=5)
        self.temp_label = ttk.Label(sensor_frame, text="Temperature; --")
        self.temp_label.pack(anchor='w', pady=5)


        # top right block: Device details
        device_frame = ttk.LabelFrame(self.root, text="Device Details", padding=10)
        device_frame.grid(row=0, column=2, sticky='nsew', padx=10, pady=10) 
        self.device_name_label = ttk.Label(device_frame, text="Device Name: --")
        self.device_name_label.pack(anchor='w', pady=5)
        self.device_id_label = ttk.Label(device_frame, text="Device ID: --")
        self.device_id_label.pack(anchor='w', pady=5)
        self.status_label = ttk.Label(device_frame, text="Status: --", font=('Helvetica', 12, 'bold'))
        self.status_label.pack(anchor='w', pady=5)


        # Bottom block: Plot area
        plot_frame = ttk.LabelFrame(self.root, text="graph", padding=10)
        plot_frame.grid(row=1, column=0, columnspan=2, sticky='nsew', padx=10, pady=10)
        self.fig1, self.ax1 = plt.subplots(figsize=(7, 3))
        self.ax1.set_title("palceholder")
        self.ax1.set_xlabel("Time Step")
        self.ax1.set_ylabel("Vibration")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plot_frame)
        self.canvas1.get_tk_widget().pack(fill='both', expand=True)


        # health block
        health_frame = ttk.LabelFrame(self.root, text="Health Indicator", padding=10)
        health_frame.grid(row=1, column=2, sticky='nsew', padx=10, pady=10)
        self.fig2, self.ax2 = plt.subplots(figsize=(5, 3))
        self.ax2.set_title("palceholder")
        self.ax2.set_xlabel("Time Step")
        self.ax2.set_ylabel("Vibration")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=health_frame)
        self.canvas2.get_tk_widget().pack(fill='both', expand=True)

        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        self.start_btn = ttk.Button(button_frame, text="Start Monitoring", command=self.start_monitoring)
        self.start_btn.pack(side='left', padx=10)
        self.stop_btn = ttk.Button(button_frame, text="Stop Monitoring", command=self.stop_monitoring)
        self.stop_btn.pack(side='left', padx=10)
    
    def update_data(self):
        if self.streaming:
            new_time = self.time_step
            new_vibration = random.uniform(0, 1)
            new_health = random.uniform(0, 1)

            self.time_data.append(new_time)
            # self.vibration_data.append(new_vibration)
            # self.health_data.append(new_health)
            sensor_6_val = self.data.iloc[self.sensor_6_iterator, 6]
            self.sensor_6_data.append(sensor_6_val)
            # test
            # --prediction logic--
            input_features = self.data.loc[self.sensor_6_iterator, self.feature_cols].values.reshape(1, -1)
            input_df = pd.DataFrame(input_features, columns=self.feature_cols)
            predicted_rul = self.model.predict(input_df)[0]
            self.health_data.append(predicted_rul)

            self.vibration_label.config(text=f"Sensor 6 RMS: {sensor_6_val:.3f}")

            self.device_name_label.config(text="Device Name: TurboFan #3")
            self.device_id_label.config(text="Device ID: TF3X-991")

            # --- Status Alerts ---
            if predicted_rul < 30:
                self.status_label.config(text=f"Status: CRITICAL - RUL {predicted_rul:.1f}", foreground="red")
            elif predicted_rul < 70:
                self.status_label.config(text=f"Status: WARNING - RUL {predicted_rul:.1f}", foreground="orange")
            else:
                self.status_label.config(text=f"Status: HEALTHY - RUL {predicted_rul:.1f}", foreground="green")
            
            self.ax1.clear()
            self.ax1.plot(self.time_data, self.sensor_6_data, color='blue')
            self.ax1.set_title("Sensor 6 Rolling rms Over time")
            self.ax1.set_xlabel("Time Step")
            self.ax1.set_ylabel("Sensor 6 RMS")

            self.ax2.clear()
            self.ax2.plot(self.time_data, self.health_data, color='green')
            self.ax2.set_title("Predicted RUL Over time")
            self.ax2.set_xlabel("Time Step")
            self.ax2.set_ylabel("Remaining Useful Life (RUL)")

            self.canvas1.draw()
            self.canvas2.draw()

            self.time_step += 1
            self.sensor_6_iterator += 1
            self.root.after(1000, self.update_data) # 1 second interval

    def start_monitoring(self):
        if not self.streaming:
            self.streaming = True
            self.update_data()
    
    def stop_monitoring(self):
        self.streaming = False

    def run(self):
        self.root.mainloop()
