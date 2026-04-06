import tkinter as tk
from tkinter import ttk
import subprocess
import sys
import os
from TestScript.Integration.integration_test import integration_test

def run_app():
    def start_test():

        selected = script_var.get()
        script_path = script_map.get(selected)

        if not script_path or not os.path.exists(script_path):
            print("Status: Script not found")
            return

        if selected == "Testing All":
            integration_test()
        else:
            print(f"Status: Running [{selected}] ...")
            subprocess.Popen([sys.executable, script_path])

    root = tk.Tk()
    root.title("Computer Vision Project")
    root.geometry("450x320")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=0)

    script_frame = ttk.LabelFrame(root, text="Choose Test Script", padding=10)
    script_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    script_frame.columnconfigure(0, weight=1)

    script_var = tk.StringVar(value="Testing All")

    scripts = [
        "Testing All",
        "Food Vs Fruit",
        "Food Classification 1",
        "Food Classification 2",
        "Fruit Classification",
        "Binary Segmentation",
        "Multi Class Segmentation",
    ]

    script_map = {
        "Testing All": "TestScript/Integration/integration_test.py",
        "Food Vs Fruit": "TestScript/food_vs_fruit.py",
        "Food Classification 1": "TestScript/food_classification1.py",
        "Food Classification 2": "TestScript/food_classification2.py",
        "Fruit Classification": "TestScript/fruit_classification.py",
        "Binary Segmentation": "TestScript/binary_segmentation.py",
        "Multi Class Segmentation": "TestScript/multiclass_segmentation.py",
    }

    for i, script in enumerate(scripts):
        ttk.Radiobutton(
            script_frame,
            text=script,
            variable=script_var,
            value=script
        ).grid(row=i, column=0, sticky="w", pady=3)

    tk.Button(
        root,
        text="Start Test",
        bg="#9C27B0",
        fg="white",
        width=25,
        command=start_test
    ).grid(row=1, column=0, pady=20)

    root.mainloop()

run_app()
