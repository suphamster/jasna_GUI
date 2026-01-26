import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, Checkbutton
import subprocess
import os
import sys
import cv2
from PIL import Image, ImageTk
import threading
import shutil
import uuid
import time
import json
from datetime import datetime
from tkinterdnd2 import DND_FILES, TkinterDnD
import re
import numpy as np
from queue import Queue
import glob

class MosaicRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JASNA GUI 20260126 for jasna 0.3")
        self.root.geometry("1120x1000")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Fix path acquisition method
        if getattr(sys, 'frozen', False):
            # When running as an EXE (directory where the EXE itself is located)
            self.script_dir = os.path.dirname(sys.executable)
        else:
            # When running as a normal Python script
            self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Set path to jasna.exe
        self.jasna_path = os.path.join(self.script_dir, "jasna.exe")
        self.output_dir = os.path.join(self.script_dir, "output")
        self.log_file = os.path.join(self.script_dir, "LOG_JASNA_GUI.txt")
        
        # Paths for model weights - located near _internal folder, not inside it
        self.model_weights_dir = os.path.join(self.script_dir, "model_weights")
        
        # Paths inside _internal folder (only for ffmpeg)
        internal_dir = os.path.join(self.script_dir, "_internal")
        self.ffmpeg_bin_dir = os.path.join(internal_dir, "bin") if os.path.exists(internal_dir) else None
        if self.ffmpeg_bin_dir and os.path.exists(self.ffmpeg_bin_dir):
            self.ffmpeg_path = os.path.join(self.ffmpeg_bin_dir, "ffmpeg.exe")
            # Important: Add ffmpeg location to system PATH
            os.environ["PATH"] = self.ffmpeg_bin_dir + os.pathsep + os.environ["PATH"]
        else:
            self.ffmpeg_path = "ffmpeg"  # Use system ffmpeg
        
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop:DND_Files>>', self.drop_file)
        
        self.cap = None
        self.paused = True
        self.current_frame = 0
        self.video_fps = 30.0
        self.actual_fps = 30.0
        self.video_total_frames = 0
        self.process = None
        self.start_frame = 0
        self.end_frame = 0
        self.video_path = ""
        
        self.config_file = "config.ini"
        self.queue_file = "processing_queue.json"
        self.cli_options = {
            "model_choice": "",
            "encoding_preset": "hevc-nvidia-gpu-balanced",
            "fp16": True,  # Changed to boolean
            "crf_value": "19",
            "ffmpeg_option": "no_trim",
            "max_clip_size": "60",  # NEW: Default max clip size
            "temporal_overlap": "5",  # NEW: Default temporal overlap
            "detection_score_threshold": "0.2",  # NEW: Default detection score threshold
            "codec": "hevc",  # NEW: Default codec (currently only hevc supported)
            "encoder_settings": ""  # NEW: Default encoder settings
        }
        self.processing_queue = self.load_queue()
        self.is_batch_processing = False
        self.is_running = False
        
        self.frame_queue = Queue(maxsize=3)
        self.frame_buffer_thread = None
        self.buffer_running = False
        self.cap_lock = threading.Lock()
        self.last_frame_time = time.time()
        
        # Initialize tkinter variables
        self.model_var = tk.StringVar()
        self.fp16_var = tk.BooleanVar(value=True)
        self.max_clip_size_var = tk.StringVar(value=self.cli_options["max_clip_size"])
        self.temporal_overlap_var = tk.StringVar(value=self.cli_options["temporal_overlap"])
        self.detection_score_threshold_var = tk.StringVar(value=self.cli_options["detection_score_threshold"])  # NEW
        self.codec_var = tk.StringVar(value=self.cli_options["codec"])  # NEW
        self.encoder_settings_var = tk.StringVar(value=self.cli_options["encoder_settings"])  # NEW
        self.crf_var = tk.StringVar(value=self.cli_options["crf_value"])
        self.ffmpeg_option_var = tk.StringVar(value=self.cli_options["ffmpeg_option"])
        self.output_folder_var = tk.StringVar(value=self.output_dir)
        self.vr_processing_var = tk.BooleanVar(value=False)
        self.vr_simple_mode_var = tk.BooleanVar(value=True)
        self.save_trimmed_video_var = tk.BooleanVar(value=False)
        self.show_completion_dialog_var = tk.BooleanVar(value=True)
        self.suppress_queue_message_var = tk.BooleanVar(value=True)
        
        # Check if jasna.exe exists
        if not os.path.exists(self.jasna_path):
            messagebox.showerror("Error", "jasna.exe not found.")
            self.jasna_path = None
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.model_weights_dir):
            os.makedirs(self.model_weights_dir)
        
        # Dynamically search for detection models
        self.available_detection_models = self.scan_detection_models()
        
        self.create_widgets()
        self.load_config()
        
        # Bind keys after widgets are created
        self.bind_keys(self.root)
        self.root.bind('<Configure>', self.on_window_resize)
        
        self.fullscreen_window = None
        self.fullscreen_progress_canvas = None
        self.fullscreen_progress_bar = None
        self.fullscreen_start_marker = None
        self.fullscreen_end_marker = None
        self.fullscreen_progress_text = None

    def scan_detection_models(self):
        """Search for .onnx files in model_weights folder"""
        # Search for onnx files used by JASNA
        pattern = os.path.join(self.model_weights_dir, "*.onnx")
        model_files = glob.glob(pattern)
        models = []
        
        for file_path in model_files:
            filename = os.path.basename(file_path)
            
            # Remove extension and use filename as label
            label = os.path.splitext(filename)[0]
            models.append((label, file_path))
                
        # Sort by display name alphabetically
        models.sort(key=lambda x: x[0])
        return models

    def bind_keys(self, window):
        window.bind('<Right>', self.move_frame)
        window.bind('<Left>', self.move_frame)
        window.bind('<Shift-Right>', self.move_frame)
        window.bind('<Shift-Left>', self.move_frame)
        window.bind('<space>', self.toggle_play_pause)
        window.bind('<Up>', self.jump_to_start)
        window.bind('<Down>', self.jump_to_end)
        window.bind('<Control-Up>', self.set_start_point_by_key)
        window.bind('<Control-Down>', self.set_end_point_by_key)
        window.bind('<Control-e>', self.add_to_queue)
        window.bind('<Control-q>', lambda e: self.open_queue_window())
        window.bind('<Control-r>', lambda e: self.reset_points())
        window.bind('f', self.toggle_fullscreen)
        window.bind('j', self.move_one_frame_backward)
        window.bind('k', self.toggle_play_pause)
        window.bind('l', self.move_one_frame_forward)
        window.bind('h', self.move_one_second_backward)
        window.bind(';', self.move_one_second_forward)
        window.bind('<Home>', lambda e: self.set_frame_and_start(0))
        window.bind('<End>', lambda e: self.set_frame_and_end(self.video_total_frames))
        window.bind('s', self.jump_to_video_start)
        window.bind('e', self.jump_to_video_end)
        for i in range(1, 10):
            window.bind(str(i), lambda e, percentage=i*10: self.jump_to_percentage(percentage))

    def jump_to_video_start(self, event=None):
        if not self.cap or not self.cap.isOpened():
            return
        self.current_frame = 0
        self.clear_frame_queue()
        with self.cap_lock:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    if self.fullscreen_window:
                        self.display_frame_fullscreen(frame)
                self.on_progress_update()
                self.update_time_labels()
            except Exception as e:
                self.write_log(f"Jump to video start error: {e}")

    def jump_to_video_end(self, event=None):
        if not self.cap or not self.cap.isOpened():
            return
        self.current_frame = max(0, self.video_total_frames - 1)
        self.clear_frame_queue()
        with self.cap_lock:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    if self.fullscreen_window:
                        self.display_frame_fullscreen(frame)
                self.on_progress_update()
                self.update_time_labels()
            except Exception as e:
                self.write_log(f"Jump to video end error: {e}")

    def exit_fullscreen(self, event=None):
        if self.fullscreen_window:
            self.buffer_running = False
            try:
                self.fullscreen_window.destroy()
            except:
                pass
            self.fullscreen_window = None
            self.fullscreen_progress_canvas = None
            self.fullscreen_progress_bar = None
            self.fullscreen_start_marker = None
            self.fullscreen_end_marker = None
            self.fullscreen_progress_text = None
            self.clear_frame_queue()

    def set_frame_and_start(self, frame):
        if self.cap and self.cap.isOpened():
            self.current_frame = frame
            self.clear_frame_queue()
            with self.cap_lock:
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ret, frame_data = self.cap.read()
                    if ret:
                        self.display_frame(frame_data)
                        if self.fullscreen_window:
                            self.display_frame_fullscreen(frame_data)
                    self.on_progress_update()
                    self.update_time_labels()
                    self.set_start_point_by_key()
                except Exception as e:
                    self.write_log(f"Start frame setting error: {e}")

    def set_frame_and_end(self, frame):
        if self.cap and self.cap.isOpened():
            self.current_frame = frame
            self.clear_frame_queue()
            with self.cap_lock:
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                    ret, frame_data = self.cap.read()
                    if ret:
                        self.display_frame(frame_data)
                        if self.fullscreen_window:
                            self.display_frame_fullscreen(frame_data)
                    self.on_progress_update()
                    self.update_time_labels()
                    self.set_end_point_by_key()
                except Exception as e:
                    self.write_log(f"End frame setting error: {e}")

    def load_queue(self):
        if os.path.exists(self.queue_file):
            try:
                with open(self.queue_file, 'r', encoding='utf-8') as f:
                    queue = json.load(f)
                    self.write_log(f"Queue loaded: {len(queue)} items")
                    return queue
            except Exception as e:
                self.write_log(f"Queue loading error: {e}")
                messagebox.showwarning("Warning", f"Failed to load queue: {e}. Continuing with empty queue.")
                return []
        self.write_log("Queue does not exist. Creating new one.")
        return []

    def save_queue(self):
        try:
            with open(self.queue_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_queue, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.write_log(f"Queue saving error: {e}")
            messagebox.showwarning("Warning", f"Failed to save queue: {e}. Please check manually.")

    def generate_unique_filepath(self, base_path):
        if not os.path.exists(base_path):
            return base_path
        base, ext = os.path.splitext(base_path)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        main_frame.grid_rowconfigure(2, weight=4)
        main_frame.grid_rowconfigure(5, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        file_frame = tk.LabelFrame(main_frame, text="1. Video File Selection", padx=10, pady=10)
        file_frame.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.file_path_entry = tk.Entry(file_frame)
        self.file_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.browse_button = tk.Button(file_frame, text="Browse...", command=self.browse_file)
        self.browse_button.pack(side=tk.LEFT, padx=5)

        # 重新排列的选项设置部分 - 所有选项在一行
        options_frame = tk.LabelFrame(main_frame, text="2. Option Settings", padx=10, pady=10)
        options_frame.grid(row=1, column=0, sticky="ew", pady=5)
        
        # 创建第一行：所有选项水平排列
        row1_frame = tk.Frame(options_frame)
        row1_frame.pack(fill=tk.X, expand=True, pady=(0, 5))
        
        # Detection Model
        tk.Label(row1_frame, text="Detection Model:").pack(side=tk.LEFT, padx=(0, 5))
        
        # 创建模型选择菜单
        model_labels = [label for label, _ in self.available_detection_models]
        if not model_labels:
            model_labels = ["No model found"]
            self.model_menu = tk.OptionMenu(row1_frame, self.model_var, *model_labels)
            self.model_menu.config(state="disabled")
        else:
            self.model_menu = tk.OptionMenu(row1_frame, self.model_var, *model_labels)
        self.model_menu.pack(side=tk.LEFT, padx=(0, 15))
        
        # FP16 checkbox
        self.fp16_check = Checkbutton(row1_frame, text="FP16", variable=self.fp16_var)
        self.fp16_check.pack(side=tk.LEFT, padx=(0, 15))
        
        # Max Clip Size
        tk.Label(row1_frame, text="Max Clip Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.max_clip_size_spinbox = tk.Spinbox(row1_frame, from_=1, to=300, width=6, 
                                                textvariable=self.max_clip_size_var)
        self.max_clip_size_spinbox.pack(side=tk.LEFT, padx=(0, 15))
        
        # Temporal Overlap
        tk.Label(row1_frame, text="Temporal Overlap:").pack(side=tk.LEFT, padx=(0, 5))
        self.temporal_overlap_spinbox = tk.Spinbox(row1_frame, from_=0, to=20, width=6, 
                                                   textvariable=self.temporal_overlap_var)
        self.temporal_overlap_spinbox.pack(side=tk.LEFT, padx=(0, 15))
        
        # Output Folder
        tk.Label(row1_frame, text="Output Folder:").pack(side=tk.LEFT, padx=(0, 5))
        self.output_folder_entry = tk.Entry(row1_frame, textvariable=self.output_folder_var, width=30)
        self.output_folder_entry.pack(side=tk.LEFT, padx=(0, 5))
        self.output_folder_button = tk.Button(row1_frame, text="Change...", command=self.change_output_folder)
        self.output_folder_button.pack(side=tk.LEFT)

        # NEW: Second row for new options
        row2_frame = tk.Frame(options_frame)
        row2_frame.pack(fill=tk.X, expand=True, pady=(5, 0))
        
        # Detection Score Threshold
        tk.Label(row2_frame, text="Detection Score Threshold:").pack(side=tk.LEFT, padx=(0, 5))
        self.detection_score_threshold_spinbox = tk.Spinbox(
            row2_frame, 
            from_=0.0, 
            to=1.0, 
            increment=0.1,
            width=6, 
            textvariable=self.detection_score_threshold_var,
            format="%.1f"
        )
        self.detection_score_threshold_spinbox.pack(side=tk.LEFT, padx=(0, 15))
        
        # Codec selection (currently only hevc supported)
        tk.Label(row2_frame, text="Codec:").pack(side=tk.LEFT, padx=(0, 5))
        self.codec_menu = tk.OptionMenu(row2_frame, self.codec_var, "hevc")
        self.codec_menu.pack(side=tk.LEFT, padx=(0, 15))
        
        # Encoder Settings
        tk.Label(row2_frame, text="Encoder Settings:").pack(side=tk.LEFT, padx=(0, 5))
        self.encoder_settings_entry = tk.Entry(row2_frame, textvariable=self.encoder_settings_var, width=30)
        self.encoder_settings_entry.pack(side=tk.LEFT, padx=(0, 5))
        tk.Label(row2_frame, text="(e.g., cq=22,lookahead=32)").pack(side=tk.LEFT)

        preview_frame = tk.LabelFrame(main_frame, text="3. Processing Range Specification", padx=10, pady=10)
        preview_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(preview_frame, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        self.video_label.bind("<Button-1>", self.toggle_play_pause)
        self.video_label.bind("<Double-Button-1>", self.toggle_fullscreen)
        self.video_label.bind("<MouseWheel>", self.on_mouse_wheel)

        self.progress_canvas = tk.Canvas(preview_frame, height=20, bg="grey")
        self.progress_canvas.grid(row=1, column=0, sticky="ew", pady=2)
        self.progress_canvas.bind("<Button-1>", self.on_progress_click)
        self.progress_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.progress_bar = self.progress_canvas.create_rectangle(0, 0, 0, 20, fill="green")
        self.start_marker = self.progress_canvas.create_line(0, 0, 0, 20, fill="red", width=2)
        self.end_marker = self.progress_canvas.create_line(0, 0, 0, 20, fill="blue", width=2)
        
        control_range_frame = tk.Frame(preview_frame)
        control_range_frame.grid(row=2, column=0, pady=2)

        self.play_pause_button = tk.Button(control_range_frame, text="▶ Play", command=self.toggle_play_pause)
        self.play_pause_button.pack(side=tk.LEFT)

        self.current_time_label = tk.Label(control_range_frame, text="00:00:00 / 00:00:00")
        self.current_time_label.pack(side=tk.LEFT, padx=10)

        self.set_start_button = tk.Button(control_range_frame, text="Set Start Point", command=self.set_start_point)
        self.set_start_button.pack(side=tk.LEFT, padx=5)
        
        self.set_end_button = tk.Button(control_range_frame, text="Set End Point", command=self.set_end_point)
        self.set_end_button.pack(side=tk.LEFT, padx=5)

        time_display_frame = tk.Frame(preview_frame)
        time_display_frame.grid(row=3, column=0, pady=2, padx=100)
        tk.Label(time_display_frame, text="Start Time:").pack(side=tk.LEFT)
        self.start_time_label = tk.Label(time_display_frame, text="00:00:00", width=12, relief="sunken")
        self.start_time_label.pack(side=tk.LEFT)
        tk.Label(time_display_frame, text="End Time:").pack(side=tk.LEFT, padx=(10, 0))
        self.end_time_label = tk.Label(time_display_frame, text="00:00:00", width=12, relief="sunken")
        self.end_time_label.pack(side=tk.LEFT)
        
        self.reset_button = tk.Button(time_display_frame, text="Reset Range", command=self.reset_points)
        self.reset_button.pack(side=tk.LEFT, padx=10)

        ffmpeg_frame = tk.LabelFrame(main_frame, text="4. Video Extraction Settings", padx=10, pady=10)
        ffmpeg_frame.grid(row=3, column=0, sticky="ew", pady=5)
        
        tk.Radiobutton(ffmpeg_frame, text="-c copy (Fast)", variable=self.ffmpeg_option_var, value="copy").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(ffmpeg_frame, text="-c copy +genpts (Timestamp fix)", variable=self.ffmpeg_option_var, value="copy_genpts").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(ffmpeg_frame, text="Re-encode (NVENC)", variable=self.ffmpeg_option_var, value="re_encode").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(ffmpeg_frame, text="As-is (No trimming)", variable=self.ffmpeg_option_var, value="no_trim").pack(side=tk.LEFT, padx=5)
        
        crf_label = tk.Label(ffmpeg_frame, text="Video Quality(5-30):")
        crf_label.pack(side=tk.LEFT, padx=(15, 5))
        crf_values = [str(i) for i in range(5, 31)]
        self.crf_menu = tk.OptionMenu(ffmpeg_frame, self.crf_var, *crf_values)
        self.crf_menu.pack(side=tk.LEFT, padx=5)
        
        self.batch_count_label = tk.Label(ffmpeg_frame, text="", fg="blue")
        self.batch_count_label.pack(side=tk.RIGHT, padx=5)

        # VR processing checkbox
        vr_frame = tk.LabelFrame(main_frame, text="5. VR Video Processing", padx=10, pady=10)
        vr_frame.grid(row=4, column=0, sticky="ew", pady=5)
        
        self.vr_processing_check = Checkbutton(vr_frame, text="VR Processing(180-degree SBS format)", 
                                               variable=self.vr_processing_var, command=self.on_vr_mode_toggle, state=tk.DISABLED)
        self.vr_processing_check.pack(side=tk.LEFT, padx=5)
        
        self.vr_simple_mode_check = Checkbutton(vr_frame, text="Simple Processing Mode (Center 70% only)", 
                                                variable=self.vr_simple_mode_var, state=tk.DISABLED)
        self.vr_simple_mode_check.pack(side=tk.LEFT, padx=5)
        
        self.log_button = tk.Button(
            vr_frame, 
            text="Log", 
            width=8,
            command=self.open_log_file
        )
        self.log_button.pack(side=tk.RIGHT, padx=5)

        control_frame = tk.Frame(main_frame, pady=10)
        control_frame.grid(row=5, column=0, sticky="ew")
        
        self.save_trimmed_video_check = Checkbutton(control_frame, text="Save trimmed video", variable=self.save_trimmed_video_var)
        self.save_trimmed_video_check.pack(side=tk.LEFT, padx=5)
        
        self.show_completion_dialog_check = Checkbutton(control_frame, text="Show completion dialog", variable=self.show_completion_dialog_var)
        self.show_completion_dialog_check.pack(side=tk.LEFT, padx=5)
        
        self.start_button = tk.Button(control_frame, text="Start Processing (Single)", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.queue_view_button = tk.Button(control_frame, text="View Queue", command=self.open_queue_window)
        self.queue_view_button.pack(side=tk.LEFT, padx=5)
        
        self.batch_button = tk.Button(control_frame, text="Batch Start", command=lambda: self.start_batch_processing(control_frame))
        self.batch_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(control_frame, text="Ready", fg="blue")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.abort_button = tk.Button(control_frame, text="Abort", command=self.abort_processing, bg="orange", fg="white")
        self.abort_button.pack(side=tk.RIGHT, padx=5)

        time_display_frame2 = tk.Frame(preview_frame)
        time_display_frame2.grid(row=5, column=0, pady=2, padx=100)
        self.queue_add_button = tk.Button(time_display_frame2, text="Add to Queue", command=self.add_to_queue, bg="#87CEEB")
        self.queue_add_button.pack(side=tk.LEFT, padx=5)

        self.suppress_queue_message_check = Checkbutton(time_display_frame2, text="No message when adding to queue", 
                                                         variable=self.suppress_queue_message_var)
        self.suppress_queue_message_check.pack(side=tk.RIGHT, padx=5)

        jasna_info_frame = tk.LabelFrame(main_frame, text="JASNA Processing Information", padx=10, pady=10)
        jasna_info_frame.grid(row=6, column=0, sticky="nsew", pady=5)

        self.console_text = scrolledtext.ScrolledText(jasna_info_frame, height=5, state=tk.DISABLED)
        self.console_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Set trace callbacks after widgets are created
        self.setup_trace_callbacks()
        
        # Start preview update after widgets are created
        self.root.after(100, self.update_preview)
        
    def setup_trace_callbacks(self):
        """Setup trace callbacks for all variables that need to save config"""
        self.model_var.trace_add("write", self.save_config_callback)
        self.fp16_var.trace_add("write", self.save_config_callback)
        self.max_clip_size_var.trace_add("write", self.save_config_callback)
        self.temporal_overlap_var.trace_add("write", self.save_config_callback)
        self.detection_score_threshold_var.trace_add("write", self.save_config_callback)  # NEW
        self.codec_var.trace_add("write", self.save_config_callback)  # NEW
        self.encoder_settings_var.trace_add("write", self.save_config_callback)  # NEW
        self.crf_var.trace_add("write", self.save_config_callback)
        self.ffmpeg_option_var.trace_add("write", self.save_config_callback)
        self.output_folder_var.trace_add("write", self.save_config_callback)
    
    def change_output_folder(self):
        new_folder = filedialog.askdirectory(initialdir=self.output_dir)
        if new_folder:
            self.output_dir = new_folder
            self.output_folder_var.set(new_folder)
            self.save_config()
            self.write_log(f"Output folder changed to: {new_folder}")
    
    def on_vr_mode_toggle(self):
        if self.vr_processing_var.get():
            self.vr_simple_mode_var.set(True)
            self.vr_simple_mode_check.config(state=tk.DISABLED)
            self.write_log("VR mode enabled: Operating in simple processing mode (center 70%)")
        else:
            self.vr_simple_mode_check.config(state=tk.NORMAL)

    def get_selected_detection_model_path(self):
        selected_label = self.model_var.get()
        for label, path in self.available_detection_models:
            if label == selected_label:
                return path
        return None

    def apply_vr_undistortion(self, input_file, output_file, unique_id):
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"Extracting VR video center area (70% area)...\n")
        self.console_text.config(state=tk.DISABLED)
        self.write_log("VR center area extraction started (70% area)")
        
        crop_center_cmd = [
            'ffmpeg', '-y', '-i', input_file,
            '-vf', 'crop=iw*0.837:ih*0.837:iw*0.0815:ih*0.0815',
            '-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq', '18',
            '-an',
            output_file
        ]
        subprocess.run(crop_center_cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        self.write_log("VR center area extraction completed")

    def apply_vr_distortion(self, input_file, output_file, unique_id):
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"Compositing processed area back to original video...\n")
        self.console_text.config(state=tk.DISABLED)
        self.write_log("Compositing to original video started")
        
        trimmed_file = None
        for file in os.listdir(self.output_dir):
            if file.startswith(f'trimmed_{unique_id}') and file.endswith(('.mp4', '.avi', '.mkv', '.mov')):
                trimmed_file = os.path.join(self.output_dir, file)
                break
        
        if not trimmed_file or not os.path.exists(trimmed_file):
            self.write_log("Error: Original trimmed video not found")
            raise Exception("Original trimmed video not found")
        
        overlay_cmd = [
            'ffmpeg', '-y',
            '-i', trimmed_file,
            '-i', input_file,
            '-filter_complex', '[0:v][1:v]overlay=(W-w)/2:(H-h)/2[v]',
            '-map', '[v]',
            '-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq', '18',
            output_file
        ]
        subprocess.run(overlay_cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        self.write_log("Compositing to original video completed")

    def split_vr_video(self, input_file, unique_id):
        audio_file = os.path.join(self.output_dir, f'{unique_id}_audio.aac')
        extract_audio_command = [
            'ffmpeg', '-y', '-i', input_file,
            '-vn', '-acodec', 'copy', audio_file
        ]
        
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"Extracting audio...\n")
        self.console_text.config(state=tk.DISABLED)
        self.write_log("Audio extraction from VR video started")
        
        try:
            subprocess.run(extract_audio_command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.write_log("Audio extraction completed")
        except subprocess.CalledProcessError:
            self.write_log("Audio extraction failed or no audio track")
            audio_file = None
        
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"VR simple mode: Extracting center area only\n")
        self.console_text.config(state=tk.DISABLED)
        self.write_log("VR simple mode: Center area extraction started")
        
        center_file = os.path.join(self.output_dir, f'{unique_id}_center.mp4')
        self.apply_vr_undistortion(input_file, center_file, unique_id)
        
        parts = ['center']
        
        return parts, audio_file

    def run_jasna(self, input_file, unique_id):
        if not self.jasna_path or not os.path.exists(self.jasna_path):
            raise FileNotFoundError(f"jasna.exe not found: {self.jasna_path}")
        
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_file_temp = os.path.join(self.output_dir, f"{input_basename}_jasna_temp_{unique_id}.mp4")
        
        detect_model_path = self.get_selected_detection_model_path()
        if not detect_model_path or not os.path.exists(detect_model_path):
            raise FileNotFoundError(f"Selected detection model not found: {detect_model_path}")
        
        # Build JASNA command
        jasna_command = [
            self.jasna_path,
            "--input", input_file,
            "--output", output_file_temp,
            "--detection-model", "rfdetr",
            "--detection-model-path", detect_model_path,
            "--device", "cuda:0",
            "--max-clip-size", self.max_clip_size_var.get(),  # Use user setting
            "--temporal-overlap", self.temporal_overlap_var.get(),  # Use user setting
            "--detection-score-threshold", self.detection_score_threshold_var.get(),  # NEW: Add detection score threshold
            "--codec", self.codec_var.get()  # NEW: Add codec
        ]
        
        # Add encoder settings if provided
        encoder_settings = self.encoder_settings_var.get().strip()
        if encoder_settings:
            jasna_command.extend(["--encoder-settings", encoder_settings])
        
        # Add FP16 option based on user setting
        if self.fp16_var.get():  # Now using BooleanVar
            jasna_command.append("--fp16")
            self.write_log("FP16 acceleration enabled")
        else:
            self.write_log("FP16 acceleration disabled")
        
        # Check if restoration model exists
        restore_pattern = os.path.join(self.model_weights_dir, "*.pth")
        restore_files = glob.glob(restore_pattern)
        if restore_files:
            restore_model_path = restore_files[0]
            jasna_command.extend(["--restoration-model-name", "basicvsrpp"])
            jasna_command.extend(["--restoration-model-path", restore_model_path])
        else:
            self.write_log("Warning: No restoration model (.pth) found, running detection only")
        
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, f"Starting JASNA processing...\nExecution command: {' '.join(jasna_command)}\n")
        self.console_text.config(state=tk.DISABLED)
        self.write_log(f"JASNA processing started: {' '.join(jasna_command)}")
        self.root.update()
        
        # Fix: Assign to self.process for class-wide access
        self.process = subprocess.Popen(
            jasna_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW,
            encoding='cp932',
            errors='replace'
        )
        
        try:
            # Waiting logic (same as before)
            while True:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None:
                    break
                if line:
                    line_stripped = line.strip()
                    is_progress_message = line_stripped.startswith("Processing frames:") or \
                                        line_stripped.startswith("Processing video:")
                    
                    self.console_text.config(state=tk.NORMAL)
                    self.console_text.insert(tk.END, line)
                    self.console_text.see(tk.END)
                    self.console_text.config(state=tk.DISABLED)

                    if not is_progress_message:
                        self.write_log(line_stripped)
                    
                    self.root.update()
            
            return_code = self.process.wait()
        finally:
            # Clear reference after completion or interruption
            self.process = None
        
        if return_code != 0:
            # Check if interrupted by looking at is_running flag
            if not self.is_running:
                self.write_log("JASNA processing interrupted by user.")
            else:
                raise Exception(f"jasna.exe execution failed (exit code: {return_code})")
        
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, "JASNA processing completed.\n")
        self.console_text.config(state=tk.DISABLED)
        self.write_log("JASNA processing completed")
        
        return output_file_temp

    def abort_processing(self):
        if not (hasattr(self, 'is_running') and self.is_running) and \
        not (hasattr(self, 'is_batch_processing') and self.is_batch_processing):
            messagebox.showinfo("Information", "No processing is currently running.")
            return
        
        if not messagebox.askyesno("Confirmation", "Interrupt current processing?"):
            return
        
        # Set abort flag
        self.is_batch_processing = False
        self.is_running = False
        self.buffer_running = False
        
        # Only terminate if self.process exists and is still running
        if hasattr(self, 'process') and self.process and self.process.poll() is None:
            try:
                pid = self.process.pid
                self.write_log(f"Terminating self-started process (PID: {pid}) and its child processes...")
                
                if os.name == 'nt':
                    # Windows: /T forces termination of entire tree including child processes (ffmpeg, etc.)
                    # Specifies the PID we forked, so it won't interfere with other running jasna instances
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                capture_output=True, check=False, 
                                creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    self.process.kill()
                
                # Wait a bit for termination
                self.process.wait(timeout=3)
                self.write_log(f"Process tree with PID {pid} terminated.")
            except Exception as e:
                self.write_log(f"Error during forced termination: {e}")
            finally:
                self.process = None
        else:
            self.write_log("No running self-started process found.")
        
        # Restore UI state
        self.start_button.config(state=tk.NORMAL, text="Start Processing (Single)")
        self.batch_button.config(state=tk.NORMAL)
        self.queue_add_button.config(state=tk.NORMAL)
        self.queue_view_button.config(state=tk.NORMAL)
        self.root.bind('<Control-e>', self.add_to_queue)
        
        self.status_label.config(text="Processing interrupted", fg="red")
        self.batch_count_label.config(text="")
        self.console_text.config(state=tk.NORMAL)
        self.console_text.insert(tk.END, "Processing interrupted.\n")
        self.console_text.config(state=tk.DISABLED)
        
        self.write_log("Interruption completed.")
        messagebox.showinfo("Interruption Complete", "Only the processing we started has been safely interrupted.")
    
    def add_to_queue(self, event=None):
        if self.is_batch_processing:
            messagebox.showwarning("Warning", "Cannot add to queue during batch processing.")
            return
        input_file = self.file_path_entry.get()
        if not input_file or not os.path.exists(input_file):
            messagebox.showerror("Error", "Please select a valid video file.")
            return
        
        with self.cap_lock:
            cap_temp = cv2.VideoCapture(input_file)
            if not cap_temp.isOpened():
                messagebox.showerror("Error", "Cannot open video file.")
                cap_temp.release()
                return
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
            cap_temp.release()
        
        if self.ffmpeg_option_var.get() == "no_trim" and \
           ((self.start_frame > 0) or (self.end_frame < total_frames)):
            messagebox.showerror("Error", "Range specification is not allowed when 'As-is' is selected.")
            return

        queue_entry = {
            'video_path': input_file,
            'model': self.model_var.get(),
            'fp16': self.fp16_var.get(),  # Now Boolean value
            'max_clip_size': int(self.max_clip_size_var.get()),  # NEW: Add max clip size
            'temporal_overlap': int(self.temporal_overlap_var.get()),  # NEW: Add temporal overlap
            'detection_score_threshold': float(self.detection_score_threshold_var.get()),  # NEW: Add detection score threshold
            'codec': self.codec_var.get(),  # NEW: Add codec
            'encoder_settings': self.encoder_settings_var.get(),  # NEW: Add encoder settings
            'start_frame': self.start_frame,
            'end_frame': min(self.end_frame, total_frames),
            'ffmpeg_option': self.ffmpeg_option_var.get(),
            'save_trimmed': self.save_trimmed_video_var.get(),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'fps': fps,
            'crf_value': int(self.crf_var.get()),
            'vr_processing': self.vr_processing_var.get(),
            'vr_simple_mode': self.vr_simple_mode_var.get()
        }
        
        self.processing_queue.append(queue_entry)
        self.save_queue()
        self.write_log(f"Added to queue: {os.path.basename(input_file)}")
        if not self.suppress_queue_message_var.get():
            messagebox.showinfo("Addition Complete", f"Added {os.path.basename(input_file)} to queue.\nTotal queue items: {len(self.processing_queue)}")

    def open_queue_window(self):
        if not self.processing_queue:
            self.write_log("Queue check: Queue is empty")
            messagebox.showinfo("Information", "Queue is empty.")
            return
        
        queue_window = tk.Toplevel(self.root)
        queue_window.title("Processing Queue Check")
        queue_window.geometry("900x400")
        
        list_frame = tk.Frame(queue_window)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.queue_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=("MS Gothic", 10), width=100)
        self.queue_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.queue_listbox.yview)
        
        self.update_queue_listbox()
        
        queue_status_label = tk.Label(queue_window, text=f"Current queue count: {len(self.processing_queue)}", fg="blue")
        queue_status_label.pack(pady=(5, 10))
        
        btn_frame = tk.Frame(queue_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        up_btn = tk.Button(btn_frame, text="↑ Up", command=lambda: self.move_queue_item(-1))
        up_btn.pack(side=tk.LEFT, padx=5)
        
        down_btn = tk.Button(btn_frame, text="↓ Down", command=lambda: self.move_queue_item(1))
        down_btn.pack(side=tk.LEFT, padx=5)
        
        delete_btn = tk.Button(
            btn_frame, 
            text="Delete", 
            command=lambda: self.delete_queue_item(
                queue_window=queue_window, 
                queue_status_label=queue_status_label
            )
        )
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        clear_all_btn = tk.Button(btn_frame, text="Clear All", command=lambda: self.clear_all_queue(queue_window, queue_status_label))
        clear_all_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = tk.Button(btn_frame, text="Close", command=queue_window.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)
        
        queue_window.lift()

    def clear_all_queue(self, queue_window, queue_status_label):
        if not self.processing_queue:
            queue_status_label.config(text="Queue is already empty.", fg="blue")
            queue_window.lift()
            return
        
        self.processing_queue.clear()
        self.save_queue()
        self.update_queue_listbox()
        queue_status_label.config(text="All queue items cleared. Current queue count: 0", fg="blue")
        queue_window.lift()
        self.write_log("All queue items cleared.")

    def update_queue_listbox(self):
        self.queue_listbox.delete(0, tk.END)
        ffmpeg_display_map = {
            'copy': 'Fast',
            'copy_genpts': 'Timestamp fix',
            're_encode': 'Re-encode (NVENC)',
            'no_trim': 'As-is (No trimming)'
        }
        for i, entry in enumerate(self.processing_queue):
            try:
                filename = os.path.basename(entry['video_path'])
                model_label = entry['model']
                fps = entry.get('fps', 30.0)
                start_time = self.format_time(entry['start_frame'] / fps if fps > 0 else 0)
                end_time = self.format_time(entry['end_frame'] / fps if fps > 0 else 0)
                ffmpeg_option = entry.get('ffmpeg_option', 're_encode')
                ffmpeg_display = ffmpeg_display_map.get(ffmpeg_option, ffmpeg_option)
                save_trimmed = 'Save' if entry['save_trimmed'] else 'Don\'t save'
                crf_value = entry.get('crf_value', 19)
                fp16 = 'FP16' if entry.get('fp16', True) else 'No FP16'
                max_clip_size = entry.get('max_clip_size', 30)  # NEW: Get max clip size
                temporal_overlap = entry.get('temporal_overlap', 3)  # NEW: Get temporal overlap
                detection_score_threshold = entry.get('detection_score_threshold', 0.2)  # NEW: Get detection score threshold
                codec = entry.get('codec', 'hevc')  # NEW: Get codec
                encoder_settings = entry.get('encoder_settings', '')  # NEW: Get encoder settings
                vr_mode = 'VR' if entry.get('vr_processing', False) else '2D'
                simple_mode = 'Simple' if entry.get('vr_simple_mode', False) else 'Normal'
                
                if ffmpeg_option == 'no_trim':
                    range_display = "Full range"
                else:
                    range_display = f"Range:{start_time}-{end_time}"
                
                # Build display text with all parameters
                display_text = (f"{i+1}. {filename}, Model:{model_label}, "
                               f"{range_display}, FFmpeg:{ffmpeg_display}, CRF:{crf_value}, "
                               f"FP16:{fp16}, Clip:{max_clip_size}, Overlap:{temporal_overlap}, "
                               f"DetectThresh:{detection_score_threshold}, Codec:{codec}, "
                               f"Encoder:{encoder_settings[:20] if encoder_settings else 'default'}, "
                               f"SaveTrim:{save_trimmed}, Mode:{vr_mode}, VRMode:{simple_mode}")
                self.queue_listbox.insert(tk.END, display_text)
            except Exception as e:
                self.queue_listbox.insert(tk.END, f"{i+1}. Display error: {e}")

    def move_queue_item(self, direction):
        sel = self.queue_listbox.curselection()
        if not sel:
            messagebox.showwarning("Warning", "Please select an item.")
            return
        idx = sel[0]
        new_idx = idx + direction
        if 0 <= new_idx < len(self.processing_queue):
            self.processing_queue[idx], self.processing_queue[new_idx] = self.processing_queue[new_idx], self.processing_queue[idx]
            self.save_queue()
            self.update_queue_listbox()
            self.queue_listbox.selection_set(new_idx)

    def delete_queue_item(self, queue_window, queue_status_label):
        sel = self.queue_listbox.curselection()
        if not sel:
            queue_status_label.config(text="Please select an item to delete.", fg="red")
            queue_window.lift()
            return

        deleted_index = sel[0]
        items_deleted = 0

        if 0 <= deleted_index < len(self.processing_queue):
            del self.processing_queue[deleted_index]
            self.save_queue()
            self.update_queue_listbox()
            items_deleted = 1

        queue_status_label.config(
            text=f"{items_deleted} item(s) deleted. Current queue count: {len(self.processing_queue)}",
            fg="blue" if items_deleted > 0 else "red"
        )
        queue_window.lift()
        self.write_log(f"{items_deleted} item(s) removed from queue.")

    def save_config_callback(self, *args):
        self.save_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line.startswith("model="):
                            model_label = line.split("=")[1]
                            model_labels = [lbl for lbl, _ in self.available_detection_models]
                            if model_label in model_labels:
                                self.model_var.set(model_label)
                                self.cli_options["model_choice"] = model_label
                        elif line.startswith("crf="):
                            crf = line.split("=")[1]
                            if crf.isdigit() and 5 <= int(crf) <= 30:
                                self.cli_options["crf_value"] = crf
                                self.crf_var.set(crf)
                        elif line.startswith("output_dir="):
                            output_dir = line.split("=")[1]
                            if os.path.exists(output_dir):
                                self.output_dir = output_dir
                                self.output_folder_var.set(output_dir)
                        elif line.startswith("ffmpeg_option="):
                            opt = line.split("=")[1]
                            valid_options = ["copy", "copy_genpts", "re_encode", "no_trim"]
                            if opt in valid_options:
                                self.cli_options["ffmpeg_option"] = opt
                                self.ffmpeg_option_var.set(opt)
                        elif line.startswith("fp16="):  # Load FP16 setting
                            fp16_val = line.split("=")[1]
                            if fp16_val.lower() == "true":
                                self.cli_options["fp16"] = True
                                self.fp16_var.set(True)
                            elif fp16_val.lower() == "false":
                                self.cli_options["fp16"] = False
                                self.fp16_var.set(False)
                        elif line.startswith("max_clip_size="):  # Load max clip size
                            max_clip_size = line.split("=")[1]
                            if max_clip_size.isdigit() and 1 <= int(max_clip_size) <= 300:
                                self.cli_options["max_clip_size"] = max_clip_size
                                self.max_clip_size_var.set(max_clip_size)
                        elif line.startswith("temporal_overlap="):  # Load temporal overlap
                            temporal_overlap = line.split("=")[1]
                            if temporal_overlap.isdigit() and 0 <= int(temporal_overlap) <= 20:
                                self.cli_options["temporal_overlap"] = temporal_overlap
                                self.temporal_overlap_var.set(temporal_overlap)
                        elif line.startswith("detection_score_threshold="):  # NEW: Load detection score threshold
                            threshold = line.split("=")[1]
                            try:
                                threshold_val = float(threshold)
                                if 0.0 <= threshold_val <= 1.0:
                                    self.cli_options["detection_score_threshold"] = threshold
                                    self.detection_score_threshold_var.set(threshold)
                            except ValueError:
                                pass
                        elif line.startswith("codec="):  # NEW: Load codec
                            codec_val = line.split("=")[1]
                            if codec_val.lower() == "hevc":
                                self.cli_options["codec"] = codec_val
                                self.codec_var.set(codec_val)
                        elif line.startswith("encoder_settings="):  # NEW: Load encoder settings
                            encoder_settings_val = line.split("=")[1]
                            self.cli_options["encoder_settings"] = encoder_settings_val
                            self.encoder_settings_var.set(encoder_settings_val)
            except Exception as e:
                self.write_log(f"Failed to load config file: {e}")
                messagebox.showwarning("Warning", f"Failed to load config file: {e}. Continuing with default values.")

        # Select first model if model is empty
        if self.available_detection_models and not self.model_var.get():
            first_label = self.available_detection_models[0][0]
            self.model_var.set(first_label)
            self.cli_options["model_choice"] = first_label
        
        # Set default FP16 if not loaded
        if not self.fp16_var.get():
            self.fp16_var.set(True)
            self.cli_options["fp16"] = True

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                selected_model_label = self.model_var.get()
                f.write(f"model={selected_model_label}\n")
                f.write(f"crf={self.crf_var.get()}\n")
                f.write(f"output_dir={self.output_dir}\n")
                f.write(f"ffmpeg_option={self.ffmpeg_option_var.get()}\n")
                f.write(f"fp16={self.fp16_var.get()}\n")  # Save FP16 setting
                f.write(f"max_clip_size={self.max_clip_size_var.get()}\n")  # Save max clip size
                f.write(f"temporal_overlap={self.temporal_overlap_var.get()}\n")  # Save temporal overlap
                f.write(f"detection_score_threshold={self.detection_score_threshold_var.get()}\n")  # NEW: Save detection score threshold
                f.write(f"codec={self.codec_var.get()}\n")  # NEW: Save codec
                f.write(f"encoder_settings={self.encoder_settings_var.get()}\n")  # NEW: Save encoder settings
        except Exception as e:
            self.write_log(f"Failed to save config file: {e}")
            messagebox.showwarning("Warning", f"Failed to save config file: {e}.")

    def write_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} {message}\n"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Failed to write log: {e}")

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if file_path:
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            self.load_video(file_path)

    def drop_file(self, event):
        raw_paths = self.root.tk.splitlist(event.data)
        file_paths = []
        valid_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.ts', '.wmv', '.flv')
        
        for path in raw_paths:
            if path.startswith('{') and path.endswith('}'):
                path = path[1:-1]
            
            path = os.path.normpath(path)
            
            if os.path.exists(path) and path.lower().endswith(valid_extensions):
                file_paths.append(path)
            else:
                self.write_log(f"D&D skip: Invalid file or format: {path}")

        if not file_paths:
            self.write_log("D&D error: No valid files found")
            messagebox.showerror("Error", "No valid video files were dropped.")
            return
        
        if len(file_paths) == 1:
            file_path = file_paths[0]
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, file_path)
            self.load_video(file_path)
            self.current_frame = 0
            self.on_progress_update()
            self.reset_points()
            return
        
        if not messagebox.askyesno("Confirmation", f"{len(file_paths)} files specified. Add to queue?"):
            return
        
        added_files = 0
        for file_path in file_paths:
            with self.cap_lock:
                cap_temp = cv2.VideoCapture(file_path)
                if not cap_temp.isOpened():
                    cap_temp.release()
                    continue
                total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
                cap_temp.release()
            
            queue_entry = {
                'video_path': file_path,
                'model': self.model_var.get(),
                'fp16': self.fp16_var.get(),  # Boolean value
                'max_clip_size': int(self.max_clip_size_var.get()),  # Add max clip size
                'temporal_overlap': int(self.temporal_overlap_var.get()),  # Add temporal overlap
                'detection_score_threshold': float(self.detection_score_threshold_var.get()),  # NEW: Add detection score threshold
                'codec': self.codec_var.get(),  # NEW: Add codec
                'encoder_settings': self.encoder_settings_var.get(),  # NEW: Add encoder settings
                'start_frame': 0,
                'end_frame': total_frames,
                'ffmpeg_option': self.ffmpeg_option_var.get(),
                'save_trimmed': self.save_trimmed_video_var.get(),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'fps': fps,
                'crf_value': int(self.crf_var.get()),
                'vr_processing': self.vr_processing_var.get(),
                'vr_simple_mode': self.vr_simple_mode_var.get()
            }
            
            self.processing_queue.append(queue_entry)
            added_files += 1
        
        if added_files > 0:
            self.save_queue()
            if not self.suppress_queue_message_var.get():
                messagebox.showinfo("Addition Complete", f"{added_files} file(s) added to queue.\nTotal queue items: {len(self.processing_queue)}")
            
            last_file_path = file_paths[-1]
            self.file_path_entry.delete(0, tk.END)
            self.file_path_entry.insert(0, last_file_path)
            self.load_video(last_file_path)
            self.current_frame = 0
            self.on_progress_update()
            self.reset_points()

    def start_batch_processing(self, control_frame):
        if not self.processing_queue:
            messagebox.showinfo("Information", "Queue is empty.")
            return
        if hasattr(self, 'is_running') and self.is_running:
            messagebox.showwarning("Warning", "Processing in progress. Please run after completion.")
            return
        
        for entry in self.processing_queue:
            with self.cap_lock:
                cap_temp = cv2.VideoCapture(entry['video_path'])
                if not cap_temp.isOpened():
                    cap_temp.release()
                    messagebox.showerror("Error", f"Cannot open video file for queue item '{os.path.basename(entry['video_path'])}'. Please remove from queue.")
                    return
                total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_temp.release()

            if entry['ffmpeg_option'] == "no_trim" and (entry['start_frame'] != 0 or entry['end_frame'] != total_frames):
                messagebox.showerror("Error", f"Queue item '{os.path.basename(entry['video_path'])}' has 'As-is' selected but has range specification, cannot execute.")
                return

        self.queue_add_button.config(state=tk.DISABLED)
        self.queue_view_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.batch_button.config(state=tk.DISABLED)
        self.root.bind('<Control-e>', lambda e: None)

        self.is_batch_processing = True
        self.is_running = True
        self.batch_thread = threading.Thread(target=self.batch_process_main)
        self.batch_thread.daemon = True
        self.batch_thread.start()

    def batch_process_main(self):
        original_batch_count = len(self.processing_queue)
        processed_items = 0

        if original_batch_count == 0:
            self.root.after(0, lambda: self.status_label.config(text="Queue is empty", fg="blue"))
            self.is_batch_processing = False
            self.is_running = False
            self.root.after(0, lambda: self.batch_count_label.config(text=""))
            self.root.after(0, lambda: self.queue_add_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.queue_view_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.batch_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.root.bind('<Control-e>', self.add_to_queue))
            return
        
        self.root.after(0, lambda idx=1: self.batch_count_label.config(
            text=f"Batch processing: {idx}/{original_batch_count}",
            fg="red"
        ))
        
        while self.processing_queue and self.is_batch_processing:
            entry = self.processing_queue[0]
            processed_items += 1
            current_count = processed_items
            
            self.root.after(0, lambda idx=current_count: self.batch_count_label.config(
                text=f"Batch processing: {idx}/{original_batch_count}",
                fg="red"
            ))
            
            self.root.after(0, lambda: self.status_label.config(text=f"Processing: {os.path.basename(entry['video_path'])}"))
            self.console_text.config(state=tk.NORMAL)
            self.console_text.insert(tk.END, f"Processing: {os.path.basename(entry['video_path'])}\n")
            self.console_text.config(state=tk.DISABLED)
            
            processing_success = False
            
            try:
                start_time_sec = entry['start_frame'] / entry['fps']
                end_time_sec = entry['end_frame'] / entry['fps']
                
                self.processing_main(
                    entry['video_path'], 
                    start_time_sec, 
                    end_time_sec,
                    entry.get('vr_processing', False) and entry.get('vr_simple_mode', False),
                    entry.get('ffmpeg_option')
                )
                
                processing_success = True
                
            except Exception as e:
                self.write_log(f"Error during processing: {os.path.basename(entry['video_path'])}, Error: {e}")
                self.root.after(0, lambda: self.status_label.config(text=f"Error interrupt: {os.path.basename(entry['video_path'])}", fg="red"))
                self.root.after(0, lambda: messagebox.showerror("Processing Error", f"Error occurred while processing {os.path.basename(entry['video_path'])}. Batch processing interrupted."))
                break
            
            if processing_success and self.is_batch_processing:
                del self.processing_queue[0]
                self.save_queue()

                self.root.after(0, lambda: self.status_label.config(text=f"Completed: {os.path.basename(entry['video_path'])}"))
                self.console_text.config(state=tk.NORMAL)
                self.console_text.insert(tk.END, f"Completed: {os.path.basename(entry['video_path'])}\n")
                self.console_text.config(state=tk.DISABLED)

        self.root.after(0, lambda: self.status_label.config(text="Batch processing completed"))
        if self.show_completion_dialog_var.get():
            self.root.after(0, lambda: messagebox.showinfo("Batch Complete", f"Batch processing completed!\nTotal processed: {original_batch_count} (Processed: {processed_items})"))
        
        self.is_batch_processing = False
        self.is_running = False
        self.root.after(0, lambda: self.batch_count_label.config(text=""))
        self.root.after(0, lambda: self.queue_add_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.queue_view_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.batch_button.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.root.bind('<Control-e>', self.add_to_queue))

    def start_processing(self):
        if not self.validate_inputs():
            return
        
        self.is_batch_processing = False
        self.is_running = True
        
        self.batch_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED, text="Processing...")
        
        self.status_label.config(text="Extracting video...", fg="orange")
        self.console_text.config(state=tk.NORMAL)
        self.console_text.delete('1.0', tk.END)
        self.console_text.insert(tk.END, "Starting processing...\n")
        self.console_text.config(state=tk.DISABLED)
        
        input_file = self.file_path_entry.get()
        start_time_sec = self.start_frame / self.video_fps
        end_time_sec = self.end_frame / self.video_fps
        
        self.processing_thread = threading.Thread(target=self.processing_main, 
                                                  args=(input_file, start_time_sec, end_time_sec, 
                                                        self.vr_processing_var.get() and self.vr_simple_mode_var.get(), 
                                                        self.ffmpeg_option_var.get()))
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def validate_inputs(self):
        if not self.file_path_entry.get():
            messagebox.showerror("Error", "Please select a video file.")
            return False
        if not os.path.exists(self.file_path_entry.get()):
            messagebox.showerror("Error", "Specified video file does not exist.")
            return False
        
        # Validate max clip size
        try:
            max_clip_size = int(self.max_clip_size_var.get())
            if max_clip_size <= 0:
                messagebox.showerror("Error", "--max-clip-size must be > 0")
                return False
        except ValueError:
            messagebox.showerror("Error", "Max Clip Size must be an integer")
            return False
        
        # Validate temporal overlap
        try:
            temporal_overlap = int(self.temporal_overlap_var.get())
            if temporal_overlap < 0:
                messagebox.showerror("Error", "--temporal-overlap must be >= 0")
                return False
            if temporal_overlap >= max_clip_size:
                messagebox.showerror("Error", "--temporal-overlap must be < --max-clip-size")
                return False
        except ValueError:
            messagebox.showerror("Error", "Temporal Overlap must be an integer")
            return False
        
        # Validate detection score threshold
        try:
            detection_score_threshold = float(self.detection_score_threshold_var.get())
            if not (0.0 <= detection_score_threshold <= 1.0):
                messagebox.showerror("Error", "--detection-score-threshold must be in [0, 1]")
                return False
        except ValueError:
            messagebox.showerror("Error", "Detection Score Threshold must be a number")
            return False
        
        # Validate codec
        codec = self.codec_var.get().lower()
        if codec != "hevc":
            messagebox.showerror("Error", f"Unsupported codec: {codec} (only hevc supported)")
            return False
        
        option = self.ffmpeg_option_var.get()
        total_frames = self.video_total_frames
        
        is_range_specified = (self.start_frame > 0) or (self.end_frame < total_frames)
        
        if option == "no_trim":
            if is_range_specified:
                messagebox.showerror("Error", "Range specification is not allowed when 'As-is (No trimming)' is selected.")
                return False
        
        # NEW: Warn about copy modes with range
        if option in ["copy", "copy_genpts"] and is_range_specified:
            if not messagebox.askyesno("Warning", 
                "Copy modes (-c copy) may not trim accurately.\n\n"
                "FFmpeg can only cut on keyframes, which may result in:\n"
                "- Inaccurate start/end times\n"
                "- Wrong duration\n\n"
                "For precise trimming, use 'Re-encode (NVENC)'.\n\n"
                "Continue with copy mode anyway?"):
                return False
        
        if option != "no_trim" and self.start_frame >= self.end_frame:
            messagebox.showerror("Error", "Start frame is greater than or equal to end frame.")
            return False
            
        return True

    def processing_main(self, input_file, start_time_sec, end_time_sec, vr_simple_mode, ffmpeg_option):
        self.write_log(f"Processing started: {os.path.basename(input_file)}")

        unique_id = uuid.uuid4().hex
        input_ext = os.path.splitext(input_file)[1]
        
        trimmed_base_name = f"trimmed_{unique_id}"
        trimmed_file_path = ""
        is_original_file = False  # Flag to track if we're using original file

        try:
            if ffmpeg_option == "no_trim":
                # Use original file directly, no copying needed
                trimmed_file_path = input_file
                is_original_file = True
                
                self.console_text.config(state=tk.NORMAL)
                self.console_text.insert(tk.END, f"Video extraction: Skipped (using original file directly)...\n")
                self.console_text.config(state=tk.DISABLED)
                
                self.console_text.config(state=tk.NORMAL)
                self.console_text.insert(tk.END, "No temporary file needed.\n")
                self.console_text.config(state=tk.DISABLED)

            else:
                trimmed_file_ext = '.mp4' if ffmpeg_option == "re_encode" else input_ext
                trimmed_file_path = os.path.join(self.output_dir, f"{trimmed_base_name}{trimmed_file_ext}")
                
                start_time_str = self.format_time(start_time_sec)
                end_time_str = self.format_time(end_time_sec)
                
                if ffmpeg_option == "re_encode":
                    crf_value = self.crf_var.get()
                    ffmpeg_command = [
                        "ffmpeg", "-y", "-ss", start_time_str, "-to", end_time_str, "-i", input_file,
                        "-c:v", "hevc_nvenc", "-c:a", "aac", "-preset", "fast", "-rc", "vbr_hq", "-cq", crf_value,
                        trimmed_file_path
                    ]
                elif ffmpeg_option == "copy":
                    ffmpeg_command = [
                        "ffmpeg", "-y", "-ss", start_time_str, "-to", end_time_str, "-i", input_file,
                        "-c", "copy", trimmed_file_path
                    ]
                elif ffmpeg_option == "copy_genpts":
                    ffmpeg_command = [
                        "ffmpeg", "-y", "-ss", start_time_str, "-to", end_time_str, "-i", input_file,
                        "-c", "copy", "-fflags", "+genpts", trimmed_file_path
                    ]
                else:
                    raise ValueError("Invalid FFmpeg option.")
                
                self.console_text.config(state=tk.NORMAL)
                self.console_text.insert(tk.END, f"Extracting video...\nExecution command: {' '.join(ffmpeg_command)}\n")
                self.console_text.config(state=tk.DISABLED)
                
                subprocess.run(ffmpeg_command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

                self.console_text.config(state=tk.NORMAL)
                self.console_text.insert(tk.END, "Video extraction completed.\n")
                self.console_text.config(state=tk.DISABLED)

            self.status_label.config(text="Extraction/copy completed. Starting mosaic removal...", fg="green")
            self.save_config()

            is_vr_mode = self.vr_processing_var.get()
            
            if is_vr_mode:
                self.console_text.config(state=tk.NORMAL)
                self.console_text.insert(tk.END, "Running in VR processing mode\n")
                self.console_text.config(state=tk.DISABLED)
                
                # For VR mode, we need to process the file (original or trimmed)
                parts, audio_file = self.split_vr_video(trimmed_file_path, unique_id)
                
                center_file = os.path.join(self.output_dir, f'{unique_id}_center.mp4')
                
                center_processed_temp = self.run_jasna(center_file, unique_id)
                
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                start_time_str_renamed = self.format_time(start_time_sec).replace(':', '')
                end_time_str_renamed = self.format_time(end_time_sec).replace(':', '')
                timestamp_tag = f"{start_time_str_renamed}-{end_time_str_renamed}"
                cli_options_tag = f"model{self.model_var.get()}"

                saved_processed_name = f"{base_name}_{timestamp_tag}_{cli_options_tag}_VR_unmosaiced.mp4"
                saved_processed_path = os.path.join(self.output_dir, saved_processed_name)
                saved_processed_path = self.generate_unique_filepath(saved_processed_path)

                self.merge_vr_video(unique_id, parts, saved_processed_path, audio_file, center_processed_temp, center_file)

                self.status_label.config(text=f"VR processing completed: {os.path.basename(saved_processed_path)}", fg="blue")

                if not self.is_batch_processing and self.show_completion_dialog_var.get():
                    messagebox.showinfo("Complete", f"Video mosaic removal completed! (Mode: VR)\n\nFilename: " + os.path.basename(saved_processed_path))
                
            else:
                # For 2D mode, pass the appropriate file to JASNA
                if is_original_file:
                    # Use original file directly
                    processed_file_path_temp = self.run_jasna(input_file, unique_id)
                else:
                    # Use trimmed file
                    processed_file_path_temp = self.run_jasna(trimmed_file_path, unique_id)
                
                base_name = os.path.splitext(os.path.basename(input_file))[0]
                start_time_str_renamed = self.format_time(start_time_sec).replace(':', '')
                end_time_str_renamed = self.format_time(end_time_sec).replace(':', '')
                timestamp_tag = f"{start_time_str_renamed}-{end_time_str_renamed}"
                cli_options_tag = f"model{self.model_var.get()}"
                
                saved_processed_name = f"{base_name}_{timestamp_tag}_{cli_options_tag}_unmosaiced.mp4"
                saved_processed_path = os.path.join(self.output_dir, saved_processed_name)
                saved_processed_path = self.generate_unique_filepath(saved_processed_path)
                
                os.rename(processed_file_path_temp, saved_processed_path)
                self.status_label.config(text=f"Processed video saved: {os.path.basename(saved_processed_path)}", fg="blue")

                # Only save trimmed video if we created one AND user wants to save it
                if self.save_trimmed_video_var.get() and trimmed_file_path and not is_original_file:
                    saved_trimmed_name = f"{base_name}_{timestamp_tag}_trimmed{input_ext}"
                    saved_trimmed_path = os.path.join(self.output_dir, saved_trimmed_name)
                    saved_trimmed_path = self.generate_unique_filepath(saved_trimmed_path)
                    shutil.move(trimmed_file_path, saved_trimmed_path)
                    trimmed_file_path = ""  # Clear so it won't be deleted in finally block

                if not self.is_batch_processing and self.show_completion_dialog_var.get():
                    messagebox.showinfo("Complete", f"Video mosaic removal completed! (Mode: 2D)\n\nFilename: " + os.path.basename(saved_processed_path))
            
        except Exception as e:
            self.status_label.config(text="Error occurred", fg="red")
            self.console_text.config(state=tk.NORMAL)
            self.console_text.insert(tk.END, f"Error: {e}\n")
            self.console_text.config(state=tk.DISABLED)
            self.write_log(f"Error: {e}")
        finally:
            # Only delete temporary files if:
            # 1. We created a trimmed file (not using original)
            # 2. User doesn't want to save it
            # 3. The file still exists
            if trimmed_file_path and os.path.exists(trimmed_file_path) and not is_original_file and not self.save_trimmed_video_var.get():
                try:
                    os.remove(trimmed_file_path)
                except:
                    pass

            if not self.is_batch_processing:
                self.start_button.config(state=tk.NORMAL, text="Start Processing (Single)")
                self.batch_button.config(state=tk.NORMAL)
            self.is_running = False

    def merge_vr_video(self, unique_id, parts, output_file, audio_file, center_processed_temp, center_file):
        center_processed = center_processed_temp
        
        temp_video = os.path.join(self.output_dir, f'{unique_id}_temp_composited.mp4')
        self.apply_vr_distortion(center_processed, temp_video, unique_id)
        
        if audio_file and os.path.exists(audio_file):
            final_merge_cmd = [
                'ffmpeg', '-y', '-i', temp_video, '-i', audio_file,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-shortest', output_file
            ]
            subprocess.run(final_merge_cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            if os.path.exists(temp_video):
                os.rename(temp_video, output_file)
        
        # Clean up temporary files
        for f in [center_processed, center_file, audio_file, temp_video]:
            if f and os.path.exists(f) and f != output_file:
                try:
                    os.remove(f)
                except:
                    pass

    def load_video(self, file_path):
        with self.cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            
            try:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    file_path_bytes = file_path.encode('utf-8')
                    self.cap = cv2.VideoCapture(file_path_bytes)
                    if not self.cap.isOpened():
                        raise Exception("Cannot open video file")
                        
                self.video_path = file_path
                
                self.video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if raw_fps <= 0 or raw_fps > 120:
                    self.video_fps = 30.0
                else:
                    self.video_fps = raw_fps
                
                self.actual_fps = self.video_fps
                
                self.reset_points()
                
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                
                self.paused = True
                self.play_pause_button.config(text="▶ Play")
                self.on_progress_update()
                self.update_time_labels()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None

    def toggle_play_pause(self, event=None):
        if not self.video_path or not self.cap or not self.cap.isOpened():
            return
        
        if self.paused:
            self.paused = False
            self.play_pause_button.config(text="|| Pause")
            self.buffer_running = True
            self.last_frame_time = time.time()
            self.start_frame_buffer()
            self.update_frame()
        else:
            self.paused = True
            self.buffer_running = False
            self.play_pause_button.config(text="▶ Play")
            self.clear_frame_queue()

    def start_frame_buffer(self):
        if not self.buffer_running or not self.cap or not self.cap.isOpened():
            return
        if not self.frame_buffer_thread or not self.frame_buffer_thread.is_alive():
            self.buffer_running = True
            self.frame_buffer_thread = threading.Thread(target=self.buffer_frames)
            self.frame_buffer_thread.daemon = True
            self.frame_buffer_thread.start()

    def buffer_frames(self):
        while self.buffer_running and self.cap and self.cap.isOpened():
            with self.cap_lock:
                try:
                    ret, frame = self.cap.read()
                    if ret and not self.frame_queue.full():
                        self.frame_queue.put(frame, block=False)
                    elif not ret:
                        self.buffer_running = False
                        self.root.after(0, self.toggle_play_pause)
                        break
                except Exception as e:
                    self.buffer_running = False
                    self.root.after(0, self.toggle_play_pause)
                    break
            target_interval = 1.0 / self.actual_fps if self.actual_fps > 0 else 1.0 / 30.0
            time.sleep(target_interval * 0.5)

    def update_frame(self):
        if not self.cap or not self.cap.isOpened() or self.paused or not self.root.winfo_exists():
            return
        
        current_time = time.time()
        target_interval = 1.0 / self.actual_fps if self.actual_fps > 0 else 1.0 / 30.0
        
        if current_time - self.last_frame_time < target_interval:
            remaining_time = target_interval - (current_time - self.last_frame_time)
            self.root.after(max(1, int(remaining_time * 1000)), self.update_frame)
            return
        
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                with self.cap_lock:
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                if frame is not None:
                    self.display_frame(frame)
                    if self.fullscreen_window:
                        self.display_frame_fullscreen(frame)
                    self.update_time_labels()
                    self.on_progress_update()
                    if self.fullscreen_window:
                        self.update_fullscreen_progress()
                    
                    self.last_frame_time = current_time
                
                if self.current_frame >= self.video_total_frames - 1:
                    self.toggle_play_pause()
                    return
                
                self.root.after(max(1, int(target_interval * 1000)), self.update_frame)
            else:
                self.root.after(10, self.update_frame)
        except Exception as e:
            self.root.after(max(1, int(target_interval * 1000)), self.update_frame)

    def clear_frame_queue(self):
        with self.cap_lock:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass

    def on_progress_update(self):
        if self.video_total_frames > 0:
            width = self.progress_canvas.winfo_width()
            progress_width = (self.current_frame / self.video_total_frames) * width
            self.progress_canvas.coords(self.progress_bar, 0, 0, progress_width, 20)
            
            start_pos = (self.start_frame / self.video_total_frames) * width
            end_pos = (self.end_frame / self.video_total_frames) * width
            self.progress_canvas.coords(self.start_marker, start_pos, 0, start_pos, 20)
            self.progress_canvas.coords(self.end_marker, end_pos, 0, end_pos, 20)

    def on_progress_click(self, event):
        if self.video_total_frames > 0 and self.cap and self.cap.isOpened():
            width = self.progress_canvas.winfo_width()
            click_pos = event.x / width
            new_frame = int(click_pos * self.video_total_frames)
            self.current_frame = new_frame
            self.clear_frame_queue()
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                    if self.fullscreen_window:
                        self.display_frame_fullscreen(frame)
                self.on_progress_update()
                self.update_time_labels()

    def move_frame(self, event):
        if not self.cap or not self.cap.isOpened():
            return
        
        steps = 300 if event.state & 0x0001 == 0 else 30
            
        current_pos = self.current_frame
        new_pos = current_pos
        if event.keysym == 'Right':
            new_pos = min(self.video_total_frames - 1, current_pos + steps)
        elif event.keysym == 'Left':
            new_pos = max(0, current_pos - steps)
            
        self.current_frame = new_pos
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def move_one_frame_backward(self, event=None):
        new_frame = max(0, self.current_frame - 1)
        self.current_frame = new_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def move_one_frame_forward(self, event=None):
        new_frame = min(self.video_total_frames - 1, self.current_frame + 1)
        self.current_frame = new_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def move_one_second_backward(self, event=None):
        step_frames = int(self.video_fps)
        new_frame = max(0, self.current_frame - step_frames)
        self.current_frame = new_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def move_one_second_forward(self, event=None):
        step_frames = int(self.video_fps)
        new_frame = min(self.video_total_frames - 1, self.current_frame + step_frames)
        self.current_frame = new_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def jump_to_start(self, event=None):
        self.current_frame = self.start_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def jump_to_end(self, event=None):
        self.current_frame = self.end_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.end_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def set_start_point_by_key(self, event=None):
        self.start_frame = self.current_frame
        if self.end_frame < self.start_frame:
            self.end_frame = self.start_frame
        self.update_time_labels()
        self.on_progress_update()

    def set_end_point_by_key(self, event=None):
        self.end_frame = self.current_frame
        if self.start_frame > self.end_frame:
            self.start_frame = self.end_frame
        self.update_time_labels()
        self.on_progress_update()

    def jump_to_percentage(self, percentage):
        new_frame = int((percentage / 100) * self.video_total_frames)
        new_frame = min(max(0, new_frame), self.video_total_frames - 1)
        self.current_frame = new_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()
            if self.fullscreen_window:
                self.update_fullscreen_progress()

    def on_mouse_wheel(self, event):
        delta = -1 if event.delta < 0 else 1
        step_frames = int(5 * self.video_fps)
        if delta > 0:
            new_frame = max(0, self.current_frame - step_frames)
        else:
            new_frame = min(self.video_total_frames - 1, self.current_frame + step_frames)
            
        self.current_frame = new_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                if self.fullscreen_window:
                    self.display_frame_fullscreen(frame)
            self.on_progress_update()
            self.update_time_labels()

    def toggle_fullscreen(self, event=None):
        if self.fullscreen_window:
            self.exit_fullscreen()
        else:
            self.fullscreen_window = tk.Toplevel(self.root)
            self.fullscreen_window.attributes('-fullscreen', True)
            self.bind_keys(self.fullscreen_window)
            
            self.fullscreen_label = tk.Label(self.fullscreen_window, bg="black")
            self.fullscreen_label.pack(fill=tk.BOTH, expand=True)
            self.fullscreen_label.bind("<Button-1>", self.toggle_play_pause)
            self.fullscreen_label.bind("<Double-Button-1>", self.toggle_fullscreen)
            self.fullscreen_label.bind("<MouseWheel>", self.on_mouse_wheel)
            
            self.fullscreen_progress_canvas = tk.Canvas(self.fullscreen_window, height=30, bg="grey", highlightthickness=0)
            self.fullscreen_progress_canvas.place(relx=0, rely=1, anchor="sw", relwidth=1)
            self.fullscreen_progress_bar = self.fullscreen_progress_canvas.create_rectangle(0, 0, 0, 30, fill="green")
            self.fullscreen_start_marker = self.fullscreen_progress_canvas.create_line(0, 0, 0, 30, fill="red", width=3)
            self.fullscreen_end_marker = self.fullscreen_progress_canvas.create_line(0, 0, 0, 30, fill="blue", width=3)
            self.fullscreen_progress_text = self.fullscreen_progress_canvas.create_text(10, 15, anchor="w", fill="white", text="00:00:00 / 00:00:00")
            
            self.fullscreen_progress_canvas.bind("<Button-1>", self.on_fullscreen_progress_click)
            self.fullscreen_progress_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
            
            with self.cap_lock:
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = self.cap.read()
                    if ret:
                        self.display_frame_fullscreen(frame)
                    self.update_fullscreen_progress()
                    self.start_frame_buffer()

    def display_frame_fullscreen(self, frame):
        if self.fullscreen_window and frame is not None:
            screen_width = self.fullscreen_window.winfo_screenwidth()
            screen_height = self.fullscreen_window.winfo_screenheight()
            
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if screen_width / screen_height > aspect_ratio:
                new_height = screen_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = screen_width
                new_height = int(new_width / aspect_ratio)
            
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
            black_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            offset_x = (screen_width - new_width) // 2
            offset_y = (screen_height - new_height) // 2
            black_bg[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_frame
            
            rgb_bg = cv2.cvtColor(black_bg, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_bg)
            imgtk = ImageTk.PhotoImage(image=img)
            self.fullscreen_label.configure(image=imgtk)
            self.fullscreen_label.image = imgtk

    def on_fullscreen_progress_click(self, event):
        if not self.video_total_frames > 0 or not self.cap or not self.cap.isOpened():
            return
        
        width = self.fullscreen_progress_canvas.winfo_width()
        if width <= 0:
            return
        click_pos = event.x / width
        new_frame = int(click_pos * self.video_total_frames)
        new_frame = max(0, min(new_frame, self.video_total_frames - 1))
        self.current_frame = new_frame
        self.clear_frame_queue()
        with self.cap_lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.display_frame_fullscreen(frame)
        self.on_progress_update()
        self.update_time_labels()
        self.update_fullscreen_progress()

    def update_fullscreen_progress(self):
        if self.fullscreen_progress_canvas and self.video_total_frames > 0:
            width = self.fullscreen_progress_canvas.winfo_width()
            if width <= 0:
                return
            progress_width = (self.current_frame / self.video_total_frames) * width
            self.fullscreen_progress_canvas.coords(self.fullscreen_progress_bar, 0, 0, progress_width, 30)
            
            start_pos = (self.start_frame / self.video_total_frames) * width
            end_pos = (self.end_frame / self.video_total_frames) * width
            self.fullscreen_progress_canvas.coords(self.fullscreen_start_marker, start_pos, 0, start_pos, 30)
            self.fullscreen_progress_canvas.coords(self.fullscreen_end_marker, end_pos, 0, end_pos, 30)
            
            total_time_sec = self.video_total_frames / self.video_fps if self.video_fps > 0 else 0
            current_time_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
            current_time_str = self.format_time(current_time_sec)
            total_time_str = self.format_time(total_time_sec)
            self.fullscreen_progress_canvas.itemconfig(self.fullscreen_progress_text, text=f"{current_time_str} / {total_time_str}")

    def display_frame(self, frame):
        if frame is None or not hasattr(self, 'video_label') or not self.video_label.winfo_exists():
            self.display_black_frame()
            return
            
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width <= 0 or label_height <= 0:
            self.display_black_frame()
            return
                
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height
        label_aspect_ratio = label_width / label_height
            
        if aspect_ratio > label_aspect_ratio:
            new_width = label_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(new_height * aspect_ratio)

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            
        black_bg = np.zeros((label_height, label_width, 3), dtype=np.uint8)
        offset_x = (label_width - new_width) // 2
        offset_y = (label_height - new_height) // 2
        black_bg[offset_y:offset_y+new_height, offset_x:offset_x+new_width] = resized_frame
            
        rgb_bg = cv2.cvtColor(black_bg, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_bg)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

    def on_window_resize(self, event):
        if hasattr(self, 'after_id') and self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(100, self.update_preview)

    def update_preview(self):
        if self.cap and self.cap.isOpened():
            with self.cap_lock:
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                else:
                    self.display_black_frame()
        else:
            self.display_black_frame()

    def display_black_frame(self):
        if not hasattr(self, 'video_label') or not self.video_label.winfo_exists():
            return
            
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
            
        if label_width > 0 and label_height > 0:
            black_bg = np.zeros((label_height, label_width, 3), dtype=np.uint8)
            rgb_bg = cv2.cvtColor(black_bg, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_bg)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk

    def set_start_point(self):
        self.start_frame = self.current_frame
        if self.end_frame < self.start_frame:
            self.end_frame = self.start_frame
        self.update_time_labels()
        self.on_progress_update()

    def set_end_point(self):
        self.end_frame = self.current_frame
        if self.start_frame > self.end_frame:
            self.start_frame = self.end_frame
        self.update_time_labels()
        self.on_progress_update()

    def reset_points(self):
        self.start_frame = 0
        self.end_frame = self.video_total_frames
        self.update_time_labels()
        self.on_progress_update()

    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_time_labels(self):
        current_time_sec = self.current_frame / self.video_fps if self.video_fps > 0 else 0
        total_time_sec = self.video_total_frames / self.video_fps if self.video_fps > 0 else 0
        current_time_str = self.format_time(current_time_sec)
        total_time_str = self.format_time(total_time_sec)
        self.current_time_label.config(text=f"{current_time_str} / {total_time_str}")
        
        start_time_sec = self.start_frame / self.video_fps if self.video_fps > 0 else 0
        end_time_sec = self.end_frame / self.video_fps if self.video_fps > 0 else 0
        self.start_time_label.config(text=self.format_time(start_time_sec))
        self.end_time_label.config(text=self.format_time(end_time_sec))
            
    def open_log_file(self):
        try:
            if os.path.exists(self.log_file):
                os.startfile(self.log_file) 
            else:
                messagebox.showerror("Error", "Log file not found: LOG_JASNA_GUI.txt")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open log file: {str(e)}")

    def on_closing(self):
        self.buffer_running = False
        with self.cap_lock:
            if self.cap:
                try:
                    self.cap.release()
                    cv2.destroyAllWindows()
                except:
                    pass
                self.cap = None
        if (hasattr(self, 'is_running') and self.is_running) or \
           (hasattr(self, 'is_batch_processing') and self.is_batch_processing):
            if messagebox.askyesno("Confirmation", "Processing is currently running. Interrupt and exit?"):
                if hasattr(self, 'process') and self.process and self.process.poll() is None:
                    self.process.kill()
                self.root.destroy()
        else:
            self.root.destroy()

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    try:
        app = MosaicRemoverApp(root)
        if app.root is not None:
            root.mainloop()
    except Exception as e:
        messagebox.showerror("Startup Error", f"Error occurred while starting program.\n{e}")
    finally:
        if 'root' in locals():
            try:
                root.destroy()
            except:
                pass
