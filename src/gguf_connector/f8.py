import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import cv2
import threading
import time
import numpy as np
from PIL import Image

MODEL_ID = "callgg/fastvlm-0.5b-bf16"
IMAGE_TOKEN_INDEX = -200

# 加载模型和分词器
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

# 全局变量
capturing = False
latest_caption = ""
current_camera = None
cap = None
latest_frame = None

def get_available_cameras():
    """检测可用的摄像头"""
    available_cameras = []
    for i in range(5):  # 检测前5个摄像头索引
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(f"Camera {i}")
            cap.release()
    return available_cameras if available_cameras else ["Camera 0"]

def initialize_camera(camera_index):
    """初始化摄像头"""
    global cap
    if cap is not None:
        cap.release()
    
    # 尝试使用不同的后端
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                # 设置摄像头参数
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return True
        except:
            continue
    
    # 如果都失败，使用默认方式
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    
    return False

def generate_caption(img: Image.Image) -> str:
    """为图像生成描述"""
    try:
        messages = [{"role": "user", "content": "<image>\n请用一句话（32个字以内）描述一下内容"}]
        rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)
        
        pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
        
        px = model.get_vision_tower().image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"].to(model.device, dtype=model.dtype)
        
        with torch.no_grad():
            out = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=32,
                temperature=0.7,
                do_sample=True,
            )
        
        caption = tok.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def capture_loop():
    """持续捕获摄像头画面并生成描述"""
    global latest_caption, capturing, cap, latest_frame
    
    while capturing and cap is not None and cap.isOpened():
        try:
            ret, frame = cap.read()
            if ret:
                # 保存最新帧用于显示
                latest_frame = frame.copy()
                
                # 将OpenCV格式转换为PIL格式用于生成描述
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # 生成描述
                latest_caption = generate_caption(pil_image)
            else:
                latest_caption = "Failed to capture frame"
                
        except Exception as e:
            latest_caption = f"Error: {str(e)}"
        
        time.sleep(1)

def start_caption(camera_name):
    """开始实时描述"""
    global capturing, current_camera
    
    if capturing:
        return "Already running..."
    
    try:
        # 从摄像头名称中提取索引
        if isinstance(camera_name, str):
            camera_index = int(camera_name.split()[-1])
        else:
            camera_index = 0
        
        if initialize_camera(camera_index):
            capturing = True
            current_camera = camera_name
            threading.Thread(target=capture_loop, daemon=True).start()
            return f"Started captioning from {camera_name}"
        else:
            return f"Failed to initialize {camera_name}"
            
    except Exception as e:
        return f"Error starting caption: {str(e)}"

def stop_caption():
    """停止实时描述"""
    global capturing, current_camera, cap
    
    capturing = False
    if cap is not None:
        cap.release()
        cap = None
    current_camera = None
    return "Stopped captioning"

def get_caption():
    """获取最新的描述"""
    return latest_caption if latest_caption else "Waiting for first caption..."

def get_frame():
    """获取最新的摄像头画面"""
    global latest_frame
    if latest_frame is not None:
        # 将BGR转换为RGB用于显示
        rgb_frame = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame
    else:
        # 返回一个黑色图像
        return np.zeros((480, 640, 3), dtype=np.uint8)

# 获取可用摄像头列表
available_cameras = get_available_cameras()

# Gradio UI
with gr.Blocks(title="Live Camera Caption by apple/fastvlm && callgg/fastvlm-0.5b-bf16") as demo:
    gr.Markdown("## 📹 Live Camera Caption by apple/fastvlm && callgg/fastvlm-0.5b-bf16")
    gr.Markdown("Select a camera and start real-time image description")
    
    with gr.Row():
        camera_dropdown = gr.Dropdown(
            choices=available_cameras,
            value=available_cameras[0] if available_cameras else "Camera 0",
            label="Select Camera"
        )
    
    with gr.Row():
        start_btn = gr.Button("▶ Start Captioning", variant="primary")
        stop_btn = gr.Button("⏹ Stop Captioning", variant="stop")
    
    with gr.Row():
        with gr.Column(scale=2):
            camera_view = gr.Image(
                label="Live Camera Feed",
                type="numpy",
                height=480,
                width=640
            )
        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Live Description",
                lines=10,
                placeholder="Camera description will appear here..."
            )
    
    # 事件绑定
    start_btn.click(
        start_caption,
        inputs=[camera_dropdown],
        outputs=output_box
    )
    
    stop_btn.click(
        stop_caption,
        outputs=output_box
    )
    
    # 定时器自动刷新 - 同时刷新描述和画面
    timer = gr.Timer(1)
    timer.tick(get_caption, outputs=output_box)
    timer.tick(get_frame, outputs=camera_view)
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
 