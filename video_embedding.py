from transformers import XCLIPProcessor, XCLIPModel
import torch

model_name = "microsoft/xclip-base-patch32"
processor = XCLIPProcessor.from_pretrained(model_name)
model = XCLIPModel.from_pretrained(model_name)

from decord import VideoReader, cpu
import numpy as np
file_path = "./video2960.mp4"
np.random.seed(0)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

vr = VideoReader(file_path, num_threads=1, ctx=cpu(0))

# sample 16 frames
vr.seek(0)
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=len(vr))
video = vr.get_batch(indices).asnumpy()
print(video.shape)



inputs = processor(text=["a cartoon animals runs through an ice cave in a video game", "eating spaghetti", "go shopping"], videos=list(video), return_tensors="pt", padding=True)

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_per_video.softmax(dim=1)
print(np.argmax(probs.numpy(), axis=1))


from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

inputs = tokenizer(["a cartoon animals runs through an ice cave in a video game"], padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)
print(text_features.shape)

import av
import torch
import numpy as np

from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


# video clip consists of 300 frames (10 seconds at 30 FPS)
#file_path = hf_hub_download(
 #   repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
#)
file_path = './video2960.mp4'
container = av.open(file_path)

# sample 8 frames
indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
video = read_video_pyav(container, indices)

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

inputs = processor(videos=list(video), return_tensors="pt")

video_features = model.get_video_features(**inputs)

print(video_features.shape)