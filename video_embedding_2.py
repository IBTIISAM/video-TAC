
import av
import torch
import numpy as np
import os
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download

np.random.seed(0)

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")



def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    print("indices: ",indices)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    if len(frames) !=0:        
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    else:
        print('empty frames')
        return frames 


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


# Function to walk through the train folder and extract features
def extract_features_from_train_folder(root_folder, processor, model, clip_len=8):
    features_list = []
    labels_list = []

    # Iterate over each class (subfolder in train)
    class_idx = 0
    for class_folder in os.listdir(root_folder):
        class_folder_path = os.path.join(root_folder, class_folder)
        
        # Only consider directories (class folders)
        if os.path.isdir(class_folder_path):
            # Iterate over each video in the class folder
            for video_file in os.listdir(class_folder_path):
                video_path = os.path.join(class_folder_path, video_file)
                
                # Only process .avi video files
                if video_file.endswith(".avi"):
                    print(f"Processing {video_path}...")

                    container = av.open(video_path)

                    # sample 8 frames
                    indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                    try:
                        video = read_video_pyav(container, indices)
                        
                    
                        inputs = processor(videos=list(video), return_tensors="pt")

                        video_features = model.get_video_features(**inputs)

                        
                        # Append features and the corresponding class index (label)
                        features_list.append(video_features.detach().numpy())  # Remove batch dimension
                        labels_list.append(class_idx)
                    except :
                        print('err')    
                    

        class_idx += 1  # Increment class index
        
    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)#np.concatenate(labels_list, axis=0)
    print("Feature shape:", features.shape, "Label shape:", labels.shape)

    return features, labels


# Main code
if __name__ == "__main__":
    dataset = "UCF101" 
    # Initialize the processor and model
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

    # Path to the 'train' directory (make sure this is correct)
    root_folder = "/home/ibtesam/Documents/ml_project/2024-ICML-TAC/data/UCF101/train"  # Update this to your actual path

    # Extract features and labels
    features, labels = extract_features_from_train_folder(root_folder, processor, model, clip_len=8)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    # Path to the 'train' directory (make sure this is correct)
    root_folder = "/home/ibtesam/Documents/ml_project/2024-ICML-TAC/data/UCF101/test"  # Update this to your actual path

    # Extract features and labels
    features_test, labels_test = extract_features_from_train_folder(root_folder, processor, model, clip_len=8)
    print("Features shape:", features_test.shape)
    print("Labels shape:", labels_test.shape)

    np.save("./data/" + dataset + "_image_embedding_train.npy", features)
    np.save("./data/" + dataset + "_image_embedding_test.npy", features_test)
    np.savetxt("./data/" + dataset + "_labels_train.txt", labels)
    np.savetxt("./data/" + dataset + "_labels_test.txt", labels_test)

    print(labels_test)

    
    

