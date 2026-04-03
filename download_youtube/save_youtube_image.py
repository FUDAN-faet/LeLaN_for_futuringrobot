import os
import shutil
import subprocess
import cv2
from pytubefix import YouTube
from pytubefix.exceptions import BotDetection

with open('list_youtube_video_release.txt', 'r') as file:
    VIDEO_URL_txt = file.read().splitlines()

VIDEO_URL_list = []
for video_url_folder in VIDEO_URL_txt:
    print(video_url_folder)
    vvideo_url, folder, num_skip = video_url_folder.split(" ")
    VIDEO_URL_list.append([vvideo_url, folder, float(num_skip)])
    
# Configuration
OUTPUT_DIR = './videos'
WEB_CLIENT = "WEB"

# Create directories if they don't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def download_video_with_ytdlp(url, video_path):
    browser = os.environ.get("YTDLP_BROWSER", "").strip()
    cookies_file = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
    user_agent = os.environ.get("YTDLP_USER_AGENT", "").strip()
    proxy = os.environ.get("YTDLP_PROXY", "").strip()

    command = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format",
        "mp4",
        "-o",
        video_path,
        url,
    ]

    if browser:
        print(f"Using browser cookies from: {browser}")
        command.extend(["--cookies-from-browser", browser])
    elif cookies_file:
        print(f"Using cookies file: {cookies_file}")
        command.extend(["--cookies", cookies_file])

    if user_agent:
        command.extend(["--user-agent", user_agent])

    if proxy:
        command.extend(["--proxy", proxy])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "yt-dlp 下载失败。当前更像是 YouTube 的风控问题，不是脚本逻辑问题。"
            "请先在浏览器里打开任意 YouTube 页面并完成验证码/登录，然后用同一浏览器的 cookies 重试。"
            "推荐：`export YTDLP_BROWSER=firefox` 后再运行本脚本。"
        ) from exc

# Download the video
def download_video(url, output_path, filename):
    video_path = os.path.join(output_path, filename)
    if os.path.exists(video_path):
        print(f"Skip download, file already exists: {video_path}")
        return

    if shutil.which("yt-dlp") is not None:
        print("Using yt-dlp to download the video.")
        download_video_with_ytdlp(url, video_path)
        return

    try:
        # pytubefix 官方现在建议在遇到 bot detection 时切换到 WEB client。
        yt = YouTube(url, client=WEB_CLIENT)
        stream = yt.streams.get_highest_resolution()
    except BotDetection as exc:
        if shutil.which("node") is None:
            raise RuntimeError(
                "YouTube 拒绝了当前请求。当前脚本已切换到 WEB client，但本机未安装 Node.js，"
                "pytubefix 无法自动生成 PoToken。请先安装 nodejs 后再重试。"
            ) from exc
        raise RuntimeError(
            "pytubefix 在生成 PoToken 时被 YouTube 风控拦住了。"
            "更稳的方案是安装 yt-dlp 后重跑：`python -m pip install yt-dlp`。"
        ) from exc

    print(f"Downloading video: {yt.title}")
    stream.download(output_path=output_path, filename=filename)
    print("Download completed!")

# Extract images from the video
def extract_frames(video_path, save_folder, num_skip):
    images_dir = os.path.join("../dataset/dataset_LeLaN_youtube", save_folder, 'image')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)    

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    pickles_dir = os.path.join("../dataset/dataset_LeLaN_youtube", save_folder, 'pickle')
    if not os.path.exists(pickles_dir):
        raise FileNotFoundError(f"缺少标注目录: {pickles_dir}")
    n_pic = len(os.listdir(pickles_dir))
    #n_pic = 10
        
    print("fps", fps, "N_pic", n_pic, "folder", save_folder)
    if fps == 0.0:
        fps = 15    

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds_per_frame = 1 / fps
    
    frame_count = 0
    save_count = 0 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # save image at 2 fps (removing first num_skip images. The total frames is less than n_pic.)
        if num_skip == 60.0 or num_skip == 5.0:
            if frame_count % int(fps*0.5) == 0 and frame_count > num_skip*fps and save_count < n_pic:      
                frame_process = frame
                print(frame_count)
                cv2.imwrite(images_dir + "/" + str(save_count).zfill(8) + ".jpg", cv2.resize(frame_process, (224, 224), interpolation = cv2.INTER_LINEAR))
                save_count += 1
        else:
            if (frame_count - int(num_skip*fps) - 1) % int(fps*0.5) == 0 and frame_count > num_skip*fps and save_count < n_pic:        
                frame_process = frame
                print(frame_count)
                cv2.imwrite(images_dir + "/" + str(save_count).zfill(8) + ".jpg", cv2.resize(frame_process, (224, 224), interpolation = cv2.INTER_LINEAR))
                save_count += 1
                        
        if save_count == n_pic:
            break
        frame_count += 1
        
    cap.release()
    print(f"Extracted {frame_count} frames!")

# Main function
def main(VIDEO_URL, folder, num_skip):
    # Define paths
    video_file = os.path.join(OUTPUT_DIR, folder + '.mp4')
    # Download video
    print(VIDEO_URL, OUTPUT_DIR)
    download_video(VIDEO_URL, OUTPUT_DIR, folder + '.mp4')

    # Extract frames
    extract_frames(video_file, folder, num_skip)

if __name__ == "__main__":
    for VIDEO_URL, folder, num_skip in VIDEO_URL_list:
        try:
            main(VIDEO_URL, folder, num_skip)
        except Exception as exc:
            print(f"[FAILED] {folder}: {exc}")
