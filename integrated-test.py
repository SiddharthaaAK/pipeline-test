import os
import shutil
import pandas as pd
from pathlib import Path
import cv2
import subprocess
from visist_ai_badminton.trajectory import Trajectory
from visist_ai_badminton.pose import read_player_poses, process_pose_file
from visist_ai_badminton.court import Court, read_court
from visist_ai_badminton.video_annotator import annotate_video, annotate_video_3d
from visist_ai_badminton.rally_reconstructor import RallyReconstructor, read_hits_file, Court3D

class Pipeline:
    def __init__(self, hit_model):
        self.shotcut_model = shotcut_model
        self.pose_model = pose_model
        self.hit_model = hit_model
        self.output_path = None

    def make_output_dirs(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f'{output_path}/shot', exist_ok=True)
        os.makedirs(f'{output_path}/annotated', exist_ok=True)
        os.makedirs(f'{output_path}/rally_video', exist_ok=True)
        os.makedirs(f'{output_path}/court', exist_ok=True)  # For court detection output
        os.makedirs(f'{output_path}/ball_trajectory', exist_ok=True)  # For ball trajectory output
        os.makedirs(f'{output_path}/poses', exist_ok=True)  # For poses output
        os.makedirs(f'{output_path}/3d', exist_ok=True)  # For 3D trajectory output
        os.makedirs(f'{output_path}/annotated3d', exist_ok=True)  # For annotated 3D video output
    def set_video_variables(self, video_path):
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.video_prefix, _ = os.path.splitext(self.video_name)
    def save_video(self, path=None):
        # Dummy implementation for saving video
        if not path:
            path = f'{self.output_path}/rally_video/{self.video_name}'
        print(f"Saving video to {path}")
    def split_cuts(self, shotcut_model):
        # Dummy implementation for splitting cuts
        print("Splitting cuts using shotcut model")
        return False
    def detect_court(self):
        court_detect_dir = '/home/ubuntu/court-detection'
        out_file = f'/home/ubuntu/pipeline-test/test-output/court/{self.video_prefix}.out'
        
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        
        print(f'Processing video: /home/ubuntu/pipeline/input_video/match1/video/1_01_00.mp4')
        print(f'Changing directory to {court_detect_dir}')
        print(f'Command: ./detect /home/ubuntu/pipeline/input_video/match1/video/1_01_00.mp4 {out_file}')
        
        os.chdir(court_detect_dir)
        
        try:
            result = subprocess.run(
                f'./detect /home/ubuntu/pipeline/input_video/match1/video/1_01_00.mp4 {out_file}',
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ
            )
            print(f'Command output: {result.stdout.decode()}')
            print(f'Command error: {result.stderr.decode()}')
        except subprocess.CalledProcessError as e:
            print(f'Command failed with exit code {e.returncode}')
            print(f'Stdout: {e.stdout.decode()}')
            print(f'Stderr: {e.stderr.decode()}')
            # Check file permissions and existence
            if not os.path.exists(out_file):
                print(f'Output file does not exist: {out_file}')
            else:
                print(f'Output file exists but is not usable: {out_file}')
        finally:
            os.chdir('/root/Visist-Pipeline/Court-Traj')  # Change back to original directory
    
        print(f'Court detection output saved to {out_file}')
        
    def detect_ball_trajectory(self):
        trajectory_dir = '/home/ubuntu/TrackNetV3'
        tracknet_file = '/home/ubuntu/TrackNetV3/ckpts/TrackNet_best.pt'
        inpaintnet_file = '/home/ubuntu/TrackNetV3/ckpts/InpaintNet_best.pt'
        save_dir = f'{self.output_path}/ball_trajectory'
        out_file = f'{save_dir}/{self.video_prefix}_ball.csv'
        batchsize = 8
        # Change directory and run the ball trajectory command
        os.chdir(trajectory_dir)
        python_executable = "/usr/bin/python3"  # Ensure this is the correct path to your Python interpreter
        command = f'{python_executable} predict.py --video_file {self.video_path} --tracknet_file {tracknet_file} --inpaintnet_file {inpaintnet_file} --batch_size {batchsize} --save_dir {save_dir}'
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f'Command output: {result.stdout.decode()}')
            print(f'Command error: {result.stderr.decode()}')
        except subprocess.CalledProcessError as e:
            print(f'Command failed with exit code {e.returncode}')
            print(f'Stdout: {e.stdout.decode()}')
            print(f'Stderr: {e.stderr.decode()}')

        print(f'Ball trajectory output saved to {out_file}')

    def detect_poses(self):
        # Activate mmdet environment and run the first command
        mmdet_command = (
        'cd /home/ubuntu/mmdet'
        'conda run -n aicup python run_mmpose_mmdet.py pisa_faster_rcnn_r50_fpn_1x_coco.py pisa_faster_rcnn_r50_fpn_1x_coco-dea93523.pth --video-path /home/ubuntu/pipeline/input_video/match1/video/1_01_00.mp4  --out-file 1_01_00.out --out-path /home/ubuntu/pipeline-test/test-output/poses/1_01_00/'
        )

        try:
            result = subprocess.run(mmdet_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')
            print(f'Command output (mmdet): {result.stdout.decode()}')
            print(f'Command error (mmdet): {result.stderr.decode()}')
        except subprocess.CalledProcessError as e:
            print(f'mmdet command failed with exit code {e.returncode}')
            print(f'Stdout (mmdet): {e.stdout.decode()}')
            print(f'Stderr (mmdet): {e.stderr.decode()}')

        # Activate mmpose environment and run the second command
        mmpose_command = (
            'cd /home/ubuntu/mmp0251/mmpose-0.25.1'
            'conda run -n mmpose python demo/run_mmpose_pose.py configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth --json-file tests/data/coco/test_coco.json --video-path /home/ubuntu/pipeline/input_video/match1/video/1_01_00.mp4 --out-file 1_01_00.out --out-path /home/ubuntu/pipeline-test/test-output/poses/1_01_00/'
        )

        try:
            result = subprocess.run(mmpose_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, executable='/bin/bash')
            print(f'Command output (mmpose): {result.stdout.decode()}')
            print(f'Command error (mmpose): {result.stderr.decode()}')
        except subprocess.CalledProcessError as e:
            print(f'mmpose command failed with exit code {e.returncode}')
            print(f'Stdout (mmpose): {e.stdout.decode()}')
            print(f'Stderr (mmpose): {e.stderr.decode()}')

        print(f'Pose detection output saved to {self.output_path}/poses/{self.video_prefix}.out and {self.output_path}/poses/{self.video_prefix}_poses.out')

        court_path = '/home/ubuntu/pipeline-test/test-output/court/'
        court_pts = read_court(str(p))
        corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
        court = Court(corners)
        process_pose_file(str('/home/ubuntu/pipeline-test/test-output/poses/1_01_00/1_01_00.out'),
                              str('/home/ubuntu/pipeline-test/test-output/poses/1_01_00/1_01_00'),
                              court,
                              True)

    def detect_hits(self, hit_model, annotate):
        from visist_ai_badminton import hit_detector

        cap = cv2.VideoCapture(f'{self.video_path}')
        if not cap.isOpened():
            print('Error opening video stream or file')
            return

        fps = cap.get(cv2.CAP_PROP_FPS)

        trajectory_path = f'{self.output_path}/ball_trajectory/{self.video_prefix}_ball.csv'
        trajectory = Trajectory(trajectory_path)

        court_pts = read_court(Path(self.output_path) / f'court/{self.video_prefix}.out')
        corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
        court = Court(corners)

        poses_path = Path(self.output_path) / f'poses/{self.video_prefix}'
        poses = read_player_poses(str(poses_path))

        detector = hit_detector.MLHitDetector(
            court,
            poses,
            trajectory,
            model_path=hit_model
        )
        result, is_hit = detector.detect_hits()

        L = len(trajectory.X)
        frames = list(range(L))
        hits = [0] * L
        for fid, pid in zip(result, is_hit):
            hits[fid] = pid

        data = {'frame': frames, 'hit': hits}
        df = pd.DataFrame(data=data)
        df.to_csv(f'{self.output_path}/shot/{self.video_prefix}_hit.csv')

        if annotate and len(result) > 3:
            outfile = f'{self.output_path}/annotated/{self.video_prefix}_annotated.mp4'
            annotate_video(
                cap, court, poses, trajectory, result, is_hit,
                outfile=outfile
            )

            os.system(f"ffmpeg -i {outfile} -c:v libx264 -crf 18 -preset slow -c:a copy tmp.mp4")
            os.system(f"mv tmp.mp4 {outfile}")

    def reconstruct_trajectory(self, visualize=True):
        match_dir = Path(self.video_path).parents[1]
        trajectory_path = {output_directory} / f'ball_trajectory/{self.video_prefix}_ball.csv'
        trajectory = Trajectory(trajectory_path)
        poses_path = match_dir / f'poses/{self.video_prefix}'
        poses = read_player_poses(str(poses_path))
        hits_path = Path(self.output_path) / f'shot/{self.video_prefix}_hit.csv'
        hits = read_hits_file(hits_path)

        court_pts = read_court(match_dir / f'court/{self.video_prefix}.out')
        corners = [court_pts[1], court_pts[2], court_pts[0], court_pts[3]]
        pole_tips = [court_pts[4], court_pts[5]]
        court3d = Court3D(corners, pole_tips)

        cap = cv2.VideoCapture(f'{self.video_path}')
        if not cap.isOpened():
            print('Error opening video stream or file')
            return

        fps = cap.get(cv2.CAP_PROP_FPS)

        reconstructor = RallyReconstructor(
            court3d,
            poses,
            trajectory,
            hits
        )

        try:
            results = reconstructor.reconstruct(fps)
        except IndexError as e:
            print(f"Error during reconstruction: {e}")
            print("No hits detected, skipping trajectory reconstruction.")
            return

        # Write hit to csv file
        L = min(len(trajectory.X), results.shape[0])
        frames = list(range(L))

        data = {
            'frame': frames,
            'ball_x': results[:, 0].tolist(),
            'ball_y': results[:, 1].tolist(),
            'ball_z': results[:, 2].tolist()
        }
        df = pd.DataFrame(data=data)
        df.to_csv(f'{self.output_path}/3d/{self.video_prefix}_3d.csv')

        if visualize:
            annotated2d = f'{self.output_path}/annotated/{self.video_prefix}_annotated.mp4'
            cap = cv2.VideoCapture(annotated2d)
            if not cap.isOpened():
                cap = cv2.VideoCapture(f'{self.video_path}')
            df = pd.read_csv(hits_path)
            is_hit = df['hit'].tolist()

            outfile = f'{self.output_path}/annotated3d/{self.video_prefix}_annotated.mp4'
            annotate_video_3d(cap, results, outfile)

            # Re-encode with ffmpeg from mp41 to mp42
            os.system(f"ffmpeg -i {outfile} -c:v libx264 -crf 18 -preset slow -c:a copy tmp.mp4")
            os.system(f"mv tmp.mp4 {outfile}")

    def run_pipeline(self, video_path, output_path, annotate=True, split_cuts=False):
        logfile = open(f'{output_path}/log.txt', 'w')
        self.output_path = output_path
        self.make_output_dirs(output_path)
        self.set_video_variables(video_path)
        
        logfile.write('Detecting courts...\n')
        self.detect_court()
        
        logfile.write('Detecting ball trajectory...\n')
        self.detect_ball_trajectory()
        
        logfile.write('Detecting poses...\n')
        self.detect_poses()
        
        logfile.write('Detecting hits...\n')
        self.detect_hits(self.hit_model, annotate)
        
        logfile.write('Reconstructing trajectory...\n')
        self.reconstruct_trajectory()
        
        if split_cuts:
            logfile.write('Splitting file into shots...\n')
            status = self.split_cuts(self.shotcut_model)
            if status:
                logfile.write('Error during splitting cuts.\n')
                return

            self.save_video()
            logfile.write('Done file splitting!\n')
        else:
            self.save_video(f'{self.output_path}/rally_video/{self.video_name}')
            logfile.write('Assumed entire video is one rally.\n')

        logfile.write('Pipeline processing complete.\n')
        logfile.close()
        return 0

if __name__ == "__main__":
    hit_model = "/home/ubuntu/pipeline-test/hit-model/hitnet_conv_model1_predict_direction-12-6-0.h5"
    input_video = "/home/ubuntu/pipeline/input_video/match1/video/1_01_00.mp4"
    output_directory = "/home/ubuntu/pipeline-test/test-output"

    pipeline = Pipeline(hit_model)

    # Ensure output directories exist
    pipeline.make_output_dirs(output_directory)

    # Process the input video file
    print(f"Processing video: {input_video}")
    pipeline.run_pipeline(input_video, output_directory, annotate=True, split_cuts=False)
