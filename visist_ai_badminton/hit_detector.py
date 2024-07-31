import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving import hdf5_format
import h5py
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Reshape, Bidirectional, GRU, GlobalMaxPool1D, Dense, Softmax
from tensorflow.keras.models import Model

from pathlib import Path

from .pose import Pose

class HitDetector(object):
    def __init__(self, court, poses, trajectory):
        self.court = court
        self.poses = poses
        self.trajectory = trajectory

    # Returns the hits in the given trajectory as well as who hit it
    # Output is are two lists of values:
    #   - List of frame ids where things are hit
    #   - 0 (no hit), 1 (bottom player hits), 2 (top player hits)
    def detect_hits(self):
        pass

class AdhocHitDetector(HitDetector):
    def __init__(self, poses, trajectory):
        super().__init__(None, poses, trajectory)

    def _detect_hits_1d(self, z, thresh=4, window=8):
        z = np.array(z)
        bpts = []
        for i in range(window+1, len(z)-window-1):
            if (z[i]-z[i-1]) * (z[i]-z[i+1]) < 0:
                continue

            left = abs(np.median(z[i-window+1:i+1] - z[i-window:i]))
            right = abs(np.median(z[i+1:i+window+1] - z[i:i+window]))
            if max(left, right) > thresh:
                bpts.append(i)
        return bpts

    def _merge_hits(self, x, y, closeness=2):
        bpt = []
        for t in sorted(x + y):
            if len(bpt) == 0 or bpt[-1] < t - closeness:
                bpt.append(t)
        return bpt

    def _detect_hits(self, x, y, thresh=10, window=7, closeness=15):
        return self._merge_hits(
            self._detect_hits_1d(x, thresh, window),
            self._detect_hits_1d(y, thresh, window),
            closeness
        )

    def detect_hits(self, fps=25):
        Xb, Yb = self.trajectory.X, self.trajectory.Y
        result = self._detect_hits(Xb, Yb)

        is_hit = []
        last_hit = -1
        avg_hit = np.average(np.diff(result))
        for i, fid in enumerate(result):
            if i+1 < len(result) and result[i+1] - fid > 1.6 * avg_hit:
                is_hit.append(0)
                continue

            if fid > self.poses[0].values.shape[0]:
                break

            c = np.array([Xb[fid], Yb[fid]])
            reached_by = 0
            dist_reached = 1e99
            for j in range(2):
                xy = self.poses[j].iloc[fid].to_list()
                pose = Pose()
                pose.init_from_kparray(xy)
                if pose.can_reach(c):
                    pdist = np.linalg.norm(c - pose.get_centroid())
                    if not reached_by or reached_by == last_hit or pdist < dist_reached:
                        reached_by = j + 1
                        dist_reached = pdist

            if reached_by:
                last_hit = reached_by
            is_hit.append(reached_by)

        print('Total shots hit by players:', sum(x > 0 for x in is_hit))
        print('Total impacts detected:', len(result))
        print('Distribution of shot times:')
        plt.hist(np.diff(result))
        print('Average time between shots (s):', np.average(np.diff(result)) / fps)
        return result, is_hit

def scale_data(x):
    x = np.array(x)
    def scale_by_col(x, cols, eps=1e-6):
        x_ = np.array(x[:, cols])
        idx = np.abs(x_) < eps
        m, M = np.min(x_[~idx]), np.max(x_[~idx])
        x_[~idx] = (x_[~idx] - m) / (M - m) + 1
        x[:, cols] = x_
        return x

    even_cols = [2*i for i in range(x.shape[1] // 2)]
    odd_cols = [2*i+1 for i in range(x.shape[1] // 2)]
    x = scale_by_col(x, even_cols)
    x = scale_by_col(x, odd_cols)
    return x

class MLHitDetector(HitDetector):
    @staticmethod
    def create_model(feature_dim, num_consecutive_frames):
        input_layer = Input(shape=(feature_dim,))
        X = input_layer
        X = Reshape(
            target_shape=(num_consecutive_frames, feature_dim // num_consecutive_frames))(X)
        X = Bidirectional(GRU(64, return_sequences=True))(X)
        X = Bidirectional(GRU(64, return_sequences=True))(X)
        X = GlobalMaxPool1D()(X)
        X = Dense(3)(X)
        X = Softmax()(X)
        output_layer = X
        model = Model(input_layer, output_layer)
        return model

    def __init__(self, court, poses, trajectory, model_path, fps=25, debug=True):
        super().__init__(court, poses, trajectory)

        self.fps = fps
        self.debug = debug
        with h5py.File(model_path, mode='r') as f:
            self.temperature = f.attrs['temperature']
            self.model = MLHitDetector.create_model(936, 12) # 12-6-0
            self.model.load_weights(model_path)

        trainable_count = np.sum([K.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([K.count_params(w) for w in self.model.non_trainable_weights])

        if debug:
            print('Number of layers:', len(self.model.layers))
            print('Total params: %d' % (trainable_count + non_trainable_count))
            print('Trainable params: %d' % trainable_count)
            print('Non-trainable params: %d' % non_trainable_count)

    def naive_postprocessing(self, y_pred, detect_thresh=0.1):
        Xb, Yb = self.trajectory.X, self.trajectory.Y
        court_pts = self.court.corners
        num_consec = int(self.model.input_shape[1] // (2 * (34 + 4 + 1)))

        detections = np.where(y_pred[:,0] < detect_thresh)[0]
        result, clusters, who_hit = [], [], []
        min_x, max_x = np.min(court_pts, axis=0)[0], np.max(court_pts, axis=0)[0]
        for t in detections:
            if len(clusters) == 0 or clusters[-1][0] < t - self.fps / 2:
                clusters.append([t])
            else:
                clusters[-1].append(t)

        delta = 0.1 * (max_x - min_x)
        for cluster in clusters:
            any_out = False
            votes = np.array([0.] * y_pred.shape[1])
            for t in cluster:
                if Xb[t] < min_x + delta or Xb[t] > max_x - delta:
                    any_out = True
                    break
                votes += y_pred[t]
            if not any_out:
                gap = 4
                result.append(int(np.median(cluster) + num_consec - gap))
                who_hit.append(int(np.argmax(votes)))

        is_hit = []
        avg_hit = np.average(np.diff(result))
        last_hit, last_time = -1, -1
        to_delete = [0] * len(result)
        for i, fid in enumerate(result):
            if i >= len(who_hit):
                break

            if fid - last_time < 0.8 * self.fps and last_hit == who_hit[i]:
                to_delete[i] = 1
                continue

            is_hit.append(who_hit[i])
            last_time = fid
            last_hit = who_hit[i]

        result = [r for i, r in enumerate(result) if not to_delete[i]]
        return result, is_hit

    def dp_postprocessing(self, y_pred):
        tau = .9 * np.mean(y_pred[y_pred[:, 0] < 0.1, 1:3])
        score = y_pred - tau
        for i in range(3):
            score[:, i] = np.convolve(score[:, i], np.ones(2)/2, mode='same')

        N = y_pred.shape[0]
        T = int(1.2 * N // self.fps)
        D = int(self.fps * .7)
        C = np.full((N, T, 3), -np.inf)
        L = np.zeros((N, T, 3), dtype='int')

        for t in range(3):
            C[:D, 0, t] = score[:D, t]
            L[:D, 0, t] = -1

        for t in range(1, T):
            for d in range(D, N):
                for p1 in range(3):
                    for p2 in range(3):
                        if p1 != p2:
                            c = C[d - D, t - 1, p1] + score[d, p2]
                            if c > C[d, t, p2]:
                                C[d, t, p2] = c
                                L[d, t, p2] = p1

        t = np.argmax(C[-1, -1])
        result = []
        p = np.argmax(C[-1, t])
        d = L[-1, t, p]
        result.append([N-1, p])
        while d >= 0:
            t -= 1
            p = L[d, t, result[-1][1]]
            result.append([d, p])
            d -= D

        result = sorted(result)[1:]
        result, is_hit = list(zip(*result))
        return result, is_hit

    def detect_hits(self, fps=25, postprocessing='naive'):
        X, y = [], []
        for t in range(6, self.poses[0].values.shape[0] - 6):
            kps = []
            for pose in self.poses:
                kps.extend(pose.values[t-6:t+6+1].ravel())
            kps = scale_data(kps)
            traj = np.array([self.trajectory.X[t-6:t+6+1], self.trajectory.Y[t-6:t+6+1]]).T
            traj = scale_data(traj)
            court = self.court.corners.ravel()
            X.append(np.hstack([kps, traj.ravel(), court]).ravel())

        X = np.array(X)
        X = scale_data(X)
        y_pred = np.array([tf.squeeze(self.model(np.array([X[i]]), training=False)).numpy() for i in range(X.shape[0])])

        if self.debug:
            plt.hist(np.argmax(y_pred, axis=-1))
            plt.show()

        if postprocessing == 'naive':
            return self.naive_postprocessing(y_pred)
        else:
            return self.dp_postprocessing(y_pred)

    def detect_hits(self):
        Xb, Yb = self.trajectory.X, self.trajectory.Y
        num_consec = int(self.model.input_shape[1] // (2 * (34 + 4 + 1)))
        court_pts = self.court.corners
        corner_coords = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

        bottom_player = self.poses[0]
        top_player = self.poses[1]

        corners = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

        x_list = []
        L = min(bottom_player.values.shape[0], len(Xb))
        for i in range(num_consec):
            end = L-num_consec+i+1
            x_bird = np.array(list(zip(Xb[i:end], Yb[i:end])))
            x_pose = np.hstack([bottom_player.values[i:end], top_player.values[i:end]])
            x = np.hstack([x_bird, x_pose, np.array([corners for j in range(i, end)])])

            x_list.append(x)
        x_inp = np.hstack(x_list)
        x_inp = scale_data(x_inp)
        x_inp_tensor = tf.convert_to_tensor(x_inp, dtype=tf.float32)
        # Replace the problematic lines with this
        @tf.function
        def compute_logits(model, input_tensor):
            return model(input_tensor)
        
        logits = compute_logits(self.model, x_inp_tensor)
        y_pred = tf.nn.softmax(logits / self.temperature).numpy()

        if self.debug:
            print('Sum of predicted scores:', np.sum(y_pred, axis=0))

        # result, is_hit = self.naive_postprocessing(y_pred)
        result, is_hit = self.dp_postprocessing(y_pred)

        if self.debug:
            num_hits = sum(x > 0 for x in is_hit)
            print('Total shots hit by players:', num_hits)
            if num_hits:
                print('Percentage of shots hit by player 1:', sum(x == 1 for x in is_hit) / num_hits)
            else:
                print('No hits detected.')
            print('Total impacts detected:', len(result))
            print('Distribution of shot times:')
            plt.hist(np.diff(result))
            print('Average time between shots (s):', np.average(np.diff(result)) / self.fps)
        return result, is_hit

def read_hits(hit_prediction_path):

    assert hit_prediction_path.is_file(), f"Invalid path for hit prediction: {hit_prediction_path}"

    hits = pd.read_csv(str(hit_prediction_path)).values

    compressed_hits = np.compress(hits[:,1] != 0, hits, axis=0)
    return {
        "frames": list(compressed_hits[:,0]),
        "player_ids": list(compressed_hits[:,1])
    }
