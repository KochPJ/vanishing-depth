# ------------------------------------------------------------------------
# PoET: Pose Estimation Transformer for Single-View, Multi-Object 6D Pose Estimation
# Copyright (c) 2022 Thomas Jantos (thomas.jantos@aau.at), University of Klagenfurt - Control of Networked Systems (CNS). All Rights Reserved.
# Licensed under the BSD-2-Clause-License with no commercial use [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from CDPN (https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi)
# Licensed under the Apache License, Version 2.0 [see LICENSE_CDPN in the LICENSES folder for details]
# ------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

import os
import shutil
import copy
import json
import struct

from scipy import spatial
import numpy as np
import os
from scipy.linalg import logm
import numpy.linalg as LA
try:
    from scipy.integrate import simps
except:
    from scipy.integrate import simpson as simps
#from model.deformable_detr.pose_evaluator_lmo import PoseEvaluatorLMO


class PoseEvaluator(object):
    def __init__(self, models, classes, model_info, model_symmetry, depth_scale=0.1):
        """
        Initialization of the Pose Evaluator for the YCB-V dataset.

        It can calculate the average rotation and translation error, as well as the ADD, ADD-S and ADD-(S) metric
        as described in https://arxiv.org/pdf/1711.00199.pdf (PoseCNN)

        Parameters
            - models: Array containing the points of each object 3D model (Contains the 3D points for each class)
            - classes: Array containing the information about the object classes (mapping between class ids and class names)
            - model_info: Information about the models (diameter and extension)
            - model_symmetry: Indication whether the 3D model of a certain class is symmetric (axis, plane) or not.
        """
        self.models = models
        self.classes = classes
        self.models_info = model_info
        self.model_symmetry = model_symmetry

        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}
        self.depth_scale = depth_scale

        self.reset()  # Initialize

        print('classes in pose_evaluator',self.classes)

    def reset(self):
        """
        Reset the PoseEvaluator stored poses. Necessary when the same evaluator is used during training
        """
        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}

        for cls in self.classes:
            self.num[cls] = 0.
            self.poses_pred[cls] = []
            self.poses_gt[cls] = []
            self.poses_img[cls] = []
            self.camera_intrinsics[cls] = []

    def evaluate_pose_adds(self, output_path):
        """
        Evaluate 6D pose by ADD(-S) metric
        Symmetric Object --> ADD-S
        NonSymmetric Objects --> ADD

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """
        output_dir = output_path + "adds/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "adds/adds.log", 'w')
        json_file = open(output_path + "adds/adds.json", 'w')

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        models = self.models
        model_symmetry = self.model_symmetry

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ADD(-S)', '-' * 100))
        log_file.write("\n")

        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {k: np.zeros((n_classes), dtype=np.float32) for k in ['0.02', '0.05', '0.10', '0.10p-dia']}

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        threshold_010p = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1))  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        adds_results = {}
        adds_results["thresholds"] = [0.02, 0.05, 0.10]

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10
            threshold_010p[i] = 0.10 * float(self.models_info[cls_name]['diameter']) / 1000

            symmetry_flag = model_symmetry[cls_name]
            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]['pts']
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                if symmetry_flag:
                    eval_method = 'adi'
                    error = self.calc_adi(model_pts, pose_pred, pose_gt)
                else:
                    eval_method = 'add'
                    error = self.calc_add(model_pts, pose_pred, pose_gt)
                if error < threshold_002[i]:
                    count_correct['0.02'][i] += 1
                if error < threshold_005[i]:
                    count_correct['0.05'][i] += 1
                if error < threshold_010[i]:
                    count_correct['0.10'][i] += 1
                if error < threshold_010p[i]:
                    count_correct['0.10p-dia'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1

            adds_results[cls_name] = {}
            adds_results[cls_name]["threshold"] = {'0.02': count_correct['0.02'][i].tolist(),
                                                   '0.05': count_correct['0.05'][i].tolist(),
                                                   '0.10': count_correct['0.10'][i].tolist(),
                                                   '0.10p-dia': count_correct['0.10p-dia'][i].tolist(),
                                                   'mean': count_correct['mean'][i].tolist()}

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        sum_acc_010p = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct['0.02'][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct['0.05'][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct['0.10'][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010
            acc_010p = 100 * float(count_correct['0.10p-dia'][i]) / float(count_all[i])
            sum_acc_010p[0] += acc_010p

            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(acc_mean))
            log_file.write("\n")
            log_file.write('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.02'][i],
                count_all[i],
                acc_002))
            log_file.write("\n")
            log_file.write('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.05'][i],
                count_all[i],
                acc_005))
            log_file.write("\n")
            log_file.write('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10'][i],
                count_all[i],
                acc_010))
            log_file.write("\n")
            log_file.write('threshold=0.10p-dia, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10p-dia'][i],
                count_all[i],
                acc_010p))
            log_file.write("\n")
            log_file.write("\n")
            adds_results[cls_name]["accuracy"] = {'n_poses': count_all[i].tolist(),
                                                  '0.02': acc_002,
                                                  '0.05': acc_005,
                                                  '0.10': acc_010,
                                                  '0.10p-dia': acc_010p,
                                                  'auc': acc_mean}

        log_file.write("=" * 30)
        log_file.write("\n")

        for iter_i in range(1):
            log_file.write("---------- ADD(-S) performance over {} classes -----------".format(num_valid_class))
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.02, mean accuracy: {:.2f}'.format(
                sum_acc_002[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.05, mean accuracy: {:.2f}'.format(
                sum_acc_005[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10, mean accuracy: {:.2f}'.format(
                sum_acc_010[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10p-dia, mean accuracy: {:.2f}'.format(
                sum_acc_010p[iter_i] / num_valid_class))
            log_file.write("\n")
        log_file.write("=" * 30)
        adds_results["accuracy"] = {'0.02': sum_acc_002[0].tolist() / num_valid_class,
                                    '0.05': sum_acc_005[0].tolist() / num_valid_class,
                                    '0.10': sum_acc_010[0].tolist() / num_valid_class,
                                    '0.10p-dia': sum_acc_010p[0].tolist() / num_valid_class,
                                    'auc': sum_acc_mean[0].tolist() / num_valid_class}

        log_file.write("\n")
        log_file.close()
        json.dump(adds_results, json_file)
        json_file.close()
        return adds_results["accuracy"]

    def evaluate_pose_adi(self, output_path):
        """
        Evaluate 6D pose by ADD-S metric

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """
        output_dir = output_path + "adi/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "adi/adds.log", 'w')
        json_file = open(output_path + "adi/adds.json", 'w')

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        models = self.models

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ADD-S', '-' * 100))
        log_file.write("\n")

        eval_method = 'adi'
        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {k: np.zeros((n_classes), dtype=np.float32) for k in ['0.02', '0.05', '0.10', '0.10p-dia']}

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        threshold_010p = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1))  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        adi_results = {}
        adi_results["thresholds"] = [0.02, 0.05, 0.10]

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10
            threshold_010p[i] = 0.10 * float(self.models_info[cls_name]['diameter']) / 1000

            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]['pts']
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                error = self.calc_adi(model_pts, pose_pred, pose_gt)
                if error < threshold_002[i]:
                    count_correct['0.02'][i] += 1
                if error < threshold_005[i]:
                    count_correct['0.05'][i] += 1
                if error < threshold_010[i]:
                    count_correct['0.10'][i] += 1
                if error < threshold_010p[i]:
                    count_correct['0.10p-dia'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1
            adi_results[cls_name] = {}
            adi_results[cls_name]["threshold"] = {'0.02': count_correct['0.02'][i].tolist(),
                                                  '0.05': count_correct['0.05'][i].tolist(),
                                                  '0.10': count_correct['0.10'][i].tolist(),
                                                  '0.10p-dia': count_correct['0.10p-dia'][i].tolist(),
                                                  'mean': count_correct['mean'][i].tolist()}

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        sum_acc_010p = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct['0.02'][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct['0.05'][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct['0.10'][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010
            acc_010p = 100 * float(count_correct['0.10p-dia'][i]) / float(count_all[i])
            sum_acc_010p[0] += acc_010p

            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(acc_mean))
            log_file.write("\n")
            log_file.write('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.02'][i],
                count_all[i],
                acc_002))
            log_file.write("\n")
            log_file.write('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.05'][i],
                count_all[i],
                acc_005))
            log_file.write("\n")
            log_file.write('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10'][i],
                count_all[i],
                acc_010))
            log_file.write("\n")
            log_file.write('threshold=0.10p-dia, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10p-dia'][i],
                count_all[i],
                acc_010p))
            log_file.write("\n")
            log_file.write("\n")

            adi_results[cls_name]["accuracy"] = {'n_poses': count_all[i].tolist(),
                                                 '0.02': acc_002,
                                                 '0.05': acc_005,
                                                 '0.10': acc_010,
                                                 '0.10p-dia': acc_010p,
                                                 'auc': acc_mean}

        log_file.write("=" * 30)
        log_file.write('\n')

        for iter_i in range(1):
            log_file.write("---------- ADD-S performance over {} classes -----------".format(num_valid_class))
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.02, mean accuracy: {:.2f}'.format(
                sum_acc_002[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.05, mean accuracy: {:.2f}'.format(
                sum_acc_005[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10, mean accuracy: {:.2f}'.format(
                sum_acc_010[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10p-dia, mean accuracy: {:.2f}'.format(
                sum_acc_010p[iter_i] / num_valid_class))
            log_file.write("\n")
        log_file.write("=" * 30)
        adi_results["accuracy"] = {'0.02': sum_acc_002[0].tolist() / num_valid_class,
                                   '0.05': sum_acc_005[0].tolist() / num_valid_class,
                                   '0.10': sum_acc_010[0].tolist() / num_valid_class,
                                   '0.10p-dia': sum_acc_010p[0].tolist() / num_valid_class,
                                   'auc': sum_acc_mean[0].tolist() / num_valid_class}

        log_file.write("\n")
        log_file.close()
        json.dump(adi_results, json_file)
        json_file.close()
        return adi_results["accuracy"]

    def evaluate_pose_add(self, output_path):
        """
        Evaluate 6D pose by ADD Metric

        For metric definition we refer to PoseCNN: https://arxiv.org/pdf/1711.00199.pdf
        """

        output_dir = output_path + "/add/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "add/add.log", 'w')
        json_file = open(output_path + "add/add.json", 'w')


        json_file_results = {} #open(output_path + "add/add_results.json", 'w')

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        models_info = self.models_info
        models = self.models

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ADD', '-' * 100))
        log_file.write("\n")

        eval_method = 'add'
        n_classes = len(self.classes)
        count_all = np.zeros((n_classes), dtype=np.float32)
        count_correct = {k: np.zeros((n_classes), dtype=np.float32) for k in ['0.02', '0.05', '0.10', '0.10p-dia']}

        threshold_002 = np.zeros((n_classes), dtype=np.float32)
        threshold_005 = np.zeros((n_classes), dtype=np.float32)
        threshold_010 = np.zeros((n_classes), dtype=np.float32)
        threshold_010p = np.zeros((n_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32), (n_classes, 1)) # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((n_classes, num_thresh), dtype=np.float32)

        add_results = {}
        add_results["thresholds"] = [0.02, 0.05, 0.10]

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02
            threshold_005[i] = 0.05
            threshold_010[i] = 0.10
            threshold_010p[i] = 0.10 * float(self.models_info[cls_name]['diameter']) / 1000
            # threshold_mean[i, :] *= models_info[cls_name]['diameter']
            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]['pts']
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            for j in range(n_poses):
                pose_pred = cls_poses_pred[j]  # est pose
                pose_gt = cls_poses_gt[j]  # gt pose
                error = self.calc_add(model_pts, pose_pred, pose_gt)
                json_file_results['{}_{}'.format(cls_name, j)] = {'error': float(error), 'pred': [float(v) for v in pose_pred.flatten()], 'gt': [float(v) for v in pose_gt.flatten()]}

                if error < threshold_002[i]:
                    count_correct['0.02'][i] += 1
                if error < threshold_005[i]:
                    count_correct['0.05'][i] += 1
                if error < threshold_010[i]:
                    count_correct['0.10'][i] += 1
                if error < threshold_010p[i]:
                    count_correct['0.10p-dia'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1
            add_results[cls_name] = {}
            add_results[cls_name]["threshold"] = {'0.02': count_correct['0.02'][i].tolist(),
                                                  '0.05': count_correct['0.05'][i].tolist(),
                                                  '0.10': count_correct['0.10'][i].tolist(),
                                                  '0.10p-dia': count_correct['0.10p-dia'][i].tolist(),
                                                  'mean': count_correct['mean'][i].tolist()}

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        sum_acc_010p = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            log_file.write("** {} **".format(cls_name))
            
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct['0.02'][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct['0.05'][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct['0.10'][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010
            acc_010p = 100 * float(count_correct['0.10p-dia'][i]) / float(count_all[i])
            sum_acc_010p[0] += acc_010p

            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(acc_mean))
            log_file.write("\n")
            log_file.write('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.02'][i],
                count_all[i],
                acc_002))
            log_file.write("\n")
            log_file.write('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.05'][i],
                count_all[i],
                acc_005))
            log_file.write("\n")
            log_file.write('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10'][i],
                count_all[i],
                acc_010))
            log_file.write("\n")
            log_file.write('threshold=0.10p-dia, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10p-dia'][i],
                count_all[i],
                acc_010p))
            log_file.write("\n")
            log_file.write("\n")
            add_results[cls_name]["accuracy"] = {'n_poses': count_all[i].tolist(),
                                                  '0.02': acc_002,
                                                  '0.05': acc_005,
                                                  '0.10': acc_010,
                                                  '0.10p-dia': acc_010p,
                                                  'auc': acc_mean}

        log_file.write("=" * 30)
        log_file.write("\n")

        with open(output_path + "add/add_results.json", 'w') as f_results:
            json.dump(json_file_results, f_results)
        #json_file_results (output_path + "add/add_results.json", 'w')

        for iter_i in range(1):
            log_file.write("---------- ADD performance over {} classes -----------".format(num_valid_class))
            log_file.write("\n")
            log_file.write("** iter {} **".format(iter_i + 1))
            log_file.write("\n")
            log_file.write('threshold=[0.0, 0.10], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.02, mean accuracy: {:.2f}'.format(
                sum_acc_002[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.05, mean accuracy: {:.2f}'.format(
                sum_acc_005[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10, mean accuracy: {:.2f}'.format(
                sum_acc_010[iter_i] / num_valid_class))
            log_file.write("\n")
            log_file.write('threshold=0.10-dia, mean accuracy: {:.2f}'.format(
                sum_acc_010p[iter_i] / num_valid_class))
            log_file.write("\n")
        log_file.write("=" * 30)

        add_results["accuracy"] = {'0.02': sum_acc_002[0].tolist() / num_valid_class,
                                    '0.05': sum_acc_005[0].tolist() / num_valid_class,
                                    '0.10': sum_acc_010[0].tolist() / num_valid_class,
                                    '0.10p-dia': sum_acc_010p[0].tolist() / num_valid_class,
                                    'auc': sum_acc_mean[0].tolist() / num_valid_class}

        log_file.write("\n")
        log_file.close()
        json.dump(add_results, json_file)
        json_file.close()
        return

    def calculate_class_avg_translation_error(self, output_path):
        """
        Calculate the average translation error for each class and then the average error across all classes in meters
        """
        output_dir = output_path + "/avg_t_error/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/avg_t_error/avg_t_error.log", 'w')
        json_file = open(output_path + "avg_t_error/avg_t_error.json", 'w')

        log_file.write('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric Average Translation Error in Meters', '-' * 100))
        log_file.write("\n")

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        translation_errors = []
        cls_translation_errors = {}
        avg_translation_errors = {}
        for cls in self.classes:
            cls_translation_errors[cls] = []
            cls_poses_pred = poses_pred[cls]
            cls_poses_gt = poses_gt[cls]
            for pose_est, pose_gt in zip(cls_poses_pred, cls_poses_gt):
                t_est = pose_est[:, 3]
                t_gt = pose_gt[:, 3]
                error = np.sqrt(np.sum(np.square((t_est - t_gt))))
                cls_translation_errors[cls].append(error)
                translation_errors.append(error)
            if len(cls_translation_errors[cls]) != 0:
                avg_error = np.sum(cls_translation_errors[cls]) / len(cls_translation_errors[cls])
                avg_translation_errors[cls] = avg_error
            else:
                avg_translation_errors[cls] = np.nan
            log_file.write("Class: {} \t\t {}".format(cls, avg_translation_errors[cls]))
            log_file.write("\n")
        total_avg_error = np.sum(translation_errors) / len(translation_errors)
        log_file.write("All:\t\t\t\t\t {}".format(total_avg_error))
        avg_translation_errors["mean"] = [total_avg_error]

        log_file.write("\n")
        log_file.close()
        json.dump(avg_translation_errors, json_file)
        json_file.close()
        return avg_translation_errors["mean"]

    def calculate_class_avg_rotation_error(self, output_path):
        """
        Calculate the average rotation error given by the Geodesic distance for each class and then the average error
        across all classes in degree
        """
        output_dir = output_path + "/avg_rot_error/"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        log_file = open(output_path + "/avg_rot_error/avg_rot_error.log", 'w')
        json_file = open(output_path + "avg_rot_error/avg_rot_error.json", 'w')

        log_file.write(
            '\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric Average Rotation Error in Degrees', '-' * 100))
        log_file.write("\n")

        poses_pred = copy.deepcopy(self.poses_pred)
        poses_gt = copy.deepcopy(self.poses_gt)
        rotation_errors = []
        cls_rotation_errors = {}
        avg_rotation_errors = {}

        for cls in self.classes:
            cls_rotation_errors[cls] = []
            cls_pose_pred = poses_pred[cls]
            cls_pose_gt = poses_gt[cls]
            for pose_est, pose_gt in zip(cls_pose_pred, cls_pose_gt):
                rot_est = pose_est[:3, :3]
                rot_gt = pose_gt[:3, :3]
                rot = np.matmul(rot_est, rot_gt.T)
                trace = np.trace(rot)
                if trace < -1.0:
                    trace = -1
                elif trace > 3.0:
                    trace = 3.0
                angle_diff = np.degrees(np.arccos(0.5 * (trace - 1)))
                cls_rotation_errors[cls].append(angle_diff)
                rotation_errors.append(angle_diff)
            if len(cls_rotation_errors[cls]) != 0:
                avg_error = np.sum(cls_rotation_errors[cls]) / len(cls_rotation_errors[cls])
                avg_rotation_errors[cls] = avg_error
            else:
                avg_rotation_errors[cls] = np.nan
            log_file.write("Class: {} \t\t {}".format(cls, avg_rotation_errors[cls]))
            log_file.write("\n")
        total_avg_error = np.sum(rotation_errors) / len(rotation_errors)
        log_file.write("All:\t\t\t\t\t {}".format(total_avg_error))
        avg_rotation_errors["mean"] = [total_avg_error]

        log_file.write("\n")
        log_file.close()
        json.dump(avg_rotation_errors, json_file)
        json_file.close()
        return avg_rotation_errors["mean"]

    def se3_mul(self, RT1, RT2):
        """
        Concat 2 RT transform
        :param RT1=[R,T], 4x3 np array
        :param RT2=[R,T], 4x3 np array
        :return: RT_new = RT1 * RT2
        """
        R1 = RT1[0:3, 0:3]
        T1 = RT1[0:3, 3].reshape((3, 1))

        R2 = RT2[0:3, 0:3]
        T2 = RT2[0:3, 3].reshape((3, 1))

        RT_new = np.zeros((3, 4), dtype=np.float32)
        RT_new[0:3, 0:3] = np.dot(R1, R2)
        T_new = np.dot(R1, T2) + T1
        RT_new[0:3, 3] = T_new.reshape((3))
        return RT_new

    def transform_pts(self, pts, rot, t):
        """
        Applies a rigid transformation to 3D points.

        :param pts: nx3 ndarray with 3D points.
        :param rot: 3x3 rotation matrix.
        :param t: 3x1 translation vector.
        :return: nx3 ndarray with transformed 3D points.
        """
        assert (pts.shape[1] == 3)
        pts_t = rot.dot(pts.T) + t.reshape((3, 1))
        return pts_t.T

    def project_pts(self, pts, rot, t, K):
        """
        Applies a rigid transformation to 3D points.

        :param pts: nx3 ndarray with 3D points.
        :param rot: 3x3 rotation matrix.
        :param t: 3x1 translation vector.
        :param K: 3x3 intrinsic matrix
        :return: nx2 ndarray with transformed 2D points.
        """
        assert (pts.shape[1] == 3)
        if K.shape == (9,):
            K = K.reshape(3, 3)
        pts_t = rot.dot(pts.T) + t.reshape((3, 1))  # 3xn
        pts_c_t = K.dot(pts_t)
        n = pts.shape[0]
        pts_2d = np.zeros((n, 2))
        pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
        pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

        return pts_2d

    def proj(self, pts, pose_pred, pose_gt, K):
        '''
        average re-projection error in 2d

        :param pts: nx3 ndarray with 3D model points.
        :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
        :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
        :param K: Camera intrinsics to project the model onto the image plane.
        :return:
        '''
        rot_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:, 3]

        rot_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:, 3]

        proj_pred = self.project_pts(pts, rot_pred, t_pred, K)
        proj_gt = self.project_pts(pts, rot_gt, t_gt, K)
        e = np.linalg.norm(proj_pred - proj_gt, axis=1).mean()
        return e

    def calc_add(self, pts, pose_pred, pose_gt):
        """
        Average Distance of Model Points for objects with no indistinguishable views
        - by Hinterstoisser et al. (ACCV 2012).
        http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

        :param pts: nx3 ndarray with 3D model points.
        :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
        :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
        :return: Mean average error between the predicted and ground truth pose.
        """
        rot_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:, 3]

        rot_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:, 3]

        pts_est = self.transform_pts(pts, rot_pred, t_pred)
        pts_gt = self.transform_pts(pts, rot_gt, t_gt)
        error = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
        return error

    def calc_adi(self, pts, pose_pred, pose_gt):
        """
        Average Distance of Model Points for objects with indistinguishable views
        - by Hinterstoisser et al. (ACCV 2012).
        http://www.stefan-hinterstoisser.com/papers/hinterstoisser2012accv.pdf

        :param pts: nx3 ndarray with 3D model points.
        :param pose_pred: Estimated pose (3x3 rot. matrix and 3x1 translation vector).
        :param pose_gt: GT pose (3x3 rot. matrix and 3x1 translation vector).
        :return: Mean average error between the predicted and ground truth pose reduced by symmetry.
        """
        rot_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:, 3]

        rot_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:, 3]

        pts_pred = self.transform_pts(pts, rot_pred, t_pred)
        pts_gt = self.transform_pts(pts, rot_gt, t_gt)

        # Calculate distances to the nearest neighbors from pts_gt to pts_est
        nn_index = spatial.cKDTree(pts_pred)
        nn_dists, _ = nn_index.query(pts_gt, k=1)

        error = nn_dists.mean()
        return error

    def calc_rotation_error(self, rot_pred, r_gt):
        """
        Calculate the angular geodesic rotation error between a predicted rotation matrix and the ground truth matrix.
        :paran rot_pred: Predicted rotation matrix (3x3)
        :param rot_gt: Ground truth rotation matrix (3x3)
        """
        assert (rot_pred.shape == r_gt.shape == (3, 3))
        temp = logm(np.dot(np.transpose(rot_pred), r_gt))
        rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
        rd_deg = rd_rad / np.pi * 180
        return rd_deg

# Functions to initialize the PoseEvaluator module
def load_classes(path):
    """
    Load the class information from a json file. This file contains a mapping between class ID and class name.
    """
    with open(path, 'r') as f:
        classes = json.load(f)
    return classes


def load_model_info(points):
    """
    Load information about the 3D model from the BOP files
    """
    infos = {}
    extents = 2 * np.max(np.absolute(points), axis=0)
    infos['diameter'] = np.sqrt(np.sum(extents * extents))
    infos['min_x'], infos['min_y'], infos['min_z'] = np.min(points, axis=0)
    infos['max_x'], infos['max_y'], infos['max_z'] = np.min(points, axis=0)
    return infos


def load_ply(path):
    """Loads a 3D mesh model from a PLY file.
    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
     - 'pts' (nx3 ndarray)
     - 'normals' (nx3 ndarray), optional
     - 'colors' (nx3 ndarray), optional
     - 'faces' (mx3 ndarray), optional
     - 'texture_uv' (nx2 ndarray), optional
     - 'texture_uv_face' (mx6 ndarray), optional
     - 'texture_file' (string), optional
    """
    f = open(path, 'rb')

    # Only triangular faces are supported.
    face_n_corners = 3

    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:

        # Strip the newline character(s).
        line = f.readline().decode('utf8').rstrip('\n').rstrip('\r')

        if line.startswith('comment TextureFile'):
            texture_file = line.split()[-1]
        elif line.startswith('element vertex'):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
            header_face_section = True
        elif line.startswith('element'):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith('property') and header_vertex_section:
            # (name of the property, data type)
            pt_props.append((line.split()[-1], line.split()[-2]))
        elif line.startswith('property list') and header_face_section:
            elems = line.split()
            if elems[-1] == 'vertex_indices' or elems[-1] == 'vertex_index':
                # (name of the property, data type)
                face_props.append(('n_corners', elems[2]))
                for i in range(face_n_corners):
                    face_props.append(('ind_' + str(i), elems[3]))
            elif elems[-1] == 'texcoord':
                # (name of the property, data type)
                face_props.append(('texcoord', elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(('texcoord_ind_' + str(i), elems[3]))
            else:
                print('Warning: Not supported face property: ' + elems[-1])
        elif line.startswith('format'):
            if 'binary' in line:
                is_binary = True
        elif line.startswith('end_header'):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model['texture_file'] = texture_file
    model['pts'] = np.zeros((n_pts, 3), np.cfloat)
    if n_faces > 0:
        model['faces'] = np.zeros((n_faces, face_n_corners), np.cfloat)

    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]
    #print(pt_props_names)
    #input()
    is_normal = False
    is_color = False
    if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
        is_normal = True
        model['normals'] = np.zeros((n_pts, 3), np.cfloat)

    is_color = False
    if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
        is_color = True
        model['colors'] = np.zeros((n_pts, 3), np.cfloat)

    is_texture_pt = False
    if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model['texture_uv'] = np.zeros((n_pts, 2), np.cfloat)

    is_texture_face = False
    if {'texcoord'}.issubset(set(face_props_names)):
        is_texture_face = True
        model['texture_uv_face'] = np.zeros((n_faces, 6), np.cfloat)

    # Formats for the binary case.
    formats = {
        'float': ('f', 4),
        'double': ('d', 8),
        'int': ('i', 4),
        'uchar': ('B', 1)
    }

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                      'red', 'green', 'blue', 'texture_u', 'texture_v']
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode('utf8').rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model['pts'][pt_id, 0] = float(prop_vals['x'])
        model['pts'][pt_id, 1] = float(prop_vals['y'])
        model['pts'][pt_id, 2] = float(prop_vals['z'])

        if is_normal:
            model['normals'][pt_id, 0] = float(prop_vals['nx'])
            model['normals'][pt_id, 1] = float(prop_vals['ny'])
            model['normals'][pt_id, 2] = float(prop_vals['nz'])

        if is_color:
            model['colors'][pt_id, 0] = float(prop_vals['red'])
            model['colors'][pt_id, 1] = float(prop_vals['green'])
            model['colors'][pt_id, 2] = float(prop_vals['blue'])

        if is_texture_pt:
            model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
            model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == 'n_corners':
                    if val != face_n_corners:
                        raise ValueError('Only triangular faces are supported.')
                elif prop[0] == 'texcoord':
                    if val != face_n_corners * 2:
                        raise ValueError('Wrong number of UV face coordinates.')
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().decode('utf8').rstrip('\n').rstrip('\r').split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == 'n_corners':
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError('Only triangular faces are supported.')
                elif prop[0] == 'texcoord':
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError('Wrong number of UV face coordinates.')
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model['faces'][face_id, 0] = int(prop_vals['ind_0'])
        model['faces'][face_id, 1] = int(prop_vals['ind_1'])
        model['faces'][face_id, 2] = int(prop_vals['ind_2'])

        if is_texture_face:
            for i in range(6):
                model['texture_uv_face'][face_id, i] = float(
                    prop_vals['texcoord_ind_{}'.format(i)])

    f.close()

    return model


def load_models(path, classes):
    """
    Load the 3D model point cloud and store it in a dict.
    """

    with open(os.path.join(path, 'models_info.json'), 'r') as f:
        models_info_data = json.load(f)

    #print('models_info_data', models_info_data)
    models = {}
    models_info = {}

    for cls in classes:
        model_class = classes[cls]
        #print(cls, model_class)
        model_file = "obj_" + f'{int(cls):06d}' + ".ply"
        model = load_ply(os.path.join(path, model_file))
        models[model_class] = model
        models[model_class]['pts'] = models[model_class]['pts'] / 1000  # Scale the model to meters.
        models_info[model_class] = models_info_data[cls]
    return models, models_info


def load_model_symmetry(path, classes):
    """
    Load information whether objects are symmetric or not.
    """
    model_symmetry = {}

    with open(path, 'r') as f:
        symmetry_dict = json.load(f)

    for cls in classes:
        model_cls = classes[cls]
        model_symmetry[model_cls] = symmetry_dict[model_cls]

    return model_symmetry


def build_pose_evaluator(args, dataset):
    """
    Function to build the Pose Evaluator by loading the 3D point clouds and additional information.
    """
    classes = {str(int(key)): v['name'] for key, v in dataset.names.items()}

    models_path = os.path.join(dataset.root, dataset.models_dir, 'models_eval')
    models, models_info = load_models(models_path, classes)

    symmetries_path = os.path.join(dataset.root, dataset.symmetries_file)
    model_symmetry = load_model_symmetry(symmetries_path, classes)
    classes = [classes[k] for k in classes]
    
    if args.ds_name in ['YCB_video', 'LM-O', 'LineMod', 'Graspnet', 'homebrew']:
        evaluator = PoseEvaluator(models, classes, models_info, model_symmetry)
    #elif args.ds_name == 'LM-O':
    #    evaluator = PoseEvaluatorLMO(models, classes, models_info, model_symmetry)
    else:
        raise ValueError("Unknown dataset.")
    
    # evaluator = PoseEvaluator(models, classes, models_info, model_symmetry)

    return evaluator