import sys
import numpy as np
from sklearn.metrics import auc
import torchvision.transforms as transforms
import random
from PIL import ImageFilter, ImageOps
import torch


def bar_progress(mode, epoch, epochs, step, steps, valid_step, valid_steps, lr, step_logs, previous_logs, loss, eta_logs,
                 vanishing_depth_threshold=None):
    progress = (epoch-1) * steps + (epoch-1) * valid_steps
    if mode == 'train':
        progress += step
    else:
        progress += steps + valid_step
    progress = progress / (epochs * steps + epochs * valid_steps)
    loss_delta = None
    me_delta = None
    if vanishing_depth_threshold is not None:
        vanishing_depth_threshold = float(np.round(vanishing_depth_threshold, 3))
    mean_loss = np.mean(step_logs['loss'])
    mean_me = np.mean(step_logs['metrics']['layer_0']['dist'])
    if previous_logs is not None:
        loss_delta = float(np.round(mean_loss - previous_logs['loss'], 4))
        me_delta = float(np.round(mean_me - previous_logs['metrics']['layer_0']['dist'], 4))

    if mode == 'train':
        a = step
        b = steps
        c = 'train'
    else:
        a = valid_step
        b = valid_steps
        c = 'valid'

    d_epoch = epochs - epoch
    if mode == 'train':
        d_steps = steps - step
    else:
        d_steps = 0
    d_valid_steps = valid_steps - valid_step
    m_t = float(np.mean(eta_logs['train']))
    eta = d_steps * m_t
    if len(eta_logs['valid']) > 0:
        m_v = float(np.mean(eta_logs['valid']))
    else:
        m_v = float(np.mean(eta_logs['train']))
    eta += d_valid_steps * m_v

    if len(eta_logs['epoch']) > 0:
        eta += d_epoch * float(np.mean(eta_logs['epoch']))
    else:
        eta += d_epoch * ((steps * m_t) + (valid_steps * m_v))

    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))

    if lr is not None:
        lr = float(np.round(lr, 10))

    progress_message = "eta: {} | {}% [{}/{}] | {}% [{}/{}] | t: {} | lr: {} | \u03BCl: {} | l: {} |" \
                       " \u0394l: {} | \u03BCe: {} | \u0394\u03BCe: {}".format(
        eta,
        float(np.round(progress * 100, 4)),
        epoch,
        epochs,
        float(np.round(a / b * 100, 2)),
        a,
        b,
        #c,
        vanishing_depth_threshold,
        lr,
        float(np.round(mean_loss, 4)),
        float(np.round(loss, 4)),
        loss_delta,
        float(np.round(mean_me, 4)),
        me_delta)

    dino_loss = step_logs.get('dino_loss', [])

    if len(dino_loss) > 0:
        dino_loss = float(np.round(np.mean(dino_loss), 5))
        dino_massage = ' | dino \u03BCl: {}'.format(dino_loss)
        dino_p = step_logs.get('dino_p', [])
        if len(dino_p) > 0:
            dino_p = int(np.round(np.mean(dino_p)*100, 3))
            dino_massage += ' ({}%)'.format(dino_p)
        progress_message += dino_massage


    rdps = step_logs.get('rdps')
    #print('rdps', rdps)
    if isinstance(rdps, dict):
        rdps_ = [float(np.round(np.mean(v)*100, 1)) for v in rdps.values() if len(v) > 0]
        progress_message += ' | rdps: {}'.format(rdps_)

    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def bar_progress_det(mode, epoch, epochs, step, steps, valid_step, valid_steps, lr, losses, previous_loss, eta_logs):

    progress = (epoch-1) * steps + (epoch-1) * valid_steps
    if mode == 'train':
        progress += step
    else:
        progress += steps + valid_step
    progress = progress / (epochs * steps + epochs * valid_steps)
    loss_delta = None
    mean_loss = np.mean(losses)
    if previous_loss is not None:
        loss_delta = float(np.round(mean_loss - previous_loss, 5))

    if mode == 'train':
        a = step
        b = steps
        c = 'train step'
    else:
        a = valid_step
        b = valid_steps
        c = 'valid step'

    d_epoch = epochs - epoch
    if mode == 'train':
        d_steps = steps - step
    else:
        d_steps = 0
    d_valid_steps = valid_steps - valid_step
    m_t = float(np.mean(eta_logs['train']))
    eta = d_steps * m_t
    if len(eta_logs['valid']) > 0:
        m_v = float(np.mean(eta_logs['valid']))
    else:
        m_v = float(np.mean(eta_logs['train']))
    eta += d_valid_steps * m_v

    if len(eta_logs['epoch']) > 0:
        eta += d_epoch * float(np.mean(eta_logs['epoch']))
    else:
        eta += d_epoch * ((steps * m_t) + (valid_steps * m_v))

    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))

    if lr is not None:
        lr = float(np.round(lr, 10))

    progress_message = "eta: {} | Progress {}% [{}/{}] epochs | {}% [{}/{}] {}| lr: {} | m loss: {} | loss {} | " \
                       "loss d: {}".format(
        eta,
        float(np.round(progress * 100, 4)),
        epoch,
        epochs,
        float(np.round(a / b * 100, 2)),
        a,
        b,
        c,
        lr,
        float(np.round(mean_loss, 4)),
        float(np.round(losses[-1], 4)),
        loss_delta)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def bar_progress_test(ds_name, n_ds, num_ds, ds_step, ds_steps, total_step, total_steps, step_time, logs):
    step_time = float(np.mean(step_time))
    eta = (total_steps-total_step) * step_time

    auc = float(np.round(np.mean(logs['auc']['metrics']['layer_0']['dist']), 5))
    loss = float(np.round(np.mean(logs['auc']['loss']), 5))

    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))
    progress = float(np.round((total_step / total_steps) * 100, 4))

    progress_message = '{} | Progress {}% [{}/{}] test step | {}/{} DS name: {} | DS step {}/{} | ' \
                       'loss: {} | AUC: {}'.format(
        eta, progress, total_step, total_steps, n_ds, num_ds, ds_name, ds_step, ds_steps, loss, auc)

    dino_p = logs.get('dino_p', [])
    if len(dino_p) > 0:
        dino_p = int(np.round(np.mean(dino_p)*100, 2))
        progress_message += ' | \u03BCdino_p: {}'.format('{}%'.format(dino_p))

    dino_loss = logs.get('dino_loss', [])
    if len(dino_loss) > 0:
        dino_loss = float(np.round(np.mean(dino_loss), 5))
        progress_message += ' | dino \u03BCloss: {}'.format(dino_loss)

    rdps = logs.get('rdps')
    #print('rdps', rdps)
    if isinstance(rdps, dict):
        rdps_ = [float(np.round(np.mean(v)*100, 1)) for v in rdps.values() if len(v) > 0]
        progress_message += ' | rdps: {}'.format(rdps_)

    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def bar_progress_test2(step, steps, step_time, loss, metric, rps=None, dps=None, metric_name='Metric'):
    progress =  float(np.round((step / steps) * 100, 3))
    if loss is not None:
        loss = float(np.round(loss, 5))
    if metric is not None:
        metric = float(np.round(metric, 5))
    if isinstance(step_time, np.ndarray):
        step_time = float(np.mean(step_time))
    eta = (steps - step) * step_time
    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))

    if rps is not None:
        if len(rps) > 0:
            rps = float(np.mean(rps))
        else:
            rps = None

    if dps is not None:
        if len(dps) > 0:
            dps = float(np.mean(dps))
        else:
            dps = None

    if dps is not None and rps is not None:
        dps = float(np.round(dps / (rps + dps), 4))
        rps = float(np.round(rps / (rps + dps), 4))

    progress_message = '{} | Progress {}% [{}/{}] test steps'.format(eta, progress, step, steps)
    if loss is not None:
        progress_message += '| Loss: {}'.format(loss)

    if metric is not None:
        progress_message += '| {}: {}'.format(metric_name, metric)

    if rps is not None:
        progress_message += '| rps: {}'.format(rps)

    if dps is not None:
        progress_message += '| dps: {}'.format(dps)

    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def bar_progress_seg(mode, epoch, epochs, step, steps, valid_step, valid_steps, lr, metric, previous_metric,
                     loss, previous_loss, eta_logs, rps=None, dps=None, metric_name='mIoU'):

    progress = (epoch-1) * steps + (epoch-1) * valid_steps
    if mode == 'train':
        progress += step
    else:
        progress += steps + valid_step
    progress = progress / (epochs * steps + epochs * valid_steps)
    loss_delta = None
    me_delta = None
    mean_loss = np.mean(loss)
    if previous_loss is not None:
        loss_delta = float(np.round(mean_loss - previous_loss, 3))

    if previous_metric is not None:
        me_delta = float(np.round(metric - previous_metric, 4))

    if metric is not None:
        metric = float(np.round(metric, 4))

    if mode == 'train':
        a = step
        b = steps
        c = 'train step'
    else:
        a = valid_step
        b = valid_steps
        c = 'valid step'

    d_epoch = epochs - epoch
    if mode == 'train':
        d_steps = steps - step
    else:
        d_steps = 0
    d_valid_steps = valid_steps - valid_step
    m_t = float(np.mean(eta_logs.get('train', [1])))
    eta = d_steps * m_t
    if len(eta_logs['valid']) > 0:
        m_v = float(np.mean(eta_logs.get('valid')))
    else:
        m_v = float(np.mean(eta_logs.get('train')))
    eta += d_valid_steps * m_v

    if len(eta_logs['epoch']) > 0:
        eta += d_epoch * float(np.mean(eta_logs['epoch']))
    else:
        eta += d_epoch * ((steps * m_t) + (valid_steps * m_v))
    #eta = 1
    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))

    if lr is not None:
        lr = float(np.round(lr, 10))

    if rps is not None:
        if len(rps) > 0:
            rps = float(np.mean(rps))
        else:
            rps = None

    if dps is not None:
        if len(dps) > 0:
            dps = float(np.mean(dps))
        else:
            dps = None

    if dps is not None and rps is not None:
        dps = float(np.round(dps / (rps + dps), 4))
        rps = float(np.round(rps / (rps + dps), 4))

    progress_message = "eta: {} | Progress {}% [{}/{}] epochs | {}% [{}/{}] {} | lr: {} | mean loss: {} | loss {} | " \
                       "loss delta: {} | {}: {} | {} delta: {} | rps: {} | dps: {}".format(
        eta,
        float(np.round(progress * 100, 4)),
        epoch,
        epochs,
        float(np.round(a / b * 100, 2)),
        a,
        b,
        c,
        lr,
        float(np.round(mean_loss, 4)),
        float(np.round(loss[-1], 4)),
        loss_delta,
        metric_name,
        metric,
        metric_name,
        me_delta,
        rps,
        dps)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def bar_progress_pose(mode, epoch, epochs, step, steps, valid_step, valid_steps, lr, loss, previous_loss, eta_logs):

    progress = (epoch-1) * steps + (epoch-1) * valid_steps
    if mode == 'train':
        progress += step
    else:
        progress += steps + valid_step
    progress = progress / (epochs * steps + epochs * valid_steps)
    loss_delta = None
    mean_loss = np.mean(loss)
    if previous_loss is not None:
        loss_delta = float(np.round(mean_loss - previous_loss, 3))

    if mode == 'train':
        a = step
        b = steps
        c = 'train step'
    else:
        a = valid_step
        b = valid_steps
        c = 'valid step'

    d_epoch = epochs - epoch
    if mode == 'train':
        d_steps = steps - step
    else:
        d_steps = 0
    d_valid_steps = valid_steps - valid_step
    m_t = float(np.mean(eta_logs.get('train', [1])))
    eta = d_steps * m_t
    if len(eta_logs['valid']) > 0:
        m_v = float(np.mean(eta_logs.get('valid')))
    else:
        m_v = float(np.mean(eta_logs.get('train')))
    eta += d_valid_steps * m_v

    if len(eta_logs['epoch']) > 0:
        eta += d_epoch * float(np.mean(eta_logs['epoch']))
    else:
        eta += d_epoch * ((steps * m_t) + (valid_steps * m_v))
    #eta = 1
    hours = int(eta / 3600)
    if hours > 24:
        days = int(hours / 24)
        hours -= days * 24
        eta -= (days * 24) * 3600
    else:
        days = 0
    eta -= hours * 3600
    minutes = int(eta / 60)
    eta -= minutes * 60
    sec = int(eta)

    eta = ''
    if days > 0:
        eta += str(days) + ':'
    eta += '{}:{}:{}'.format(str(hours) if len(str(hours)) > 2 else str(hours).zfill(2),
                             str(minutes).zfill(2), str(sec).zfill(2))

    if lr is not None:
        lr = float(np.round(lr, 10))


    progress_message = "eta: {} | Progress {}% [{}/{}] epochs | {}% [{}/{}] {} | lr: {} | mean loss: {} | loss {} | " \
                       "loss delta: {}".format(
        eta,
        float(np.round(progress * 100, 4)),
        epoch,
        epochs,
        float(np.round(a / b * 100, 2)),
        a,
        b,
        c,
        lr,
        float(np.round(mean_loss, 4)),
        float(np.round(loss[-1], 4)),
        loss_delta
        )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def mean_logs(logs):
    for key, v in logs.items():
        if isinstance(v, dict):
            for key2, v2 in v.items():
                if isinstance(v2, dict):
                    for key3, v3 in v2.items():
                        if isinstance(v3, dict):
                            for key4 in v3:
                                if None in logs[key][key2][key3][key4]:
                                    logs[key][key2][key3][key4] = 0
                                else:
                                    logs[key][key2][key3][key4] = float(np.nanmean(logs[key][key2][key3][key4]))
                        else:
                            if None in logs[key][key2][key3]:
                                logs[key][key2][key3] = 0
                            else:
                                logs[key][key2][key3] = float(np.nanmean(logs[key][key2][key3]))
                else:
                    if None in logs[key][key2]:
                        logs[key][key2] = None
                    else:
                        logs[key][key2] = float(np.nanmean(logs[key][key2]))
        else:
            if None in logs[key]:
                logs[key] = 0
            else:
                logs[key] = float(np.nanmean(logs[key]))
    return logs

def add_logs(logs, loss, losses, metrics):
    logs['loss'].append(loss)
    for key, v in losses.items():
        if key not in logs['losses']:
            logs['losses'][key] = []
        logs['losses'][key].append(v)

    for key, v in metrics.items():
        if key not in logs['metrics']:
            logs['metrics'][key] = {}
        for key2, v2 in v.items():
            if key2 not in logs['metrics'][key]:
                logs['metrics'][key][key2] = []
            logs['metrics'][key][key2].append(v2)
    return logs

def compute_auc(logs, thresholds):
    for key, v in logs.items():
        if isinstance(v, dict):
            for key2, v2 in v.items():
                if isinstance(v2, dict):
                    for key3 in v2:
                        if None in logs[key][key2][key3]:
                            logs[key][key2][key3] = 0
                        else:
                            logs[key][key2][key3] = float(auc(thresholds, np.array(logs[key][key2][key3])))
                else:
                    if None in logs[key][key2]:
                        logs[key][key2] = 0
                    else:
                        logs[key][key2] = float(auc(thresholds, np.array(logs[key][key2])))
        else:
            if None in logs[key]:
                logs[key] = 0
            else:
                logs[key] = float(auc(thresholds, np.array(logs[key])))
    return logs


def get_empty_ann_file() -> dict:
    ann_file = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    return ann_file


def points2pixel(points, intr):
    pixels = []
    for point in points:
        x, y, z = point
        p1 = (x / (z / intr.get('fx')))
        p1 = int(p1 + intr.get('ppx'))
        p0 = (y / (z / intr.get('fy')))
        p0 = int(p0 + intr.get('ppy'))
        pixels.append([p0, p1])

    return pixels


def pointcloud2image(image, point_cloud, point_size, intr, color=None):

    step = int((point_size - 1) / 2)

    mark = np.zeros((point_size, point_size, 3))
    if not color:
        mark[:, :, 0] = 255
    else:
        for i, c in enumerate(color):
            mark[:, :, i] = c

    if isinstance(point_cloud, np.ndarray):
        points = point_cloud
    else:
        points = np.array(point_cloud.points)
    pixels = points2pixel(points, intr)

    for x, y in pixels:
        try:
            image[x - step:x + step + 1, y - step:y + step + 1, :] = mark * 0.3 + image[x - step:x + step + 1, y - step:y + step + 1, :] * 0.7

        except Exception as e:
            #print(e)
            pass
    #pixels = np.array(pixels)
    #print('x', np.mean(pixels[:, 0]), np.std(pixels[:, 0]), np.min(pixels[:, 0]), np.max(pixels[:, 0]))
    #print('y', np.mean(pixels[:, 1]), np.std(pixels[:, 1]), np.min(pixels[:, 1]), np.max(pixels[:, 1]))

    return image

'''
def convert_coco():
        bboxes_per_sample = []

        ann_file = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        for cat_id, cls in self.class_names.items():
            ann_file['categories'].append({'id': cat_id, 'name': cls['name']})

        ann_id = 0
        skipped = 0
        bb_counts = {cls['name']: 0 for cls in self.class_names.values()}
        for index in range(self.dirs):
            sample = self.load_sample(self.dirs[index])

            for bbox, u in sample['bboxes']:
                ann_file['annotations'].append({
                    'id': ann_id,
                    'category_id': u,
                    'image_id': index,
                    'bbox': bbox,
                    'iscrowd': 0,
                    'area': area
                })
                ann_id += 1
                new_boxes += 1
                bb_counts[self.class_names[u]] += 1

            if new_boxes > 0:
                ann_file['images'].append({
                    'id': index,
                    'width': w,
                    'height': h,
                    'file_name': path
                })
                bboxes_per_sample.append(new_boxes)
        valid_name = 'ycb_v_{}_dataset_config.json'.format('valid')
        with open(os.path.join(root, valid_name), 'w') as f:
            json.dump({'dirs': self.dirs[n:], 'ann_file': ann_file}, f)
        self.ann_file = ann_file

    if with_mesh_points and self.mesh_points is None:
        mesh_root = os.path.join(root, 'ycbv_models', )

    if with_coco and self.ann_file is not None:
        self.coco = COCO()
        self.coco.dataset = self.ann_file
        self.coco.createIndex()
        total_bboxes = len(self.ann_file['annotations'])
        print('####### {} #######'.format(mode))
        print('Number of bboxes: {}, skipped bboxes: {}'.format(total_bboxes, skipped))
        for key, value in bb_counts.items():
            print('{} | {} bboxes | {}% of total bboxes'.format(key, value,
                                                                np.round((value / total_bboxes) * 100), 3))
        print(
            'mean bbox per sample: {}, std: {}, max: {}: min:{}'.format(np.round(np.mean(bboxes_per_sample), 3),
                                                                        np.round(np.std(bboxes_per_sample), 3),
                                                                        np.max(bboxes_per_sample),
                                                                        np.min(bboxes_per_sample))
'''



def lookup(n):
    """
            Args:
                n (int): number of plots
            returns:
                x, y (int): matplotlib grid x, y
            """
    if n == 1:
        x, y = 1, 1
    elif n == 2:
        x, y = 1, 2
    elif n == 3:
        x, y = 1, 3
    elif n == 4:
        x, y = 2, 2
    elif n in [5, 6]:
        x, y = 2, 3
    elif n in [7, 8]:
        x, y = 2, 4
    elif n == 9:
        x, y = 3, 3
    elif n in [10, 11, 12]:
        x, y = 3, 4
    elif n in [13, 14, 15]:
        x, y = 3, 5
    elif n == 16:
        x, y = 4, 4
    elif n in [17, 18, 19, 20]:
        x, y = 4, 5
    elif n in [21, 22, 23, 24, 25]:
        x, y = 5, 5
    elif n in [26, 27, 28, 29, 30]:
        x, y = 5, 6
    elif n in [31, 32, 33, 34, 35, 36]:
        x, y = 6, 6
    elif n in [37, 38, 39, 40, 41, 42]:
        x, y = 6, 7
    elif n in [43, 44, 45, 46, 47, 48, 49]:
        x, y = 7, 7
    elif n in [50, 51, 52, 53, 54, 55, 56]:
        x, y = 7, 8
    elif n in [57, 58, 59, 60, 61, 62, 63, 64]:
        x, y = 8, 8
    elif n in [65, 66, 67, 68, 69, 70, 71, 72]:
        x, y = 8, 9
    else:
        raise NotImplementedError
    return int(x), int(y)


def load_fitting_state_dict(arch, state_dict, prefix=None):
    keys = list(arch.state_dict().keys())
    keys_left = list(arch.state_dict().keys())
    wrong_shape_keys = []
    #print('model keys', keys)
    wrong_key = 0
    wrong_shape = 0
    okay = 0
    #print('sd keys', state_dict.keys())

    # if prefix:
    #     state_dict ={k[len(prefix)+1: ]: v for k,v in state_dict.items() if k.startswith(prefix)} 

    for key, v in state_dict.items():
        if key not in keys:
            wrong_key += 1
            print('wrong key', key, v.shape)
            continue
        sd = {key: v}
        del keys_left[keys_left.index(key)]
        try:
            arch.load_state_dict(sd, strict=False)
        except Exception as e:
            if '.pos_embed' in key:
                pos_embed_checkpoint = state_dict[key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                try:
                    num_patches = arch.encoder.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.encoder.color_encoder.pos_embed.shape[-2] - num_patches
                except:
                    num_patches = arch.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches

                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Pos Embed {} interpolate from {}x{} to {}x{}".format(key, orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    sd = {key: new_pos_embed}
                    arch.load_state_dict(sd, strict=False)
                okay += 1


            #print(state_dict.keys())
            elif 'rope' in key:              
            
                rope_embed_checkpoint = state_dict[key]
                embedding_size = rope_embed_checkpoint.shape[-1]
                try:
                    num_patches = arch.encoder.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.encoder.color_encoder.pos_embed.shape[-2] - num_patches
                except:
                    num_patches = arch.color_encoder.patch_embed.num_patches
                    num_extra_tokens = arch.color_encoder.pos_embed.shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged

                
                #print('key', key, orig_size, new_size, rope_embed_checkpoint.shape)
                if orig_size != new_size:
                    print("Rope {} interpolate from {}x{} to {}x{}".format(key, orig_size, orig_size, new_size, new_size))
                    
                    # only the position tokens are interpolated
                    rope_tokens = rope_embed_checkpoint.reshape(orig_size, orig_size, embedding_size).unsqueeze(0).permute(0, 3, 1, 2).float()   #.view((-1, rope_embed_checkpoint.shape[-1]))
                    
                    #pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).float()
                    rope_tokens = torch.nn.functional.interpolate(
                        rope_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    
                    rope_tokens = rope_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
                    
                    #new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    #state_dict[key] = rope_tokens
                    sd = {key: rope_tokens}
                    arch.load_state_dict(sd, strict=False)
                okay += 1

            else: 
                print('wrong shape', e)
                wrong_shape += 1
                wrong_shape_keys.append(keys)
            
            #input()
            continue
        okay += 1

    msg = 'Loaded {}/{} weights, wrong key: {}, wrong shape: {}, missing: {}'.format(
        okay, len(keys), wrong_key, wrong_shape, len(keys) - (okay+wrong_key+wrong_shape))
    print(msg)
    #print('keys_left', keys_left)
    #print('wrong_shape_keys', wrong_shape_keys)
    return arch



class gray_scale(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img



class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
        return img

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class ToDict():
    def __call__(self, x):
        return {'x': x}

def get_imagenet_transforms(pretrain=False, size=224):

    tfs = [
        transforms.RandomResizedCrop(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        #transforms.RandomVerticalFlip(),

    ]

    if pretrain:
        print('adding 3-Aug to train transforms')
        tfs += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomChoice([gray_scale(p=1.0), Solarization(p=1.0), GaussianBlur(p=1.0)])
        ]

    tfs += [transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToDict()]
    tfs = transforms.Compose(tfs)
    return tfs


def get_imagenet_eval_transforms(resize=256, cropsize=224):
    tfs = transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToDict()
    ])
    return tfs



class DepthMetric(object):
    def __init__(self, min_depth=0.01, max_depth=150):
        self.errors = []
        self.min_depth = min_depth
        self.max_depth = max_depth

    def compute_errors(self, gt, pred):
        """Computation of error metrics between predicted and ground truth depths
        """
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = np.nanmean(thresh < 1.25)
        a2 = np.nanmean(thresh < 1.25 ** 2)
        a3 = np.nanmean(thresh < 1.25 ** 3)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(np.nanmean(rmse))

        diff_log = np.log(pred) - np.log(gt)
        diff_log100 = np.log(pred*100) - np.log(gt*100)
        rmse_log = np.sqrt(np.nanmean(diff_log ** 2))

        abs_rel = np.nanmean(np.abs(gt - pred) / gt)
        abs_err = np.nanmean(np.abs(gt - pred))
        
        sq_rel = np.nanmean(((gt - pred) ** 2) / gt)

        silog = np.nanmean(np.sqrt(np.power(diff_log, 2)) - 0.5 * np.power(np.nanmean(diff_log), 2))
        silog100 = np.nanmean(np.sqrt(np.power(diff_log100, 2)) - 0.5 * np.power(np.nanmean(diff_log100), 2))

        abs_log_err = np.nanmean(np.abs(diff_log))

        return [float(abs_rel), float(sq_rel), float(rmse), float(rmse_log),  float(a1), float(a2), float(a3), float(silog), float(abs_log_err), float(abs_err), float(silog100)]

    def add(self, gt, pred, ori_depth=None):

        #if ori_depth is not None:
        #    pred[ori_depth > self.min_depth] = ori_depth[ori_depth>self.min_depth] 

        #gt[gt < self.min_depth] = self.min_depth
        gt[gt > self.max_depth] = self.max_depth
        pred[pred < self.min_depth] = self.min_depth
        pred[pred > self.max_depth] = self.max_depth
       
        '''
        if ori_depth is not None:        
            gt_zeros = np.round(100 * (1 - (np.sum(gt > self.min_depth) / np.prod(gt.shape))), 4)
            pred_zeros = np.round(100 * (1 - (np.sum(pred > self.min_depth) / np.prod(pred.shape))), 4)
        '''
        
        
        #pred[gt == self.min_depth] = 0
        pred = pred[gt > self.min_depth]
        gt = gt[gt > self.min_depth]
        
        '''
        if ori_depth is not None :        
            print('')
            print('gt max: {}, mean: {}, std: {}, zeros: {}, shape: {}'.format(np.max(gt), np.mean(gt), np.std(gt), gt_zeros, gt.shape))
            print('pred max: {}, mean: {}, std: {}, zeros: {}, shape: {}'.format(np.max(pred), np.mean(pred), np.std(pred), pred_zeros, pred.shape))

            ori_depth_zeros = np.round(100 * (1 - (np.sum(ori_depth > self.min_depth) / np.prod(ori_depth.shape))), 4)
            ori_depth = ori_depth[ori_depth > self.min_depth]
            print('ori depth max: {}, min: {}, mean: {}, std: {}, zeros: {}, shape: {}'.format(np.max(ori_depth), np.min(ori_depth), np.mean(ori_depth), np.std(ori_depth), ori_depth_zeros, ori_depth.shape))
        
            print('##############################################################################################')
        '''

        me = self.compute_errors(gt, pred)
        self.errors.append(me)
        results = {'abs_rel': float(me[0]), 'sq_rel': float(me[1]), 'rmse': float(me[2]), 'rmse_log': float(me[3]),
                   'a1': float(me[4]), 'a2': float(me[5]), 'a3': float(me[6]), 'silog': float(me[7]), 'abs_log_err': float(me[8]), 'abs_err': float(me[9]), 'silog100': float(me[10])}
        return results

    def reset(self):
        self.errors = []

    def result(self):
        me = np.nanmean(np.array(self.errors), axis=0)
        results = {'abs_rel': float(me[0]), 'sq_rel': float(me[1]), 'rmse': float(me[2]), 'rmse_log': float(me[3]),
                   'a1': float(me[4]), 'a2': float(me[5]), 'a3': float(me[6]), 'silog': float(me[7]), 'abs_log_err': float(me[8]), 'abs_err': float(me[9]), 'silog100': float(me[10])}
        return results

