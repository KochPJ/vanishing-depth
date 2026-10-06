import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class VanishingDepthLoss(nn.Module):
    def __init__(
            self, normalize=False, eps=1e-6, fpn=False, loss_with_not_mask=False,
            with_point_cloud=False, lw=None, t1=None, t2=None, t3=None, max_pow=0.9, pow=0.5, max_epoch=100,
            start_epoch=20, use_pow=False, current_epoch=1, a=10, l=0.85, final_l=0.85,
            with_dino_head=False, dino_head_out_dims=65536, epochs=100, dino_loss_w=1.0, eval_loss=False,
            sqrt_errors=False
    ):
        super().__init__()
        self.normalize = normalize
        self.eps = eps
        self.fpn = fpn
        self.a = a
        self.l = l
        self.final_l = final_l
        self.loss_with_not_mask = loss_with_not_mask
        self.lw = lw
        self.min_depth = 1e-3
        self.with_point_cloud = with_point_cloud
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.max_pow = max_pow
        self.pow = pow
        self.max_epoch = max_epoch
        self.start_epoch = start_epoch
        self.use_pow = use_pow
        self.current_epoch = current_epoch
        self.with_dino_head = with_dino_head
        self.dino_head_out_dims = dino_head_out_dims
        self.dino_loss = None
        self.epochs = epochs
        self.dino_loss_w = dino_loss_w
        self.sqrt_errors = sqrt_errors
        if self.with_dino_head:
            self.dino_loss = DINOLoss(epochs, warmup_teacher_temp_epochs=start_epoch, eval_loss=eval_loss)

        steps = self.max_epoch - self.start_epoch
        self.step_size = (max_pow - pow) / steps
        self.l_step_size = (final_l - l) / steps
        #print(self.l_step_size)
        if self.current_epoch > self.start_epoch:
            self.pow += self.step_size * (self.current_epoch - self.start_epoch)
            self.l += self.l_step_size * (self.current_epoch - self.start_epoch)
            #print(self.l)
        if self.pow > self.max_pow:
            self.pow = self.max_pow
        if self.l < self.final_l:
            self.l = self.final_l

    def step_epoch(self):
        self.current_epoch += 1
        if self.current_epoch > self.start_epoch:
            new_pow = float(np.round(self.pow + self.step_size, 5))
            if new_pow <= self.max_pow:
                self.pow = new_pow
            else:
                self.pow = self.max_pow
            newl = float(np.round(self.l + self.l_step_size, 5))
            if newl >= self.final_l:
                self.l = newl
            else:
                self.l = self.final_l

    def forward(self, prediction, gt_depth, gt_mask, epoch=0, depth_mul=None):
        if depth_mul is not None:
            depth_mul = depth_mul.unsqueeze(-1).unsqueeze(-1)
        losses = {}
        metric = {}
        loss = []
        if isinstance(prediction, dict):
            if 'student_pred' in prediction and 'teacher_pred' in prediction:
                dino_l = self.dino_loss(prediction['student_pred'], prediction['teacher_pred'], epoch) * self.dino_loss_w
                if not torch.isnan(dino_l):
                    losses['dino_loss'] = float(dino_l.detach().item())
                    loss.append(dino_l)
                else:
                    losses['dino_loss'] = 0

            prediction = prediction['maps']

        if self.fpn:
            for i, (p, d, m) in enumerate(zip(prediction[::-1], gt_depth, gt_mask)):
                #print('loss at layer {}'.format(i))
                #print(i, p.flatten(1).mean(dim=1), d.flatten(1).mean(dim=1))
                #print(p.shape, d.shape, m.shape)
                #print(torch.mean(p), torch.mean(d), torch.sum(m))
                #print('mask')
                l0, me = self.lossv2(p.clone(), d.clone(), m.clone(), with_metric=True, depth_mul=depth_mul)
                #print(torch.mean(p), torch.mean(d), torch.sum(m))
                #print(l0, me)
                if self.loss_with_not_mask:
                    #print('not mask')
                    l1, me_not = self.lossv2(p, d, ~m, with_metric=False, depth_mul=None)
                else:
                    l1, me_not = None, None
                #print(l1)

                if l0 is not None and l1 is not None:
                    
                    if torch.isnan(l0):
                        l = l1
                        losses['l_{}_not'.format(i)] = float(l1.detach().item())
                    elif torch.isnan(l1):
                        l = l0
                        losses['l_{}_mask'.format(i)] = float(l0.detach().item())
                    else:
                        l = sum([l0, l1]) / 2
                        losses['l_{}_mask'.format(i)] = float(l0.detach().item())
                        losses['l_{}_not'.format(i)] = float(l1.detach().item())
                    #print('l0, l1', losses['l_{}_mask'.format(i)], losses['l_{}_not'.format(i)])
                elif l0 is not None:
                    l = l0
                elif l1 is not None:
                    l = l1
                else:
                    return None, None, None
                    # raise ValueError('loss is None, probably because no gt depth')

                losses['layer_{}'.format(i)] = float(l.detach().item())
                metric['layer_{}'.format(i)] = me

                if me_not is not None:
                    metric['layer_{}_not'.format(i)] = me_not
                    #print('me_not', me_not)

                if self.lw is not None:
                    l = l * self.lw[i]
                loss.append(l)
        else:
            #print(prediction.shape, gt_depth.shape, gt_mask.shape)
            #print(torch.mean(gt_depth), torch.sum(gt_mask), torch.sum(~gt_mask))
            l0, me = self.lossv2(prediction.clone(), gt_depth.clone(), gt_mask.clone(), with_metric=True,
                                 depth_mul=depth_mul)
            #print(torch.mean(gt_depth), torch.sum(gt_mask), torch.sum(~gt_mask))
            #print(l0, me)
            if self.loss_with_not_mask:
                l1, _ = self.lossv2(prediction, gt_depth, ~gt_mask, with_metric=False, depth_mul=None)
            else:
                l1 = None
            #print(l1)
            if l0 is not None and l1 is not None:
                l = sum([l0, l1]) / 2
                #print(l0, l1, loss)
            elif l0 is not None:
                l = l0
            elif l1 is not None:
                l = l1
            else:
                return None, None, None
                #raise ValueError('loss is None, probably because no gt depth')
            losses['layer_0'] = float(l.detach().item())
            metric['layer_0'] = me
            if self.lw is not None:
                l = self.lw[0] * l
            loss.append(l)

        loss = sum(loss) / len(loss)

        #for key, v in losses.items():
        #    print(key, v)
        #input()

        return loss, losses, metric

    def compute(self, p, d, m):
        m[d == 0] = False
        p = torch.relu(p).squeeze(1)
        metric = float(torch.mean(torch.abs(d[d != 0] - p[d != 0])).detach().item())
        if self.normalize:
            n = torch.stack((torch.max(p.detach().view(len(p), -1), dim=1).values,
                             torch.max(d.detach().view(len(d), -1), dim=1).values)).max(0).values + self.eps
            p = p.permute(1, 2, 0).div(n).permute(2, 0, 1)
            d = d.permute(1, 2, 0).div(n).permute(2, 0, 1)
        loss = torch.mean(torch.sqrt(torch.abs(d[m] - p[m])))
        return loss, metric

    def lossv2(self, p, d, m, with_metric=True, depth_mul=None):
        p = torch.relu(p)
        #print(p.shape)
        if not self.with_point_cloud:
            p = p.squeeze(1)

        #print(p.shape)
        #p = p.squeeze(1)
        if with_metric:
            p_metric = p.detach()
            d_metric = d.clone()
            if depth_mul is not None:
                if self.with_point_cloud:
                    bs, c, _, _ = p_metric.shape
                    depth_mul = depth_mul.unsqueeze(-1).repeat(1, c, 1, 1)
                p_metric = p_metric.mul(depth_mul)
                d_metric = d_metric.mul(depth_mul)

        if self.normalize:
            n = torch.stack((torch.max(p.detach().view(len(p), -1), dim=1).values,
                             torch.max(d.detach().view(len(d), -1), dim=1).values)).max(0).values + self.eps
            p = p.permute(1, 2, 0).div(n).permute(2, 0, 1)
            d = d.permute(1, 2, 0).div(n).permute(2, 0, 1)

        if not self.with_point_cloud:
            m[d < self.min_depth] = False
            m[p < self.min_depth] = False

            #print('shape', m.shape, d.shape, p.shape)
            #print('depth', torch.mean(d[m]), torch.std(d[m]), torch.min(d[m]), torch.max(d[m]))
            #print('pred', torch.mean(p[m]), torch.std(p[m]), torch.min(p[m]), torch.max(p[m]))

            if self.use_pow:
                m_ = m == True
                m_[d < 1] = False  # all distances which are above 1m
                g_pow = torch.pow(p[m_], self.pow) - torch.pow(d[m_], self.pow)
                m_ = m == True
                m_[d > 1] = False  # all distances which are below 1m
                g_log = torch.pow(p[m_], 0.5) - torch.pow(d[m_], 0.5)
                #print(type(g_pow), type(g_log))
                #print(g_pow.shape, g_log.shape)
                g = torch.cat((g_pow, g_log))
                #print(g.shape)
            else:
                g = torch.log(p[m]) - torch.log(d[m])

            t = len(g)
            if t > 0:
                loss = self.a * torch.sqrt((1 / t) * torch.sum(g.pow(2)) - (self.l / (t * t)) * torch.sum(g).pow(2))
            else:
                loss = None
        else:
            if m.shape != d.shape:
                m = m.unsqueeze(1).repeat(1, 3, 1, 1)
            #print(m.shape, d.shape, p.shape)
            m[d < self.min_depth] = False
            m[p < self.min_depth] = False
            losses = []
            for i in range(3):
                if self.use_pow:
                    g = torch.pow(p[:, i][m[:, i]], self.pow) - torch.pow(d[:, i][m[:, i]], self.pow)
                else:
                    g = torch.log(p[:, i][m[:, i]]) - torch.log(d[:, i][m[:, i]])

                if self.sqrt_errors:
                    g[g > 0] = g[g > 0].sqrt()
                    g[g < 0] = - g[g < 0].abs().sqrt()

                t = len(g)
                if t > 0:
                    loss = self.a * torch.sqrt((1 / t) * torch.sum(g.pow(2)) - (self.l / (t * t)) * torch.sum(g).pow(2))
                else:
                    loss = None

                #print(i, loss, torch.sum(g), t, torch.any(d[:, i][m[:, i]] < self.min_depth),
                #      torch.any(p[:, i][m[:, i]] < self.min_depth ))
                if loss is not None:
                    losses.append(loss)

            if len(losses) > 0:
                loss = sum(losses) / len(losses)
            else:
                loss = None

        if with_metric:
            d = d_metric
            p = p_metric
            mask = d >= self.min_depth
            mask[p < self.min_depth] = False
            if self.use_pow:
                g = torch.pow(d[mask], self.pow) - torch.pow(p.detach()[mask], self.pow)
            else:
                g = torch.log(d[mask]) - torch.log(p.detach()[mask])

            if self.sqrt_errors:
                g[g > 0] = g[g > 0].sqrt()
                g[g < 0] = - g[g < 0].abs().sqrt()

            t = len(g)

            metric = {
                'err': None,
                'dist': None,
                #'rmse': None,
                '|e|<t1': None,
                '|e|>t1': None,
                '|e|<t2': None,
                '|e|>t2': None,
                '|e|>t3': None,
                't2>|e|>t1': None,
                #'rmse<t1': None,
                #'rmse>t1': None,
                #'rmse<t2': None,
                #'rmse>t2': None,
                #'rmse>t3': None,
                #'t2>rmse>t1': None
            }
            if t > 0:
                metric['err'] = float(((1/t) * torch.sum(g.pow(2))) - ((1 / (t*t)) * torch.sum(g).pow(2)).item())
                if self.with_point_cloud:
                    metric['dist'] = float(torch.pow(d - p.detach(), 2).sum(1).sqrt()[mask.any(1)].mean().item())
                    #metric['rmse'] = float(torch.pow(d - p.detach(), 2).sum(1)[mask.any(1)].mean().sqrt().item())
                else:
                    metric['dist'] = float(torch.abs(d[mask] - p.detach()[mask]).mean().item())
                    #metric['rmse'] = float(torch.pow(d[mask] - p.detach()[mask], 2).mean().sqrt().item())

                if self.t1 is not None:
                    m_ = mask == True
                    m_[d < self.t1] = False  # disable everything below the t1 threshold
                    if self.with_point_cloud:
                        metric['|e|>t1'] = float(torch.pow(d - p.detach(), 2).sum(1).sqrt()[m_.any(1)].mean().item())
                        #metric['rmse>t1'] = float(torch.pow(d - p.detach(), 2).sum(1)[m_.any(1)].mean().sqrt().item())
                    else:
                        metric['|e|>t1'] = float(torch.abs(d[m_] - p.detach()[m_]).mean().item())
                        #metric['rmse>t1'] = float(torch.pow(d[m_] - p.detach()[m_], 2).mean().sqrt().item())

                    m_ = mask == True
                    m_[d > self.t1] = False  # disable everything above the t1 threshold
                    if self.with_point_cloud:
                        metric['|e|<t1'] = float(torch.pow(d - p.detach(), 2).sum(1).sqrt()[m_.any(1)].mean().item())
                        #metric['rmse<t1'] = float(torch.pow(d - p.detach(), 2).sum(1)[m_.any(1)].mean().sqrt().item())
                    else:
                        metric['|e|<t1'] = float(torch.abs(d[m_] - p.detach()[m_]).mean().item())
                        #metric['rmse<t1'] = float(torch.pow(d[m_] - p.detach()[m_], 2).mean().sqrt().item())

                if self.t2 is not None:
                    m_ = mask == True
                    m_[d < self.t2] = False  # disable everything below the t2 threshold
                    if self.with_point_cloud:
                        metric['|e|>t2'] = float(torch.pow(d - p.detach(), 2).sum(1).sqrt()[m_.any(1)].mean().item())
                        #metric['rmse>t2'] = float(torch.pow(d - p.detach(), 2).sum(1)[m_.any(1)].mean().sqrt().item())
                    else:
                        metric['|e|>t2'] = float(torch.abs(d[m_] - p.detach()[m_]).mean().item())
                        #metric['rmse>t2'] = float(torch.pow(d[m_] - p.detach()[m_], 2).mean().sqrt().item())

                    m_ = mask == True
                    m_[d > self.t2] = False  # disable everything above the t2 threshold
                    if self.with_point_cloud:
                        metric['|e|<t2'] = float(torch.pow(d - p.detach(), 2).sum(1).sqrt()[m_.any(1)].mean().item())
                        #metric['rmse<t2'] = float(torch.pow(d - p.detach(), 2).sum(1)[m_.any(1)].mean().sqrt().item())
                    else:
                        metric['|e|<t2'] = float(torch.abs(d[m_] - p.detach()[m_]).mean().item())
                        #metric['rmse<t2'] = float(torch.pow(d[m_] - p.detach()[m_], 2).mean().sqrt().item())

                if self.t1 is not None and self.t2 is not None:
                    m_ = mask == True
                    m_[d < self.t1] = False  # disable everything below the t1 threshold
                    m_[d > self.t2] = False  # disable everything above the t2 threshold
                    if self.with_point_cloud:
                        metric['t2>|e|>t1'] = float(torch.pow(d - p.detach(), 2).sum(1).sqrt()[m_.any(1)].mean().item())
                        #metric['t2>rmse<t1'] = float(torch.pow(d - p.detach(), 2).sum(1)[m_.any(1)].mean().sqrt().item())
                    else:
                        metric['t2>|e|>t1'] = float(torch.abs(d[m_] - p.detach()[m_]).mean().item())
                        #metric['t2>rmse<t1'] = float(torch.pow(d[m_] - p.detach()[m_], 2).mean().sqrt().item())

                if self.t3 is not None:
                    m_ = mask == True
                    m_[d < self.t3] = False  # disable everything below the t3 threshold
                    if self.with_point_cloud:
                        metric['|e|>t3'] = float(
                            torch.pow(d - p.detach(), 2).sum(1).sqrt()[m_.any(1)].mean().item())
                        #metric['rmse>t3'] = float(
                        #    torch.pow(d - p.detach(), 2).sum(1)[m_.any(1)].mean().sqrt().item())
                    else:
                        metric['|e|>t3'] = float(torch.abs(d[m_] - p.detach()[m_]).mean().item())
                        #metric['rmse>t3'] = float(torch.pow(d[m_] - p.detach()[m_], 2).mean().sqrt().item())
                    

        else:
            metric = None


        return loss, metric



class JaccardLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(JaccardLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, label):
        num_classes = prediction.shape[1]
        # print('prediction inside the JaccardLoss', prediction)
        # print('labels inside the JaccardLoss', label)
        # input()
        print('prediction shape inside the JaccardLoss', prediction.shape)
        print('labels shape inside the JaccardLoss', label.shape)
        print('prediction min inside the JaccardLoss', prediction.min().item()) # 0.0
        print('prediction max inside the JaccardLoss', prediction.max().item())
        input()
        # if torch.isnan(prediction).any() or torch.isinf(prediction).any():
        #     print('NaNs or Infs found in predicted tensor')
        # input()
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1).to(label.device)[label.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(prediction)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes).to(label.device)[label.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = torch.softmax(prediction, dim=1)
        true_1_hot = true_1_hot.type(prediction.type())
        dims = (0,) + tuple(range(2, label.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        union = cardinality - intersection
        jacc_loss = (intersection / (union + self.eps)).mean()
        loss = (1 - jacc_loss)
        print('Jacc loss', jacc_loss)
        return loss



class DINOLoss(nn.Module):
    def __init__(self, nepochs, student_temp=0.1, center_momentum=0.9, out_dim=65536,  warmup_teacher_temp=0.04,
                 teacher_temp=0.04, warmup_teacher_temp_epochs=30, eval_loss=False):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.eval_loss = eval_loss

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        if self.center.device != teacher_output.device:
            self.center = self.center.to(teacher_output.device)

        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1).detach()

        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()

        if not self.eval_loss:
            self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

