import os
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
import glob
from dataset.data_augment import _resize_subtract_mean

class MPSGaze360(data.Dataset):
    def __init__(self, txt_paths, preproc=None, res_hr=2048):
        self.preproc = preproc
        self.res_hr = res_hr
        self.imgs_path = []
        self.words = []
        if not isinstance(txt_paths, list):
            txt_paths = [txt_paths]
        for txt_path in txt_paths:
            f = open(txt_path,'r')
            lines = f.readlines()
            isFirst = True
            labels = []
            for line in lines:
                line = line.rstrip()
                if line.startswith('#'):
                    if isFirst is True:
                        isFirst = False
                    else:
                        labels_copy = labels.copy()
                        self.words.append(labels_copy)
                        labels.clear()
                    path = line[2:]
                    path = txt_path.replace(os.path.basename(txt_path),'images/') + path
                    self.imgs_path.append(path)
                else:
                    line = line.split(' ')
                    label = [float(x) for x in line]
                    labels.append(label)

            self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        img_raw = img.copy()
        height, width, _ = img.shape
        if os.path.exists(self.imgs_path[index].replace('images', 'masks')):
            mask = cv2.imread(self.imgs_path[index].replace('images', 'masks'))
        else:
            mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            # print("mask not found: ", self.imgs_path[index].replace('images', 'masks'))

        labels = self.words[index]
        annotations = np.zeros((0, 36))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 36))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[6]    # l1_x
            annotation[0, 7] = label[7]    # l1_y
            annotation[0, 8] = label[8]    # l2_x
            annotation[0, 9] = label[9]    # l2_y
            annotation[0, 10] = label[10]  # l3_x
            annotation[0, 11] = label[11]  # l3_y
            annotation[0, 12] = label[12]  # l4_x
            annotation[0, 13] = label[13]  # l4_y
            annotation[0, 14] = label[14]  # l5_x
            annotation[0, 15] = label[15]  # l5_y
            annotation[0, 16] = label[16]  # l6_x
            annotation[0, 17] = label[17]  # l6_y
            annotation[0, 18] = label[18]  # l7_x
            annotation[0, 19] = label[19]  # l7_y
            annotation[0, 20] = label[20]
            annotation[0, 21] = label[21]
            annotation[0, 22] = label[22]
            annotation[0, 23] = label[23]
            annotation[0, 24] = label[24]
            annotation[0, 25] = label[25]
            annotation[0, 26] = label[26]
            annotation[0, 27] = label[27]
            
            annotation[0, 28] = label[31]  # gaze 1
            annotation[0, 29] = label[32]  # gaze 2
            annotation[0, 30] = label[33]  # gaze 3
            
            annotation[0, 31] = label[28]  # hdps 1
            annotation[0, 32] = label[29]  # hdps 2
            annotation[0, 33] = label[30]  # hdps 3

            annotation[0, 34] = label[34]  # distance
            
            annotation[0, 35] = 1          # label

            annotations = np.append(annotations, annotation, axis=0)

            if not os.path.exists(self.imgs_path[index].replace('images', 'masks')):
                cv2.circle(mask, (int((label[4]+label[6])/2), int((label[5]+label[7])/2)), 21, (255, 255, 255), -1)
                cv2.circle(mask, (int((label[8]+label[10])/2), int((label[9]+label[11])/2)), 21, (255, 255, 255), -1)
                cv2.circle(mask, (int((label[12]+label[14])/2), int((label[13]+label[15])/2)), 21, (255, 255, 255), -1)
                cv2.circle(mask, (int(label[16]), int(label[17])), 21, (255, 255, 255), -1)
                cv2.circle(mask, (int(label[18]), int(label[19])), 21, (255, 255, 255), -1)

        if not os.path.exists(self.imgs_path[index].replace('images', 'masks')):
            os.makedirs(os.path.dirname(self.imgs_path[index].replace('images', 'masks')), exist_ok=True)
            mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA)
            cv2.imwrite(self.imgs_path[index].replace('images', 'masks'), mask)
        
        target = np.array(annotations)

        # crop face hr images
        if glob.glob(self.imgs_path[index].replace('images', 'hr_images') + '/*'):
            hr_paths = glob.glob(self.imgs_path[index].replace('images', 'hr_images') + '/*')
            hr_paths.sort()
            hr_imgs = []
            hr_offsets = []
            for ti, hr_path in enumerate(hr_paths):
                hr_img = cv2.imread(hr_path)
                hr_img = _resize_subtract_mean(hr_img, self.res_hr//8, self.preproc.rgb_means)
                hr_imgs.append(hr_img)
                t = target[ti]
                cx = int((t[0] + t[2]) / 2)
                cy = int((t[1] + t[3]) / 2)
                w = int(t[2] - t[0])
                h = int(t[3] - t[1])
                mwh = max(w, h)
                x1 = int(cx - mwh / 2)
                y1 = int(cy - mwh / 2)
                x2 = int(cx + mwh / 2)
                y2 = int(cy + mwh / 2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                hr_offset = [x1, y1, x2, y2]
                hr_offsets.append(hr_offset)
            hr_imgs = np.array(hr_imgs)
            hr_offsets = np.array(hr_offsets)
        else:
            os.makedirs(os.path.dirname(self.imgs_path[index].replace('images', 'hr_images')), exist_ok=True)
            os.makedirs(self.imgs_path[index].replace('images', 'hr_images'), exist_ok=True)
            hr_imgs = []
            hr_offsets = []
            for ti, t in enumerate(target):
                cx = int((t[0] + t[2]) / 2)
                cy = int((t[1] + t[3]) / 2)
                w = int(t[2] - t[0])
                h = int(t[3] - t[1])
                mwh = max(w, h)
                x1 = int(cx - mwh / 2)
                y1 = int(cy - mwh / 2)
                x2 = int(cx + mwh / 2)
                y2 = int(cy + mwh / 2)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                hr_img = img_raw[y1:y2, x1:x2]
                hr_img = cv2.resize(hr_img, (self.res_hr//8, self.res_hr//8), interpolation=cv2.INTER_AREA)
                cv2.imwrite(self.imgs_path[index].replace('images', 'hr_images') + '/' + f"{ti:02d}" + '.jpg', hr_img)
                hr_img = _resize_subtract_mean(hr_img, self.res_hr//8, self.preproc.rgb_means)
                hr_imgs.append(hr_img)
                hr_offset = [x1, y1, x2, y2]
                hr_offsets.append(hr_offset)
            hr_imgs = np.array(hr_imgs)
            hr_offsets = np.array(hr_offsets)

        hr_imgs = np.concatenate((hr_imgs, np.zeros((9 - len(hr_imgs), 3, self.res_hr//8, self.res_hr//8))), axis=0).astype(np.float32)
        hr_offsets = np.concatenate((hr_offsets, np.zeros((9 - len(hr_offsets), 4))), axis=0).astype(np.float32)

        hr_full_img = _resize_subtract_mean(img_raw, self.res_hr, self.preproc.rgb_means)

        if self.preproc is not None:
            img, target = self.preproc(img, target)
        mask = mask[..., 0].astype(np.float32) / 255.0

        return torch.from_numpy(img), torch.from_numpy(hr_imgs), torch.from_numpy(mask), target, torch.from_numpy(hr_offsets), torch.from_numpy(hr_full_img)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    imgs_hr = []
    masks = []
    offsets_hr = []
    imgs_full_hr = []
    for i, sample in enumerate(batch):
        imgs.append(sample[0])
        imgs_hr.append(sample[1])
        masks.append(sample[2])
        targets.append(torch.from_numpy(sample[3]).float())
        offsets_hr.append(torch.concat((torch.ones((len(sample[4]), 1)) * i, sample[4]/2048), 1))
        imgs_full_hr.append(sample[5])

    imgs_hr = torch.stack(imgs_hr, 0)
    offsets_hr = torch.stack(offsets_hr, 0)
    max_l = 0
    for j in range(len(offsets_hr)):
        l = (offsets_hr[j][:, 1] > 0).sum()
        max_l = max(max_l, l)
    imgs_hr = imgs_hr[:, :max_l+1, :, :]
    offsets_hr = offsets_hr[:, :max_l+1, :]

    return (torch.stack(imgs, 0), imgs_hr, torch.stack(masks, 0), targets, offsets_hr, torch.stack(imgs_full_hr, 0))
