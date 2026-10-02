import os
import random

import albumentations as A
import cv2
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class PolypDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) # H, W, C
        mask = np.array(mask) # H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        img = img/255 # C, H, W
        mask = (mask > 100).float()# H, W
                
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch


class FundusDataset(Dataset):
    def __init__(self, args, data_path, image_list, transform=None, is_train = False):

        self.data_path = data_path
        self.images_list = image_list
        self.transform = transform
        self.args = args
        
    def __len__(self):

        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.images_list[idx])
        img = Image.open(img_path).convert('RGB')
        label_path = img_path.replace('Images', 'Labels')
        mask = Image.open(label_path).convert('L')

        img = np.array(img)
        mask = np.array(mask)
        
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']

        img = img/255 # C, H, W
        mask = (mask > 100).float()# H, W
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_path
                 }
            
        return batch


class ProstateDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks

    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) #C H, W
        mask = np.array(mask) #H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']
            
        img = img/255#C H, W
        mask = (mask > 0).float() #H, W
        
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch


class UltrasoundDataset(Dataset):
    def __init__(self, args, images, masks, img_name_list, transform=None, is_train = False):
        
        self.args = args
        self.img_name_list = img_name_list
        self.transform = transform
        self.images = images
        self.masks = masks


    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        if self.img_name_list is not None:
            img_name = self.img_name_list[idx] #(clientname. imgname)
        img = np.array(img).astype(np.float32) #C H, W
        mask = np.array(mask) # H, W
        if self.transform is not None:
            data = {'image': img, 'mask': mask}
            augmented = self.transform(**data)
            img, mask = augmented['image'], augmented['mask']
        
        img = img/255 # C, H, W
        mask = (mask > 100).float()# H, W
                
        batch = {'img':img,
                 'mask': mask,
                 'img_name':img_name
                 }
            
        return batch


def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class Data(object):
    def __init__(self, args):
        self.args = args
        if args.dataset == 'Fundus':
            transform_train = A.Compose([
                        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(256, 256, interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])
    
            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                train_data_path = os.path.join(base_dir, client_name, 'Train/Original/Images')
                train_images_list = os.listdir(train_data_path)
                train_images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(train_images_list)
                train_len = int(len(train_images_list)*0.8)
                train_filenames = train_images_list[:train_len]
                val_filenames = train_images_list[train_len:]
                
                train_datasets = FundusDataset(args, train_data_path, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', len(train_filenames))
                
                val_datasets = FundusDataset(args, train_data_path, val_filenames, transform=transform_train, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', len(val_filenames))
                
                test_data_path = os.path.join(base_dir, client_name, 'Test/Original/Images')
                test_images_list = os.listdir(test_data_path)
                test_images_list.sort()
                test_datasets = FundusDataset(args, test_data_path, test_images_list, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, ' test', len(test_images_list))
                
            self.g_test_loaders = []
            self.g_val_loaders = []
            for client_name in args.g_client_names:
                test_data_path = os.path.join(base_dir, client_name, 'Train/Original/Images')
                test_filenames = os.listdir(test_data_path)

                test_datasets = FundusDataset(args, test_data_path, test_filenames, transform=transform_test, is_train = False)
                

                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=2, batch_size=self.args.batchsize, shuffle=False)
                self.g_test_loaders.append(test_loader)
                print(client_name, 'g_test', len(test_filenames))
                
                val_data_path = os.path.join(base_dir, client_name, 'Test/Original/Images')
                val_images_list = os.listdir(val_data_path)
                val_images_list.sort()
                val_images_list = val_images_list[:int(len(val_images_list)*1*args.data_size)]



                val_datasets = FundusDataset(args, val_data_path, val_images_list, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.g_val_loaders.append(val_loader)
                print(client_name, ' g_val', len(val_images_list))
                
        elif args.dataset == 'Polyp':
            import tifffile
            input_size = (256, 256)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])
    
            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'images'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                train_len = int(len(images_list)*0.7)
                val_len = int(len(images_list)*0.1)
                train_filenames = images_list[:train_len]
                val_filenames = images_list[train_len:train_len+val_len]
                test_filenames = images_list[train_len+val_len:]
                
                images, labels = [], []
                for filename in train_filenames:
                    if client_name == 'CVC-ClinicDB':
                        img = tifffile.imread(os.path.join(data_path, 'images', filename))
                        mask = tifffile.imread(os.path.join(data_path, 'masks', filename))
                    else:
                        img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                        img = np.array(img)
                        mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                        mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)

                train_datasets = PolypDataset(args, images, labels, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:',len(images))
                
                images, labels = [], []
                for filename in val_filenames:
                    if client_name == 'CVC-ClinicDB':
                        img = tifffile.imread(os.path.join(data_path, 'images', filename))
                        mask = tifffile.imread(os.path.join(data_path, 'masks', filename))
                    else:
                        img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                        img = np.array(img)
                        mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                        mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
 
                val_datasets = PolypDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:',len(images))

                
                images, labels = [], []
                for filename in test_filenames:
                    if client_name == 'CVC-ClinicDB':
                        img = tifffile.imread(os.path.join(data_path, 'images', filename))
                        mask = tifffile.imread(os.path.join(data_path, 'masks', filename))
                    else:
                        img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                        img = np.array(img)
                        mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                        mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                
                test_datasets = PolypDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:',len(images))
                
                
            self.g_test_loaders = []
            self.g_val_loaders = []
            for client_name in args.g_client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'images'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                test_len = int(len(images_list)*0.5)
                test_filenames = images_list[:test_len]
                
                val_len = int(len(images_list)*args.val_percent*args.data_size)
                val_filenames = images_list[test_len:test_len+val_len]
                
                images, labels = [], []
                for filename in val_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                val_datasets = PolypDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.g_val_loaders.append(val_loader)
                print(client_name, 'g_val', 'patients:',len(images))


                images, labels = [], []
                for filename in test_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                test_datasets = PolypDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.g_test_loaders.append(test_loader)
                print(client_name, 'g_test', 'patients:',len(images))

                
        elif args.dataset == 'Prostate':
            input_size = (256, 256)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])
    
            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(data_path)
                patients = [img_name[:6] for img_name in images_list]
                patients= list(set(patients))
                patients.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(patients)
                train_len = int(len(patients)*0.7)
                val_len = int(len(patients)*0.1)
                train_patients = patients[:train_len]
                val_patients = patients[train_len:train_len+val_len]
                test_patients = patients[train_len+val_len:]
                
                images, labels, slice_id = [], [], []
                for patient_id in train_patients:
                    imgdir = os.path.join(data_path, patient_id + ".nii.gz")
                    image_v = sitk.ReadImage(imgdir)
                    if client_name == 'BMC':
                        maskdir = os.path.join(data_path, patient_id + "_Segmentation.nii.gz")
                    else:
                        maskdir = os.path.join(data_path, patient_id + "_segmentation.nii.gz")
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    image_v = convert_from_nii_to_png(image_v)
                    #print(label_v.shape, image_v.shape)
                    
                    image_v = image_v[:, 192-128:192+128, 192-128:192+128]
                    label_v = label_v[:, 192-128:192+128, 192-128:192+128]
                    
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])
                        
                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, patient_id, i])
                
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                train_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:', len(train_patients) ,len(images))
                
                images, labels, slice_id = [], [], []
                for patient_id in val_patients:
                    imgdir = os.path.join(data_path, patient_id + ".nii.gz")
                    image_v = sitk.ReadImage(imgdir)
                    if client_name == 'BMC':
                        maskdir = os.path.join(data_path, patient_id + "_Segmentation.nii.gz")
                    else:
                        maskdir = os.path.join(data_path, patient_id + "_segmentation.nii.gz")
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    image_v = convert_from_nii_to_png(image_v)
                    
                    image_v = image_v[:, 192-128:192+128, 192-128:192+128]
                    label_v = label_v[:, 192-128:192+128, 192-128:192+128]
                
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])
                        
                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, patient_id, i])
                
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                val_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=2, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:', len(val_patients) ,len(images))

                images, labels, slice_id = [], [], []
                for patient_id in test_patients:
                    imgdir = os.path.join(data_path, patient_id + ".nii.gz")
                    image_v = sitk.ReadImage(imgdir)
                    if client_name == 'BMC':
                        maskdir = os.path.join(data_path, patient_id + "_Segmentation.nii.gz")
                    else:
                        maskdir = os.path.join(data_path, patient_id + "_segmentation.nii.gz")
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    image_v = convert_from_nii_to_png(image_v)
                
                    image_v = image_v[:, 192-128:192+128, 192-128:192+128]
                    label_v = label_v[:, 192-128:192+128, 192-128:192+128]
                    
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])
                        
                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, patient_id, i])
                
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                test_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=2, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:', len(test_patients) ,len(images))


            self.g_test_loaders = []
            self.g_val_loaders = []
            for client_name in args.g_client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'imagesTr'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)

                test_len = int(len(images_list)*0.5)
                test_filenames = images_list[:test_len]
                
                val_len = int(len(images_list)*args.val_percent*args.data_size)
                val_filenames = images_list[test_len:test_len+val_len]
                
                images, labels, slice_id = [], [], []
                for filename in val_filenames:
                    imgdir = os.path.join(data_path,'imagesTr', filename)
                    image_v = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path,'labelsTr', filename)
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    if len(image_v.shape)==4:
                        print(filename)
                        image_v = image_v[0,:,:,:]
                    image_v = convert_from_nii_to_png(image_v)
                    d,w,h = image_v.shape
                    image_v = image_v[:, w//2-128:w//2+128, h//2-128:h//2+128]
                    label_v = label_v[:, w//2-128:w//2+128, h//2-128:h//2+128]
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])

                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, filename, i])
                    
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                val_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.g_val_loaders.append(val_loader)
                print(client_name, 'g_val', 'patients:', len(val_filenames) ,len(images))

                images, labels, slice_id = [], [], []
                for filename in test_filenames:
                    imgdir = os.path.join(data_path,'imagesTr', filename)
                    image_v = sitk.ReadImage(imgdir)
                    maskdir = os.path.join(data_path,'labelsTr', filename)
                    label_v = sitk.ReadImage(maskdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    if len(image_v.shape)==4:
                        print(filename)
                        image_v = image_v[0,:,:,:]
                    image_v = convert_from_nii_to_png(image_v)
                
                    d,w,h = image_v.shape
                    image_v = image_v[:, w//2-128:w//2+128, h//2-128:h//2+128]
                    label_v = label_v[:, w//2-128:w//2+128, h//2-128:h//2+128]
                    
                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if (np.all(label == 0)):
                            continue
                        image = np.array(image_v[i, :, :])
                        
                        image = image[:,:, None]
                        image = np.repeat(image, 3, axis=2)
                        image = Image.fromarray(np.uint8(image))
                        image = image.resize(input_size, Image.NEAREST) # H, W, C
                        
                        label = Image.fromarray(label)
                        label = label.resize(input_size, Image.NEAREST) # H, W, C
                        
                        labels.append(label)
                        images.append(image)
                        slice_id.append([client_name, filename, i])
                    
                labels = np.array(labels).astype(int)
                images = np.array(images) # N, H, W, 3
                
                test_datasets = ProstateDataset(args, images, labels, slice_id, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.g_test_loaders.append(test_loader)
                print(client_name, 'g_test', 'patients:', len(test_filenames) ,len(images))


        elif args.dataset == 'FL_Breast_Ultrasound':
            
            input_size = (256, 256)
            transform_train = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        ToTensorV2(p=1.0)])
    
            transform_test = A.Compose([
                        A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST),
                        ToTensorV2(p=1.0)])

            client_names = args.client_names
            base_dir = args.data_path
            self.train_loaders = []
            self.val_loaders = []
            self.test_loaders = []
            for client_name in client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'images'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                train_len = int(len(images_list)*0.7)
                val_len = int(len(images_list)*0.1)
                train_filenames = images_list[:train_len]
                val_filenames = images_list[train_len:train_len+val_len]
                test_filenames = images_list[train_len+val_len:]
                
                images, labels = [], []
                for filename in train_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)

                train_datasets = UltrasoundDataset(args, images, labels, train_filenames, transform=transform_train, is_train = True)
                train_loader = torch.utils.data.DataLoader(train_datasets,num_workers=4, batch_size=self.args.batchsize, shuffle=True)
                self.train_loaders.append(train_loader)
                print(client_name, 'train', 'patients:',len(images))
                
                images, labels = [], []
                for filename in val_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
 
                val_datasets = UltrasoundDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.val_loaders.append(val_loader)
                print(client_name, 'val', 'patients:',len(images))
                
                images, labels = [], []
                for filename in test_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                
                test_datasets = UltrasoundDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.test_loaders.append(test_loader)
                print(client_name, 'test', 'patients:',len(images))

            self.g_test_loaders = []
            self.g_val_loaders = []
            for client_name in args.g_client_names:
                data_path = os.path.join(base_dir, client_name)
                images_list = os.listdir(os.path.join(data_path, 'images'))
                images_list.sort()
                np.random.seed(args.random_seed)
                random.seed(args.random_seed)
                np.random.shuffle(images_list)
                
                test_len = int(len(images_list)*0.5)
                test_filenames = images_list[:test_len]
                
                val_len = int(len(images_list)*args.val_percent)
                val_filenames = images_list[test_len:test_len+val_len]
                
                images, labels = [], []
                for filename in val_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                val_datasets = PolypDataset(args, images, labels, val_filenames, transform=transform_test, is_train = False)
                val_loader = torch.utils.data.DataLoader(val_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.g_val_loaders.append(val_loader)
                print(client_name, 'g_val', 'patients:',len(images))


                images, labels = [], []
                for filename in test_filenames:
                    img = Image.open(os.path.join(data_path, 'images', filename)).convert('RGB')
                    img = np.array(img)
                    mask = Image.open(os.path.join(data_path, 'masks', filename)).convert('L')
                    mask = np.array(mask)
                    images.append(img)
                    labels.append(mask)
                test_datasets = PolypDataset(args, images, labels, test_filenames, transform=transform_test, is_train = False)
                test_loader = torch.utils.data.DataLoader(test_datasets,num_workers=1, batch_size=1, shuffle=False)
                self.g_test_loaders.append(test_loader)
                print(client_name, 'g_test', 'patients:',len(images))

